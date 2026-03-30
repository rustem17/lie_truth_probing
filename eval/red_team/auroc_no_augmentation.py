"""
Red team diagnostic: AUROC without (d, -d) augmentation.

The standard pipeline computes AUROC on augmented data: scores_aug = [scores, -scores],
labels_aug = [1, 0]. This is trivially symmetric — scoring a vector and its negation
with opposite labels inflates AUROC. This script reports side-by-side: augmented AUROC
vs real AUROC on individual samples (lie=1, truth=0) scored by dot product with the
probe direction.

Augmented AUROC: computed on pair diffs with (d, -d) trick (as in existing pipeline).
Real AUROC: computed on individual activations (not pairs), labels from condition.
    score_i = activation_i[layer] @ direction
    label_i = 1 if lie else 0
    auroc = roc_auc_score(labels, scores)

Usage:
    python auroc_no_augmentation.py --model_tag gemma3-27b
    python auroc_no_augmentation.py --model_tag gemma3-27b --methods mass_mean,contrastive,mahalanobis_lda
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import TRAIN_DATASETS


def get_pair_diffs(activations, data, label_map):
    by_id = defaultdict(dict)
    for i, s in enumerate(data):
        is_lie = label_map[s["condition"]] == 1
        base_id = s["id"].rsplit("_", 1)[0] if s["id"].endswith(("_lie", "_truth")) else s["id"]
        by_id[base_id]["lie" if is_lie else "truth"] = i
    pair_ids, diffs = [], []
    for sid in sorted(by_id):
        pair = by_id[sid]
        if "lie" not in pair or "truth" not in pair:
            continue
        diffs.append(activations[pair["lie"]] - activations[pair["truth"]])
        pair_ids.append(sid)
    return np.stack(diffs), pair_ids


def run(model_tag,
        act_dir=None,
        data_dir="../..",
        probes_dir="../..",
        methods="mass_mean,contrastive,mahalanobis_lda",
        n_splits=5,
        output_dir="results"):
    if act_dir is None:
        act_dir = f"../../activations_{model_tag}"

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    method_list = [m.strip() for m in methods.split(",")]
    rows = []

    for method in method_list:
        for name, (filename, label_map) in TRAIN_DATASETS.items():
            probe_path = Path(probes_dir) / "probes" / method / f"{name}_probe_{model_tag}.pt"
            if not probe_path.exists():
                probe_path = Path(probes_dir) / "probes" / method / f"{name}_probe.pt"
            if not probe_path.exists():
                continue

            act_path = Path(act_dir) / f"{name}.pt"
            data_path = Path(data_dir) / filename
            if not act_path.exists() or not data_path.exists():
                continue

            probe = torch.load(probe_path, weights_only=False)
            saved = torch.load(act_path, weights_only=False)
            data = json.load(open(data_path))[:len(saved["activations"])]
            activations = saved["activations"]
            labels = np.array([label_map[s["condition"]] for s in data])
            pair_diffs, _ = get_pair_diffs(activations, data, label_map)

            best_layer = probe.get("best_layer", 1)
            best_idx = best_layer - 1
            D = pair_diffs[:, best_idx]
            n_pairs = D.shape[0]

            if "all_directions" in probe:
                direction = probe["all_directions"][best_idx]
            else:
                direction = probe["direction"]
            direction = direction / np.linalg.norm(direction)

            scores_aug = np.concatenate([D @ direction, -(D @ direction)])
            labels_aug = np.concatenate([np.ones(n_pairs), np.zeros(n_pairs)])
            auroc_aug = float(roc_auc_score(labels_aug, scores_aug))

            sample_scores = activations[:, best_idx] @ direction
            auroc_real = float(roc_auc_score(labels, sample_scores))

            delta = auroc_aug - auroc_real
            frac_positive = float((D @ direction > 0).mean())

            print(f"{method}/{name}: augmented={auroc_aug:.4f}, real={auroc_real:.4f}, inflation={delta:+.4f}, frac_positive={frac_positive:.3f}")
            rows.append({
                "method": method, "dataset": name,
                "auroc_augmented": auroc_aug, "auroc_real": auroc_real,
                "delta": delta, "frac_positive": frac_positive,
                "best_layer": best_layer, "n_pairs": n_pairs,
                "n_samples": len(labels),
            })

    if not rows:
        print("No probes found.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(output_dir) / f"auroc_no_augmentation_{model_tag}_{timestamp}.csv"
    with open(csv_path, "w") as f:
        f.write("method,dataset,auroc_augmented,auroc_real,delta,frac_positive,best_layer,n_pairs,n_samples\n")
        for r in rows:
            f.write(f"{r['method']},{r['dataset']},{r['auroc_augmented']:.4f},{r['auroc_real']:.4f},{r['delta']:+.4f},{r['frac_positive']:.3f},{r['best_layer']},{r['n_pairs']},{r['n_samples']}\n")
    print(f"\nCSV: {csv_path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")

    datasets = sorted(set(r["dataset"] for r in rows))
    methods_found = sorted(set(r["method"] for r in rows))

    fig, axes = plt.subplots(1, len(methods_found), figsize=(6 * len(methods_found), 5), squeeze=False)

    for col, method in enumerate(methods_found):
        ax = axes[0][col]
        subset = [r for r in rows if r["method"] == method]
        ds = [r["dataset"] for r in subset]
        aug = [r["auroc_augmented"] for r in subset]
        real = [r["auroc_real"] for r in subset]
        x = np.arange(len(ds))

        ax.bar(x - 0.15, aug, 0.3, label="augmented (d,-d)", color="#D68F73")
        ax.bar(x + 0.15, real, 0.3, label="real (samples)", color="#729ECE")
        ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(ds, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("AUROC")
        ax.set_title(f"{method}")
        ax.legend(fontsize=8)
        ax.set_ylim(0.4, 1.0)

    fig.suptitle(f"AUROC: augmented vs real — {model_tag}", fontsize=12)
    fig.tight_layout()
    plot_path = Path(output_dir) / f"auroc_no_augmentation_{model_tag}_{timestamp}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    fire.Fire(run)

"""
Red team diagnostic: random direction baseline.

Establishes null AUROC distribution by scoring all datasets with random unit
directions. Reports mean, std, 95th percentile of null distribution and
z-score of the trained probe relative to the null. Augmented AUROC uses
pair diffs with (d,-d) trick; real AUROC uses individual sample activations
with condition labels.

Usage:
    python random_baseline.py --model_tag gemma3-27b --probe_path ../../probes/mass_mean/shared_direction_gemma3-27b.pt
    python random_baseline.py --model_tag gemma3-27b --probe_path ../../probes/contrastive/shared_direction_gemma3-27b.pt --n_random 5000
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
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


def auroc_augmented(scores):
    n = len(scores)
    scores_all = np.concatenate([scores, -scores])
    labels_all = np.concatenate([np.ones(n), np.zeros(n)])
    return float(roc_auc_score(labels_all, scores_all))


def run(model_tag,
        probe_path,
        act_dir=None,
        data_dir="../..",
        n_random=1000,
        output_dir="results"):
    if act_dir is None:
        act_dir = f"../../activations_{model_tag}"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    probe = torch.load(probe_path, weights_only=False)
    if "all_directions" in probe:
        best_layer = probe.get("best_layer", probe.get("best_layer_transfer", 1))
        best_idx = best_layer - 1
        trained_dir = probe["all_directions"][best_idx]
    else:
        trained_dir = probe["direction"]
        best_layer = probe.get("best_layer", 1)
        best_idx = best_layer - 1
    trained_dir = trained_dir / np.linalg.norm(trained_dir)
    hidden_dim = trained_dir.shape[0]
    probe_name = Path(probe_path).stem

    rng = np.random.RandomState(42)
    random_dirs = rng.randn(n_random, hidden_dim).astype(np.float32)
    random_dirs = random_dirs / np.linalg.norm(random_dirs, axis=1, keepdims=True)

    rows = []
    null_distributions = {}

    for name, (filename, label_map) in TRAIN_DATASETS.items():
        act_path = Path(act_dir) / f"{name}.pt"
        data_path = Path(data_dir) / filename
        if not act_path.exists() or not data_path.exists():
            print(f"Skipping {name}: missing files")
            continue

        saved = torch.load(act_path, weights_only=False)
        data = json.load(open(data_path))[:len(saved["activations"])]
        pair_diffs, _ = get_pair_diffs(saved["activations"], data, label_map)
        D = pair_diffs[:, best_idx]
        labels = np.array([label_map[s["condition"]] for s in data])
        activations_layer = saved["activations"][:, best_idx]

        trained_aug = auroc_augmented(D @ trained_dir)
        trained_real = float(roc_auc_score(labels, activations_layer @ trained_dir))

        null_augs = np.empty(n_random)
        null_reals = np.empty(n_random)
        pair_scores = D @ random_dirs.T
        sample_scores = activations_layer @ random_dirs.T
        for j in range(n_random):
            null_augs[j] = auroc_augmented(pair_scores[:, j])
            null_reals[j] = float(roc_auc_score(labels, sample_scores[:, j]))

        z_aug = (trained_aug - null_augs.mean()) / max(null_augs.std(), 1e-10)
        z_real = (trained_real - null_reals.mean()) / max(null_reals.std(), 1e-10)

        print(f"{name}: trained_aug={trained_aug:.4f}, null_aug={null_augs.mean():.4f}+/-{null_augs.std():.4f}, z={z_aug:.2f}")
        print(f"  trained_real={trained_real:.4f}, null_real={null_reals.mean():.4f}+/-{null_reals.std():.4f}, z={z_real:.2f}")

        rows.append({
            "dataset": name,
            "trained_auroc_aug": trained_aug, "trained_auroc_real": trained_real,
            "null_mean_aug": float(null_augs.mean()), "null_std_aug": float(null_augs.std()),
            "null_p95_aug": float(np.percentile(null_augs, 95)),
            "null_p99_aug": float(np.percentile(null_augs, 99)),
            "z_score_aug": z_aug,
            "null_mean_real": float(null_reals.mean()), "null_std_real": float(null_reals.std()),
            "null_p95_real": float(np.percentile(null_reals, 95)),
            "z_score_real": z_real,
            "n_pairs": D.shape[0],
        })
        null_distributions[name] = {"aug": null_augs, "real": null_reals}

    if not rows:
        print("No datasets processed.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(output_dir) / f"random_baseline_{model_tag}_{timestamp}.csv"
    with open(csv_path, "w") as f:
        f.write("dataset,trained_auroc_aug,null_mean_aug,null_std_aug,null_p95_aug,null_p99_aug,z_score_aug,trained_auroc_real,null_mean_real,null_std_real,null_p95_real,z_score_real,n_pairs\n")
        for r in rows:
            f.write(f"{r['dataset']},{r['trained_auroc_aug']:.4f},{r['null_mean_aug']:.4f},{r['null_std_aug']:.4f},{r['null_p95_aug']:.4f},{r['null_p99_aug']:.4f},{r['z_score_aug']:.2f},{r['trained_auroc_real']:.4f},{r['null_mean_real']:.4f},{r['null_std_real']:.4f},{r['null_p95_real']:.4f},{r['z_score_real']:.2f},{r['n_pairs']}\n")
    print(f"\nCSV: {csv_path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")

    n_datasets = len(null_distributions)
    fig, axes = plt.subplots(2, n_datasets, figsize=(4 * n_datasets, 7), squeeze=False)

    for col, (name, nd) in enumerate(null_distributions.items()):
        r = rows[col]
        for row_idx, (metric, key, trained_val) in enumerate([
            ("augmented", "aug", r["trained_auroc_aug"]),
            ("real", "real", r["trained_auroc_real"]),
        ]):
            ax = axes[row_idx][col]
            ax.hist(nd[key], bins=50, color="#B0B0B0", alpha=0.7, density=True)
            ax.axvline(trained_val, color="#D62728", linewidth=2, label=f"trained={trained_val:.3f}")
            ax.axvline(r[f"null_p95_{key}"], color="#FF7F0E", linewidth=1, linestyle="--", label=f"p95={r[f'null_p95_{key}']:.3f}")
            ax.set_title(f"{name} ({metric})\nz={r[f'z_score_{key}']:.1f}", fontsize=9)
            ax.set_xlabel("AUROC")
            ax.legend(fontsize=7)

    fig.suptitle(f"Null AUROC distribution ({n_random} random dirs) — {probe_name} L{best_layer}", fontsize=11)
    fig.tight_layout()
    plot_path = Path(output_dir) / f"random_baseline_{model_tag}_{timestamp}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    fire.Fire(run)

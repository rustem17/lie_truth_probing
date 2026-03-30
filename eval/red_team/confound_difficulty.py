"""
Red team diagnostic: difficulty confound in spontaneous condition.

Correlates probe score with question difficulty (n_correct / n_runs from
multi-run inference). High correlation indicates the probe detects model
uncertainty/difficulty rather than deception.

Prerequisites:
    - Trained probe .pt file
    - Extracted activations for spontaneous condition
    - multi_results_{model_tag}.json from generate_datasets/spontaneous/infer.py

Usage:
    python confound_difficulty.py \
        --model_tag gemma3-27b \
        --probe_path ../../probes/mass_mean/shared_direction_gemma3-27b.pt \
        --multi_results ../../generate_datasets/spontaneous/multi_results_gemma3-27b.json
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from scipy.stats import spearmanr
import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import TRAIN_DATASETS, dataset_filename


def run(model_tag,
        probe_path,
        multi_results,
        act_dir=None,
        data_dir="../..",
        output_dir="results"):
    if act_dir is None:
        act_dir = f"../../activations_{model_tag}"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    probe = torch.load(probe_path, weights_only=False)
    if "all_directions" in probe:
        best_layer = probe.get("best_layer", probe.get("best_layer_transfer", 1))
        best_idx = best_layer - 1
        direction = probe["all_directions"][best_idx]
    else:
        direction = probe["direction"]
        best_layer = probe.get("best_layer", 1)
        best_idx = best_layer - 1
    direction = direction / np.linalg.norm(direction)

    mr = json.load(open(multi_results))
    difficulty_map = {}
    for s in mr:
        sid = s.get("id", s.get("question_id", ""))
        n_runs = s.get("n_runs", 10)
        n_correct = sum(1 for j in s.get("judge_labels", s.get("parsed_labels", [])) if j)
        difficulty_map[sid] = n_correct / n_runs

    results_all = []
    for ds_key in ["spontaneous", "spontaneous_inconsistent"]:
        if ds_key not in TRAIN_DATASETS:
            continue
        filename, label_map = TRAIN_DATASETS[ds_key]
        act_path = Path(act_dir) / f"{ds_key}.pt"
        fname = dataset_filename(filename, model_tag) if model_tag else filename
        data_path = Path(data_dir) / fname
        if not data_path.exists():
            data_path = Path(data_dir) / filename
        if not act_path.exists() or not data_path.exists():
            print(f"Skipping {ds_key}: missing files")
            continue

        saved = torch.load(act_path, weights_only=False)
        data = json.load(open(data_path))[:len(saved["activations"])]
        activations = saved["activations"]

        scores = activations[:, best_idx] @ direction
        labels = np.array([label_map[s["condition"]] for s in data])

        difficulties, sample_scores, sample_labels, sample_ids = [], [], [], []
        for i, s in enumerate(data):
            base_id = s["id"].rsplit("_", 1)[0] if s["id"].endswith(("_lie", "_truth")) else s["id"]
            sid = s.get("source_id", base_id)
            d = difficulty_map.get(sid, difficulty_map.get(base_id, None))
            if d is None:
                continue
            difficulties.append(d)
            sample_scores.append(float(scores[i]))
            sample_labels.append(int(labels[i]))
            sample_ids.append(s["id"])

        difficulties = np.array(difficulties)
        sample_scores = np.array(sample_scores)
        sample_labels = np.array(sample_labels)

        rho, pval = spearmanr(difficulties, sample_scores)
        lie_mask = sample_labels == 1
        rho_lie, p_lie = spearmanr(difficulties[lie_mask], sample_scores[lie_mask]) if lie_mask.sum() > 2 else (float("nan"), float("nan"))
        rho_truth, p_truth = spearmanr(difficulties[~lie_mask], sample_scores[~lie_mask]) if (~lie_mask).sum() > 2 else (float("nan"), float("nan"))

        print(f"{ds_key}: rho={rho:.4f} (p={pval:.2e}), lie_rho={rho_lie:.4f}, truth_rho={rho_truth:.4f}, matched={len(difficulties)}/{len(data)}")

        for sid, d, sc, lab in zip(sample_ids, difficulties, sample_scores, sample_labels):
            results_all.append({"dataset": ds_key, "question_id": sid, "difficulty": float(d), "probe_score": float(sc), "label": int(lab)})

    if not results_all:
        print("No data matched.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = Path(output_dir) / f"confound_difficulty_{model_tag}_{timestamp}.csv"
    with open(csv_path, "w") as f:
        f.write("dataset,question_id,difficulty,probe_score,label\n")
        for r in results_all:
            f.write(f"{r['dataset']},{r['question_id']},{r['difficulty']:.3f},{r['probe_score']:.6f},{r['label']}\n")
    print(f"\nCSV: {csv_path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")

    ds_names = sorted(set(r["dataset"] for r in results_all))
    fig, axes = plt.subplots(1, len(ds_names), figsize=(6 * len(ds_names), 5), squeeze=False)

    for col, ds_key in enumerate(ds_names):
        ax = axes[0][col]
        subset = [r for r in results_all if r["dataset"] == ds_key]
        diffs = np.array([r["difficulty"] for r in subset])
        scs = np.array([r["probe_score"] for r in subset])
        labs = np.array([r["label"] for r in subset])

        ax.scatter(diffs[labs == 0], scs[labs == 0], alpha=0.3, s=12, color="#729ECE", label="truth")
        ax.scatter(diffs[labs == 1], scs[labs == 1], alpha=0.3, s=12, color="#D68F73", label="lie")
        z = np.polyfit(diffs, scs, 1)
        xline = np.linspace(diffs.min(), diffs.max(), 100)
        ax.plot(xline, np.polyval(z, xline), color="black", linewidth=1, linestyle="--")

        rho, pval = spearmanr(diffs, scs)
        ax.set_title(f"{ds_key}  rho={rho:.3f} (p={pval:.1e})", fontsize=10)
        ax.set_xlabel("difficulty (n_correct / n_runs)")
        ax.set_ylabel("probe score")
        ax.legend(fontsize=8)

    fig.suptitle(f"Difficulty vs probe score — L{best_layer}", fontsize=12)
    fig.tight_layout()
    plot_path = Path(output_dir) / f"confound_difficulty_{model_tag}_{timestamp}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    fire.Fire(run)

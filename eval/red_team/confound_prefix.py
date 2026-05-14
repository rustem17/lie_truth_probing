"""
Red team diagnostic: prefix confound test.

Tests whether probe signal comes from the prefix (system prompt + user message)
rather than the assistant response. Trains mass-mean directions on pair diffs,
evaluates per-sample AUROC on activations extracted at last_user position
(last token before assistant response) vs standard response-position.

If prefix AUROC >> 0.5, the probe may be detecting prompt framing differences
between lie/truth conditions, not internal deception state.

Prerequisites:
    python extract_activations.py --model <tag> --position last_user
    python extract_activations.py --model <tag>  # default position (first)

Usage:
    python confound_prefix.py --model_tag llama-3-3-70b-instruct
    python confound_prefix.py --model_tag llama-3-3-70b-instruct --prefix_act_dir ../../activations_llama-3-3-70b-instruct_last_user --response_act_dir ../../activations_llama-3-3-70b-instruct
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
from config import (
    DEFAULT_MODEL_TAG,
    TRAIN_DATASETS,
    activation_dirname,
    resolve_dataset_path_for_activation,
    resolve_model,
    validate_dataset_provenance,
)


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


def sample_auroc(activations_layer, labels, pair_diffs_layer):
    direction = pair_diffs_layer.mean(axis=0)
    direction = direction / np.linalg.norm(direction)
    return float(roc_auc_score(labels, activations_layer @ direction))


def run(model_tag=DEFAULT_MODEL_TAG,
        prefix_act_dir=None,
        response_act_dir=None,
        data_dir="../..",
        output_dir="results"):
    model_tag, _ = resolve_model(model_tag)
    if prefix_act_dir is None:
        prefix_act_dir = f"../../{activation_dirname(model_tag, 'last_user')}"
    if response_act_dir is None:
        response_act_dir = f"../../{activation_dirname(model_tag)}"

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    rows = []

    for name, (filename, label_map) in TRAIN_DATASETS.items():
        prefix_path = Path(prefix_act_dir) / f"{name}.pt"
        response_path = Path(response_act_dir) / f"{name}.pt"
        if not prefix_path.exists():
            print(f"Skipping {name}: {prefix_path} not found")
            continue
        if not response_path.exists():
            print(f"Skipping {name}: {response_path} not found")
            continue

        prefix_saved = torch.load(prefix_path, weights_only=False)
        response_saved = torch.load(response_path, weights_only=False)
        data_path = resolve_dataset_path_for_activation(
            data_dir, filename, prefix_saved.get("model_tag", model_tag), prefix_saved
        )
        if not data_path.exists():
            print(f"Skipping {name}: {data_path} not found")
            continue
        data = json.load(open(data_path))[:len(prefix_saved["activations"])]
        validate_dataset_provenance(prefix_saved, data, name)
        validate_dataset_provenance(response_saved, data, name)

        prefix_diffs, _ = get_pair_diffs(prefix_saved["activations"], data, label_map)
        response_diffs, _ = get_pair_diffs(response_saved["activations"], data, label_map)
        n_pairs, n_layers, _ = prefix_diffs.shape

        labels = np.array([label_map[s["condition"]] for s in data])
        prefix_aurocs = [sample_auroc(prefix_saved["activations"][:, l], labels, prefix_diffs[:, l]) for l in range(n_layers)]
        response_aurocs = [sample_auroc(response_saved["activations"][:, l], labels, response_diffs[:, l]) for l in range(n_layers)]

        best_prefix_layer = int(np.argmax(prefix_aurocs))
        best_response_layer = int(np.argmax(response_aurocs))

        prefix_best = prefix_aurocs[best_prefix_layer]
        response_best = response_aurocs[best_response_layer]

        print(f"{name}: prefix AUROC={prefix_best:.4f} (L{best_prefix_layer+1}), response AUROC={response_best:.4f} (L{best_response_layer+1}), delta={response_best - prefix_best:+.4f}")
        rows.append({
            "dataset": name,
            "prefix_auroc": prefix_best,
            "prefix_best_layer": best_prefix_layer + 1,
            "response_auroc": response_best,
            "response_best_layer": best_response_layer + 1,
            "delta": response_best - prefix_best,
            "n_pairs": n_pairs,
            "prefix_layer_aurocs": prefix_aurocs,
            "response_layer_aurocs": response_aurocs,
        })

    if not rows:
        print("No datasets processed.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(output_dir) / f"confound_prefix_{model_tag}_{timestamp}.csv"
    with open(csv_path, "w") as f:
        f.write("dataset,prefix_auroc,prefix_best_layer,response_auroc,response_best_layer,delta,n_pairs\n")
        for r in rows:
            f.write(f"{r['dataset']},{r['prefix_auroc']:.4f},{r['prefix_best_layer']},{r['response_auroc']:.4f},{r['response_best_layer']},{r['delta']:+.4f},{r['n_pairs']}\n")
    print(f"\nCSV: {csv_path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    datasets = [r["dataset"] for r in rows]
    prefix_vals = [r["prefix_auroc"] for r in rows]
    response_vals = [r["response_auroc"] for r in rows]
    x = np.arange(len(datasets))

    axes[0].bar(x - 0.15, prefix_vals, 0.3, label="prefix (last_user)", color="#D68F73")
    axes[0].bar(x + 0.15, response_vals, 0.3, label="response (first)", color="#729ECE")
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("Best-layer AUROC")
    axes[0].set_title("Prefix vs Response probe (best layer)")
    axes[0].legend(loc="upper left", fontsize=8)

    for r in rows:
        n_layers = len(r["prefix_layer_aurocs"])
        layers = np.arange(1, n_layers + 1)
        axes[1].plot(layers, r["prefix_layer_aurocs"], linestyle="--", alpha=0.6, label=f"{r['dataset']} prefix")
        axes[1].plot(layers, r["response_layer_aurocs"], linestyle="-", alpha=0.8, label=f"{r['dataset']} response")
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("AUROC")
    axes[1].set_title("Layer sweep: prefix (dashed) vs response (solid)")
    axes[1].legend(loc="upper left", fontsize=6, ncol=2)

    fig.tight_layout()
    plot_path = Path(output_dir) / f"confound_prefix_{model_tag}_{timestamp}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    fire.Fire(run)

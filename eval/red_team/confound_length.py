"""
Red team diagnostic: response length confound.

Measures Spearman correlation between response length (character and token)
and probe score across all deception conditions. Reports within-class
correlations (lie-only, truth-only) as the primary confound metric.
High within-class correlation indicates the probe may be detecting verbosity
differences rather than deception.

Prerequisites: trained probe .pt file, extracted activations, paired JSON datasets

Usage:
    python confound_length.py --model_tag gemma3-27b --probe_path ../../probes/mass_mean/shared_direction_gemma3-27b.pt
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr
from transformers import AutoTokenizer
import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import TRAIN_DATASETS, dataset_filename, MODEL_REGISTRY


def run(model_tag,
        probe_path,
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
    elif "shared_direction_all" in probe:
        best_layer = probe.get("best_layer_transfer", 1)
        best_idx = best_layer - 1
        direction = probe["all_directions"][best_idx]
    else:
        direction = probe["direction"]
        best_layer = probe.get("best_layer", 1)
        best_idx = best_layer - 1

    direction = direction / np.linalg.norm(direction)
    probe_name = Path(probe_path).stem

    tokenizer = AutoTokenizer.from_pretrained(MODEL_REGISTRY[model_tag], trust_remote_code=True)

    rows = []
    scatter_data = {}

    for name, (filename, label_map) in TRAIN_DATASETS.items():
        act_path = Path(act_dir) / f"{name}.pt"
        fname = dataset_filename(filename, model_tag) if model_tag else filename
        data_path = Path(data_dir) / fname
        if not data_path.exists():
            data_path = Path(data_dir) / filename
        if not act_path.exists() or not data_path.exists():
            print(f"Skipping {name}: missing files")
            continue

        saved = torch.load(act_path, weights_only=False)
        data = json.load(open(data_path))[:len(saved["activations"])]

        char_lengths = np.array([len(s["model_response"]) for s in data])
        token_lengths = np.array([len(tokenizer.encode(s["model_response"])) for s in data])
        scores = saved["activations"][:, best_idx] @ direction
        labels = np.array([label_map[s["condition"]] for s in data])

        lie = labels == 1
        truth = labels == 0

        rho_char, p_char = spearmanr(char_lengths, scores)
        rho_char_lie, p_char_lie = spearmanr(char_lengths[lie], scores[lie])
        rho_char_truth, p_char_truth = spearmanr(char_lengths[truth], scores[truth])
        rho_tok, p_tok = spearmanr(token_lengths, scores)
        rho_tok_lie, p_tok_lie = spearmanr(token_lengths[lie], scores[lie])
        rho_tok_truth, p_tok_truth = spearmanr(token_lengths[truth], scores[truth])

        print(f"{name}: char within-class lie={rho_char_lie:.4f} truth={rho_char_truth:.4f}, "
              f"tok within-class lie={rho_tok_lie:.4f} truth={rho_tok_truth:.4f}")
        rows.append({
            "dataset": name,
            "rho_char": rho_char, "p_char": p_char,
            "rho_char_lie": rho_char_lie, "p_char_lie": p_char_lie,
            "rho_char_truth": rho_char_truth, "p_char_truth": p_char_truth,
            "rho_tok": rho_tok, "p_tok": p_tok,
            "rho_tok_lie": rho_tok_lie, "p_tok_lie": p_tok_lie,
            "rho_tok_truth": rho_tok_truth, "p_tok_truth": p_tok_truth,
            "n_samples": len(data),
            "mean_char_lie": float(char_lengths[lie].mean()),
            "mean_char_truth": float(char_lengths[truth].mean()),
            "mean_tok_lie": float(token_lengths[lie].mean()),
            "mean_tok_truth": float(token_lengths[truth].mean()),
        })
        scatter_data[name] = {
            "char_lengths": char_lengths, "token_lengths": token_lengths,
            "scores": scores, "labels": labels,
        }

    if not rows:
        print("No datasets processed.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(output_dir) / f"confound_length_{model_tag}_{timestamp}.csv"
    with open(csv_path, "w") as f:
        f.write("dataset,rho_char,p_char,rho_char_lie,p_char_lie,rho_char_truth,p_char_truth,"
                "rho_tok,p_tok,rho_tok_lie,p_tok_lie,rho_tok_truth,p_tok_truth,"
                "n_samples,mean_char_lie,mean_char_truth,mean_tok_lie,mean_tok_truth\n")
        for r in rows:
            f.write(f"{r['dataset']},{r['rho_char']:.4f},{r['p_char']:.2e},"
                    f"{r['rho_char_lie']:.4f},{r['p_char_lie']:.2e},"
                    f"{r['rho_char_truth']:.4f},{r['p_char_truth']:.2e},"
                    f"{r['rho_tok']:.4f},{r['p_tok']:.2e},"
                    f"{r['rho_tok_lie']:.4f},{r['p_tok_lie']:.2e},"
                    f"{r['rho_tok_truth']:.4f},{r['p_tok_truth']:.2e},"
                    f"{r['n_samples']},{r['mean_char_lie']:.1f},{r['mean_char_truth']:.1f},"
                    f"{r['mean_tok_lie']:.1f},{r['mean_tok_truth']:.1f}\n")
    print(f"\nCSV: {csv_path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")

    n_ds = len(scatter_data)
    cols = min(3, n_ds)
    ds_rows = (n_ds + cols - 1) // cols
    fig, axes = plt.subplots(ds_rows * 2, cols, figsize=(5 * cols, 4 * ds_rows * 2), squeeze=False)

    for idx, (name, sd) in enumerate(scatter_data.items()):
        r = rows[idx]
        lie = sd["labels"] == 1
        truth = sd["labels"] == 0
        row_base = (idx // cols) * 2
        col = idx % cols

        for row_off, (lengths, lbl, rk_lie, rk_truth) in enumerate([
            (sd["char_lengths"], "char", "rho_char_lie", "rho_char_truth"),
            (sd["token_lengths"], "token", "rho_tok_lie", "rho_tok_truth"),
        ]):
            ax = axes[row_base + row_off][col]
            ax.scatter(lengths[truth], sd["scores"][truth], alpha=0.3, s=10, color="#729ECE", label="truth")
            ax.scatter(lengths[lie], sd["scores"][lie], alpha=0.3, s=10, color="#D68F73", label="lie")
            for mask, color in [(truth, "#729ECE"), (lie, "#D68F73")]:
                if mask.sum() > 1:
                    z = np.polyfit(lengths[mask], sd["scores"][mask], 1)
                    xline = np.linspace(lengths[mask].min(), lengths[mask].max(), 100)
                    ax.plot(xline, np.polyval(z, xline), color=color, linewidth=1.5, linestyle="--")
            ax.set_title(f"{name} ({lbl})\nlie={r[rk_lie]:.3f}  truth={r[rk_truth]:.3f}", fontsize=9)
            ax.set_xlabel(f"response {lbl} length")
            ax.set_ylabel("probe score")
            ax.legend(fontsize=7)

    for idx in range(n_ds, ds_rows * cols):
        row_base = (idx // cols) * 2
        col = idx % cols
        axes[row_base][col].set_visible(False)
        axes[row_base + 1][col].set_visible(False)

    fig.suptitle(f"Length vs probe score (within-class rho) — {probe_name} (L{best_layer})", fontsize=12)
    fig.tight_layout()
    plot_path = Path(output_dir) / f"confound_length_{model_tag}_{timestamp}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    fire.Fire(run)

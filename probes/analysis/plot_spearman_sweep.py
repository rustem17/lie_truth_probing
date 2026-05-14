"""
Spearman correlation sweep across all layers.

Probes: contrastive LR directions from shared_direction.pt
Activations: first-position hidden states from activations_{model_tag}/{name}.pt
Output: spearman_sweep_{suffix}.png — line plot, x=layer, y=Spearman rho per probe pair
"""
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import combinations
from scipy.stats import spearmanr
import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    DEFAULT_MODEL_TAG,
    SHORT,
    pair_color,
    PROBING_ROOT,
    activation_dirname,
    resolve_model,
)
from artifact_utils import load_shared_direction  # noqa: E402

sns.set_theme(style="whitegrid")


def main(activations_dir=None, probes_dir=None, output_dir=None,
         datasets="instructed_system_prompt,spontaneous_1,sycophancy_answer", layer=None,
         model=DEFAULT_MODEL_TAG, allow_untagged_fallback=False):
    model_tag, _ = resolve_model(model) if model else ("", "")
    activations_dir = Path(activations_dir) if activations_dir else PROBING_ROOT / activation_dirname(model_tag)
    probes_dir = Path(probes_dir) if probes_dir else PROBING_ROOT / "probes" / "contrastive"
    output_dir = Path(output_dir) if output_dir else probes_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    shared, _ = load_shared_direction(probes_dir, model_tag, allow_untagged_fallback)
    directions = {k: v for k, v in shared["per_layer_directions"].items()}
    n_layers = next(iter(directions.values())).shape[0]

    best_layer = shared.get("best_layer_transfer", None)
    if layer is not None:
        best_layer = int(layer)

    names = set(datasets) if isinstance(datasets, (list, tuple)) else {d.strip() for d in datasets.split(",")}
    probe_names = sorted(n for n in names if n in directions)

    ds = {}
    for name in sorted(names):
        act_path = activations_dir / f"{name}.pt"
        if not act_path.exists():
            continue
        saved = torch.load(act_path, weights_only=False)
        ds[name] = {"activations": saved["activations"], "labels": saved["labels"]}

    print(f"probes: {probe_names}, datasets: {sorted(ds.keys())}, layers: {n_layers}")

    pairs = list(combinations(probe_names, 2))
    per_layer_rho = {f"{a}_vs_{b}": [] for a, b in pairs}

    for l in range(n_layers):
        all_acts = np.concatenate([ds[n]["activations"][:, l] for n in sorted(ds)])
        for a, b in pairs:
            s_a = all_acts @ directions[a][l]
            s_b = all_acts @ directions[b][l]
            rho, _ = spearmanr(s_a, s_b)
            per_layer_rho[f"{a}_vs_{b}"].append(float(rho))

    layers = np.arange(1, n_layers + 1)
    mean_rho = [np.nanmean([per_layer_rho[k][l] for k in per_layer_rho]) for l in range(n_layers)]

    fig, ax = plt.subplots(figsize=(12, 5))
    for a, b in pairs:
        k = f"{a}_vs_{b}"
        ax.plot(layers, per_layer_rho[k], label=f"{SHORT[a]} vs {SHORT[b]}",
                color=pair_color(a, b), linewidth=1.5, alpha=0.85)
    ax.plot(layers, mean_rho, label="mean", color="black", linewidth=2.2)
    if best_layer:
        ax.axvline(best_layer, color="gray", linestyle=":", linewidth=1, label=f"best_layer={best_layer}")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Spearman Score Correlation by Layer")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    suffix = "_".join(SHORT.get(n, n) for n in sorted(ds))
    out = output_dir / f"spearman_sweep_{suffix}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    fire.Fire(main)

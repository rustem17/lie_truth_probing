"""Shared plotting helpers for probe result directories."""

import json
import sys
from datetime import datetime
from pathlib import Path

import fire
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import COLORS, DEFAULT_MODEL_TAG, SHORT, pair_color, resolve_model  # noqa: E402
from artifact_utils import load_shared_direction, resolve_artifact_path  # noqa: E402

sns.set_theme(style="whitegrid")


def load_json(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def plot_cross_transfer(data, out_dir, ts, method_label):
    transfer = data["transfer"]
    layers = np.arange(1, len(transfer["per_layer_mean"]) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for pk, vals in sorted(transfer["per_layer"].items()):
        src, tgt = pk.split("\u2192")
        base = tuple(sorted([src, tgt]))
        color = pair_color(base[0], base[1])
        ls = "-" if (src, tgt) == base else "--"
        ax.plot(layers, vals, label=pk, color=color, linestyle=ls, linewidth=1.2, alpha=0.85)

    ax.plot(layers, transfer["per_layer_mean"], label="mean", color="black", linewidth=2.2)
    best = data["best_layer_transfer"]
    ax.axvline(best, color="gray", linestyle=":", linewidth=1)
    ax.annotate(f"L{best}", (best, transfer["per_layer_mean"][best - 1] - 0.02), fontsize=8, color="gray")
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC")
    ax.set_title(f"{method_label} Cross-condition Transfer AUROC by Layer")
    ax.legend(loc="upper left", fontsize=7.5, ncol=2)
    ax.set_ylim(0.45, 1.02)
    fig.tight_layout()
    fig.savefig(out_dir / f"cross_transfer_{ts}.png", dpi=150)
    plt.close(fig)


def plot_transfer_matrix(data, out_dir, ts, method_label):
    transfer = data["transfer"]
    best_idx = data["best_layer_transfer"] - 1

    names = set()
    for k in transfer["per_layer"]:
        s, t = k.split("\u2192")
        names.update([s, t])
    names = sorted(names)
    n = len(names)

    matrix = np.full((n, n), np.nan)
    for pk, vals in transfer["per_layer"].items():
        src, tgt = pk.split("\u2192")
        matrix[names.index(src), names.index(tgt)] = vals[best_idx]

    short = [SHORT.get(nm, nm[:4]) for nm in names]
    cmap = mcolors.LinearSegmentedColormap.from_list("rg", ["#d9534f", "#f5f5dc", "#2d8659"])
    fig, ax = plt.subplots(figsize=(max(6, n * 1.8), max(5, n * 1.5)))
    matrix_flip = matrix[::-1]
    im = ax.imshow(matrix_flip, cmap=cmap, vmin=0.5, vmax=1.0, aspect="equal")
    for i in range(n):
        for j in range(n):
            val = matrix_flip[i, j]
            if np.isnan(val):
                ax.text(j, i, "+", ha="center", va="center", fontsize=16, color="gray")
            else:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=14, fontweight="bold", color="white")
    ax.set_xticks(range(n))
    ax.set_xticklabels(short)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short[::-1])
    ax.set_xlabel("Target")
    ax.set_ylabel("Source")
    ax.set_title(f"{method_label} Transfer Matrix at Layer {data['best_layer_transfer']}")
    fig.colorbar(im, ax=ax, label="AUROC", shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / f"transfer_matrix_{ts}.png", dpi=150)
    plt.close(fig)


def plot_training_auroc(results, out_dir, ts, method_label):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in sorted(results.keys()):
        color = COLORS.get(name, "gray")
        vals = [r["auroc"] for r in results[name]]
        layers = [r["layer"] for r in results[name]]
        ax.plot(layers, vals, color=color, linewidth=1.2, label=name, alpha=0.85)

    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC (CV)")
    ax.set_title(f"{method_label} Per-dataset Training AUROC by Layer")
    ax.legend(loc="upper left", fontsize=7.5, ncol=2)
    ax.set_ylim(0.45, 1.02)
    fig.tight_layout()
    fig.savefig(out_dir / f"training_auroc_{ts}.png", dpi=150)
    plt.close(fig)


def plot_validation(val_data, out_dir, ts, method_label):
    entries = {}
    for key, info in val_data.items():
        probe, val_set = key.split("\u2192")
        entries.setdefault(probe, []).append((val_set, info["auroc"], info["n_pairs"]))

    if not entries:
        return

    probes = sorted(entries.keys())
    fig, ax = plt.subplots(figsize=(max(8, len(probes) * 3), 5))

    group_x = np.arange(len(probes))
    max_bars = max(len(v) for v in entries.values())
    bar_width = 0.8 / max(max_bars, 1)

    for pi, probe in enumerate(probes):
        bars = entries[probe]
        for bi, (val_set, auroc, n_pairs) in enumerate(bars):
            x = group_x[pi] + (bi - len(bars) / 2 + 0.5) * bar_width
            color = COLORS.get(probe, "#729ECE")
            ax.bar(x, auroc, bar_width * 0.9, color=color, alpha=0.85)
            ax.text(x, auroc + 0.01, f"{auroc:.2f}", ha="center", va="bottom", fontsize=8)
            ax.text(x, auroc / 2, f"{val_set}\nn={n_pairs}",
                    ha="center", va="center", fontsize=6.5, color="white", fontweight="bold")

    ax.set_xticks(group_x)
    ax.set_xticklabels([SHORT.get(p, p) for p in probes])
    ax.set_ylabel("AUROC")
    ax.set_title(f"{method_label} Validation Results by Probe")
    ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / f"validation_results_{ts}.png", dpi=150)
    plt.close(fig)


def result_path(out_dir, base_name, model_tag, allow_untagged_fallback=False):
    return resolve_artifact_path(
        out_dir,
        base_name,
        model_tag,
        allow_untagged_fallback=allow_untagged_fallback,
    )


def main(probes_dir=".", model=DEFAULT_MODEL_TAG, method_label="Probe", allow_untagged_fallback=False):
    out_dir = Path(probes_dir)
    model_tag, _ = resolve_model(model) if model else ("", "")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        data, _ = load_shared_direction(
            out_dir,
            model_tag,
            allow_untagged_fallback=allow_untagged_fallback,
        )
        plot_cross_transfer(data, out_dir, ts, method_label)
        plot_transfer_matrix(data, out_dir, ts, method_label)
        print("plots 1-2: cross_transfer, transfer_matrix")
    except FileNotFoundError:
        print("skipping shared direction plots: shared_direction.pt not found")

    results = load_json(result_path(out_dir, "results.json", model_tag, allow_untagged_fallback))
    if results:
        plot_training_auroc(results, out_dir, ts, method_label)
        print("plot 3: training_auroc")
    else:
        print("skipping training auroc: results.json not found")

    val = load_json(result_path(out_dir, "validation_results.json", model_tag, allow_untagged_fallback))
    if val:
        plot_validation(val, out_dir, ts, method_label)
        print("plot 4: validation_results")
    else:
        print("skipping validation: validation_results.json not found")

    print(f"\nsaved to {out_dir}/")


if __name__ == "__main__":
    fire.Fire(main)

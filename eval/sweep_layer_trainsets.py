"""
Per-layer AUROC sweep for all 18 configs x 4 train-set variations.

Model: default config model (offline analysis of cached activations)
Data: activations_{model_tag}/*.pt + paired JSONs
Configs: 18 from sweep_shared_configs.CONFIGS
Train variations: ISS, ISSf, GSS, GSSf (3 datasets each)
Layer range: 10-40 (1-indexed)
Metric: per-dataset AUROC of shared direction on pair diffs at each layer
Output: per-variation folders with 18 plots + CSV, plus summary heatmap
"""

import sys
import csv
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import fire
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR.parent / "probes" / "contrastive"))

from config import (
    DEFAULT_MODEL_TAG,
    TRAIN_DATASETS,
    VALIDATION_DATASETS,
    COLORS,
    SHORT,
    activation_dirname,
    control_abs_delta,
    eval_role,
    is_primary_eval,
    resolve_model,
)
from shared_direction import (
    load_all, train_all_directions, train_pooled_direction,
    aggregate_directions, transfer_auroc, cross_transfer_all_layers,
    compute_transfer_aggregates, constrained_argmax,
)
from sweep_shared_configs import load_validation_diffs, CONFIGS

LO, HI = 9, 40

TRAIN_VARIATIONS = {
    "ISS":  ["instructed_system_prompt", "spontaneous_1", "sycophancy_answer"],
    "ISSf": ["instructed_system_prompt", "spontaneous_1", "sycophancy_feedback"],
    "GSS":  ["game_werewolf", "spontaneous_1", "sycophancy_answer"],
    "GSSf": ["game_werewolf", "spontaneous_1", "sycophancy_feedback"],
}


def mean_or_zero(values):
    values = list(values)
    return round(float(np.mean(values)), 4) if values else 0.0


def min_or_zero(values):
    values = list(values)
    return round(float(min(values)), 4) if values else 0.0


def run_layer_sweep(cfg, train_diffs_subset, all_train_diffs, val_diffs, directions_cache, train_key):
    C = cfg["C"]
    cache_key = (C, train_key)
    if cache_key not in directions_cache:
        names = sorted(train_diffs_subset.keys())
        directions_cache[cache_key] = train_all_directions(train_diffs_subset, names, C=C)
    directions = directions_cache[cache_key]

    active_names = sorted(directions.keys())
    n_layers = next(iter(directions.values())).shape[0]

    all_eval = {**all_train_diffs, **val_diffs}
    eval_names = sorted(all_eval.keys())

    layer_rows = []
    for layer in range(LO, HI):
        if cfg["shared_mode"] == "pooled":
            shared = train_pooled_direction(train_diffs_subset, active_names, layer, C)
        else:
            shared = aggregate_directions(
                [directions[n][layer] for n in active_names], cfg["agg_mode"])

        row = {"layer": layer + 1}
        for name in eval_names:
            d = all_eval[name]
            if layer < d["D"].shape[1]:
                row[name] = round(transfer_auroc(shared, d["D"][:, layer]), 4)
        primary_aurocs = [row[n] for n in eval_names if n in row and is_primary_eval(n)]
        control_d = [control_abs_delta(row[n], n) for n in eval_names if n in row and eval_role(n) == "control"]
        syco_aurocs = [row[n] for n in eval_names if n in row and eval_role(n) == "sycophancy_variant"]
        row["mean_auroc"] = mean_or_zero(primary_aurocs)
        row["min_auroc"] = min_or_zero(primary_aurocs)
        row["control_mean_abs_delta"] = mean_or_zero(control_d)
        row["sycophancy_variant_mean_auroc"] = mean_or_zero(syco_aurocs)
        layer_rows.append(row)

    return layer_rows, eval_names


def plot_layer_sweep(layer_rows, eval_names, cfg_label, out_dir, role="primary_transfer"):
    eval_names = [n for n in eval_names if eval_role(n) == role]
    if not eval_names:
        return
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    layers = [r["layer"] for r in layer_rows]

    for name in eval_names:
        aurocs = np.array([r.get(name, np.nan) for r in layer_rows], dtype=float)
        y = np.abs(aurocs - 0.5) if role == "control" else aurocs
        color = COLORS.get(name, "#333333")
        short = SHORT.get(name, name[:6])
        is_val = name in VALIDATION_DATASETS
        ax.plot(layers, y, color=color, linestyle="--" if is_val else "-",
                marker="o", markersize=2, linewidth=1.2, label=short, alpha=0.85)

    ax.axhline(0.0 if role == "control" else 0.5, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("|AUROC - 0.5|" if role == "control" else "AUROC")
    ax.set_title(f"{cfg_label} / {role}")
    ax.set_ylim(0, 0.55) if role == "control" else ax.set_ylim(0.3, 1.05)
    ax.legend(loc="lower left", fontsize=7, framealpha=0.7, ncol=2)
    fig.tight_layout()
    safe_name = cfg_label.replace("/", "_").replace("+", "_").replace("=", "")
    fig.savefig(Path(out_dir) / f"{safe_name}_{role}.png", dpi=150)
    plt.close(fig)


def plot_summary_heatmap(summary_rows, dataset_cols, out_path, role="primary_transfer"):
    dataset_cols = [d for d in dataset_cols if eval_role(d) == role]
    if not dataset_cols:
        return
    sns.set_theme(style="whitegrid")
    labels = [r["label"] for r in summary_rows]
    if role == "control":
        mat = np.array([[control_abs_delta(float(r[d]), d) if d in r else np.nan
                         for d in dataset_cols] for r in summary_rows])
    else:
        mat = np.array([[float(r.get(d, np.nan)) for d in dataset_cols] for r in summary_rows])
    layers = [int(r["best_layer"]) for r in summary_rows]

    annot = np.empty_like(mat, dtype=object)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            annot[i, j] = f"{v:.2f}\nL{layers[i]}" if not np.isnan(v) else ""

    col_labels = [SHORT.get(d, d[:6]) for d in dataset_cols]
    fig, ax = plt.subplots(figsize=(max(10, len(dataset_cols) * 1.3),
                                     max(8, len(summary_rows) * 0.35)))
    cmap, vmin, vmax, cbar_label = ("Reds", 0.0, 0.5, "|AUROC - 0.5|") if role == "control" else (
        "RdYlGn", 0.4, 1.0, "AUROC")
    sns.heatmap(mat, annot=annot, fmt="", cmap=cmap, vmin=vmin, vmax=vmax,
                linewidths=0.5, ax=ax, xticklabels=col_labels, yticklabels=labels,
                cbar_kws={"label": cbar_label}, annot_kws={"fontsize": 6})
    ax.set_title(f"Layer sweep summary: {role} at best primary-mean layer")
    ax.tick_params(axis="y", labelsize=7)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(data_dir=None, activations_dir=None, output_dir=None, model=DEFAULT_MODEL_TAG):
    probing_root = Path(__file__).resolve().parents[1]
    model_tag, _ = resolve_model(model) if model else ("", "")
    if data_dir is None:
        data_dir = str(probing_root)
    if activations_dir is None:
        activations_dir = str(probing_root / activation_dirname(model_tag))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = str(Path(data_dir) / "eval" / "results")
    sweep_dir = Path(output_dir) / f"layer_sweep_{ts}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    all_train_diffs, model_tag_loaded, _ = load_all(Path(data_dir), Path(activations_dir), model_tag)
    val_diffs = load_validation_diffs(Path(data_dir), Path(activations_dir), model_tag_loaded or model_tag)
    print(f"train datasets: {sorted(all_train_diffs.keys())}")
    print(f"validation datasets: {sorted(val_diffs.keys())}")
    print(f"configs: {len(CONFIGS)}, variations: {len(TRAIN_VARIATIONS)}")
    print(f"output: {sweep_dir}\n")

    directions_cache = {}
    summary_rows = []
    all_dataset_cols = set()

    for var_name, var_datasets in TRAIN_VARIATIONS.items():
        var_dir = sweep_dir / var_name
        var_dir.mkdir(parents=True, exist_ok=True)
        train_subset = {k: v for k, v in all_train_diffs.items() if k in var_datasets}
        available = sorted(train_subset.keys())
        missing = [d for d in var_datasets if d not in train_subset]
        if missing:
            print(f"  {var_name}: missing {missing}, skipping")
            continue
        print(f"\n{'='*60}")
        print(f"variation: {var_name} — train on {available}")
        print(f"{'='*60}")

        train_key = frozenset(available)
        all_csv_rows = []

        for ci, cfg in enumerate(CONFIGS):
            label = cfg["label"]
            print(f"  [{ci+1}/{len(CONFIGS)}] {label}")

            layer_rows, eval_names = run_layer_sweep(
                cfg, train_subset, all_train_diffs, val_diffs, directions_cache, train_key)

            for r in layer_rows:
                csv_row = {"config": label, "variation": var_name}
                csv_row.update({k: cfg[k] for k in ["C", "agg_mode", "shared_mode", "layer_objective", "ensemble", "ensemble_k"]})
                csv_row.update(r)
                all_csv_rows.append(csv_row)

            for role in ["primary_transfer", "control", "sycophancy_variant"]:
                plot_layer_sweep(layer_rows, eval_names, f"{var_name} / {label}", var_dir, role)

            best_idx = int(np.argmax([r["mean_auroc"] for r in layer_rows]))
            best_row = layer_rows[best_idx]
            sr = {"label": f"{var_name}/{label}", "best_layer": best_row["layer"],
                  "mean_auroc": best_row["mean_auroc"], "min_auroc": best_row["min_auroc"],
                  "control_mean_abs_delta": best_row["control_mean_abs_delta"],
                  "sycophancy_variant_mean_auroc": best_row["sycophancy_variant_mean_auroc"]}
            for n in eval_names:
                sr[n] = best_row.get(n, np.nan)
                all_dataset_cols.add(n)
            summary_rows.append(sr)
            print(f"    best L{best_row['layer']}, mean={best_row['mean_auroc']}, min={best_row['min_auroc']}")

        csv_path = var_dir / f"{var_name}_results.csv"
        if all_csv_rows:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_csv_rows[0].keys())
                writer.writeheader()
                writer.writerows(all_csv_rows)
            print(f"  CSV: {csv_path}")

    dataset_cols = sorted(all_dataset_cols)
    for role in ["primary_transfer", "control", "sycophancy_variant"]:
        heatmap_path = sweep_dir / f"summary_heatmap_{role}.png"
        plot_summary_heatmap(summary_rows, dataset_cols, heatmap_path, role)
    print(f"\nSummary heatmaps: {sweep_dir}/summary_heatmap_<role>.png")
    print(f"Output: {sweep_dir}")


if __name__ == "__main__":
    fire.Fire(main)

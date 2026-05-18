"""
Run activation-position sweeps through the standard probing pipeline.

This is an orchestration script. It reuses the normal extraction, probe training,
shared-direction, and validation entry points instead of reimplementing probe
logic locally.

Typical usage:
  uv run python eval/sweep_positions.py \
      --model llama-3-3-70b-instruct \
      --positions first,last,mean_assistant \
      --methods mass_mean,mass_mean_iid,paired_pca,contrastive,mahalanobis_lda

Phases:
  all        extract activations, train probes, validate probes, summarize
  extract    extract activations only
  train      train probes only
  shared     build shared directions only
  validate   validate trained probes only
  summarize  collect existing validation JSONs into CSV/plots
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import fire
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from config import (  # noqa: E402
    ALL_DATASETS,
    DEFAULT_MODEL_TAG,
    SHORT,
    TRAIN_DATASETS,
    VALIDATION_DATASETS,
    activation_dirname,
    resolve_model,
    tagged_filename,
)

POSITIONS = [
    "first",
    "last",
    "first_assistant",
    "last_user",
    "mean_assistant",
    "mid_assistant",
    "first_k_assistant",
    "last_k_assistant",
]

METHODS = {
    "mass_mean": {
        "dir": ROOT_DIR / "probes" / "mass_mean",
        "train": "train.py",
        "validate": "validate.py",
        "shared": "shared_direction.py",
    },
    "mass_mean_iid": {
        "dir": ROOT_DIR / "probes" / "mass_mean_iid",
        "train": "train.py",
        "validate": "validate.py",
        "shared": "shared_direction.py",
    },
    "paired_pca": {
        "dir": ROOT_DIR / "probes" / "paired_pca",
        "train": "train.py",
        "validate": "validate.py",
        "shared": "shared_direction.py",
    },
    "contrastive": {
        "dir": ROOT_DIR / "probes" / "contrastive",
        "train": "train.py",
        "validate": "validate.py",
        "shared": "shared_direction.py",
    },
    "mahalanobis_lda": {
        "dir": ROOT_DIR / "probes" / "mahalanobis_lda",
        "train": "train.py",
        "validate": "validate.py",
        "shared": "shared_direction.py",
    },
    "irm": {
        "dir": ROOT_DIR / "probes" / "irm",
        "train": "train.py",
        "validate": "validate.py",
        "shared": None,
    },
}

DEFAULT_METHODS = ["mass_mean", "mass_mean_iid", "paired_pca", "contrastive", "mahalanobis_lda"]
ARROW = "\u2192"


def parse_list(value, default=None):
    if value is None or value == "":
        return list(default or [])
    if isinstance(value, (list, tuple)):
        return list(value)
    return [part.strip() for part in str(value).split(",") if part.strip()]


def format_cli_list(value):
    return ",".join(str(part) for part in parse_list(value))


def format_layer_range(value):
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"layer_range must have two values, got {value!r}")
        return f"{int(value[0])},{int(value[1])}"
    return str(value)


def resolve_positions(positions):
    if positions in ("all", "*"):
        return list(POSITIONS)
    pos_list = parse_list(positions, POSITIONS)
    unknown = sorted(set(pos_list) - set(POSITIONS))
    if unknown:
        raise ValueError(f"Unknown positions: {unknown}. Valid: {POSITIONS}")
    return pos_list


def resolve_methods(methods):
    if methods in ("all", "*"):
        return list(METHODS)
    method_list = parse_list(methods, DEFAULT_METHODS)
    unknown = sorted(set(method_list) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Valid: {sorted(METHODS)}")
    return method_list


def resolve_datasets(datasets):
    if datasets in (None, "", "all", "*"):
        return list(ALL_DATASETS)
    if datasets == "train":
        return list(TRAIN_DATASETS)
    if datasets == "validation":
        return list(VALIDATION_DATASETS)
    names = parse_list(datasets)
    unknown = sorted(set(names) - set(ALL_DATASETS))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}. Valid: {sorted(ALL_DATASETS)}")
    return names


def resolve_steps(phase, run_shared):
    if phase == "all":
        steps = ["extract", "train", "validate", "summarize"]
        if run_shared:
            steps.insert(2, "shared")
        return steps
    aliases = {"summary": "summarize"}
    steps = [aliases.get(step, step) for step in parse_list(phase)]
    valid = {"extract", "train", "shared", "validate", "summarize"}
    unknown = sorted(set(steps) - valid)
    if unknown:
        raise ValueError(f"Unknown phase(s): {unknown}. Valid: {sorted(valid)} or all")
    return steps


def activation_path(data_dir, model_tag, position):
    return Path(data_dir) / activation_dirname(model_tag, position)


def position_output_dir(sweep_dir, position, method):
    return Path(sweep_dir) / position / method


def train_result_path(out_dir, method, model_tag):
    if method == "irm":
        return Path(out_dir) / tagged_filename("irm_probe.pt", model_tag)
    return Path(out_dir) / tagged_filename("results.json", model_tag)


def shared_result_path(out_dir, model_tag):
    return Path(out_dir) / tagged_filename("shared_direction.pt", model_tag)


def run_command(args, cwd, dry_run=False):
    printable = " ".join(str(a) for a in args)
    print(f"    $ {printable}")
    if dry_run:
        return
    subprocess.run([str(a) for a in args], cwd=str(cwd), check=True)


def write_manifest(sweep_dir, manifest):
    Path(sweep_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(sweep_dir) / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def phase_extract(
    data_dir,
    positions,
    model_tag,
    model_id,
    datasets,
    force_extract=False,
    max_samples=None,
    max_length=2048,
    k=10,
    dry_run=False,
):
    from extract_activations import extract_with_model, load_model

    dataset_names = resolve_datasets(datasets)
    needed = []
    for pos in positions:
        act_dir = activation_path(data_dir, model_tag, pos)
        missing = [name for name in dataset_names if not (act_dir / f"{name}.pt").exists()]
        if force_extract or missing:
            needed.append((pos, missing))
        else:
            print(f"  {pos}: activations complete at {act_dir}")

    if not needed:
        return

    if dry_run:
        for pos, missing in needed:
            act_dir = activation_path(data_dir, model_tag, pos)
            reason = "force_extract=True" if force_extract else f"{len(missing)} missing dataset activation file(s)"
            print(f"  would extract position={pos} -> {act_dir} ({reason})")
        return

    model, tokenizer = load_model(model_id)
    dataset_arg = ",".join(dataset_names)
    for pos, missing in needed:
        act_dir = activation_path(data_dir, model_tag, pos)
        if force_extract:
            reason = "force_extract=True"
        else:
            preview = ", ".join(missing[:5])
            suffix = "" if len(missing) <= 5 else f", +{len(missing) - 5} more"
            reason = f"missing {preview}{suffix}"
        print(f"\n  extracting position={pos} -> {act_dir} ({reason})")
        extract_with_model(
            model,
            tokenizer,
            data_dir=str(data_dir),
            output_dir=str(act_dir),
            position=pos,
            max_length=max_length,
            datasets=dataset_arg,
            max_samples=max_samples,
            k=k,
            model_tag=model_tag,
            model_id=model_id,
        )


def phase_train(
    data_dir,
    sweep_dir,
    positions,
    methods,
    model,
    model_tag,
    n_splits=5,
    ridge=1e-4,
    mass_mean_iid_score_mode="iid",
    paired_pca_center=False,
    irm_envs="instructed_system_prompt,spontaneous_1,sycophancy_answer",
    irm_epochs=500,
    irm_penalty="irm",
    device="auto",
    skip_existing=False,
    dry_run=False,
):
    for pos in positions:
        act_dir = activation_path(data_dir, model_tag, pos)
        if not act_dir.exists():
            print(f"  {pos}: no activations at {act_dir}, skipping train")
            continue
        print(f"\n  training position={pos}")
        for method in methods:
            spec = METHODS[method]
            out_dir = position_output_dir(sweep_dir, pos, method)
            out_dir.mkdir(parents=True, exist_ok=True)
            result_path = train_result_path(out_dir, method, model_tag)
            if skip_existing and result_path.exists():
                print(f"  {method}: existing {result_path.name}, skipping train")
                continue
            args = [
                sys.executable,
                spec["train"],
                "--data_dir",
                str(data_dir),
                "--activations_dir",
                str(act_dir),
                "--output_dir",
                str(out_dir),
                "--model",
                model,
            ]
            if method in {"mass_mean", "mass_mean_iid", "paired_pca", "contrastive", "mahalanobis_lda"}:
                args += ["--n_splits", str(n_splits)]
                args += ["--device", device]
            if method in {"mass_mean_iid", "mahalanobis_lda"}:
                args += ["--ridge", str(ridge)]
            if method == "mass_mean_iid":
                args += ["--score_mode", mass_mean_iid_score_mode]
            if method == "paired_pca":
                args += [f"--center={paired_pca_center}"]
            if method == "irm":
                args += ["--envs", irm_envs, "--n_epochs", str(irm_epochs), "--penalty", irm_penalty]
            print(f"  {method}:")
            run_command(args, spec["dir"], dry_run=dry_run)


def phase_shared(
    data_dir,
    sweep_dir,
    positions,
    methods,
    model,
    model_tag,
    shared_datasets=None,
    layer_range="20,40",
    layer_objective="mean",
    shared_mode="average",
    C=1.0,
    agg_mode="mean",
    ensemble="none",
    ensemble_k=5,
    ridge=1e-4,
    mahalanobis_shared_mode="multi_env",
    device="auto",
    mass_mean_iid_score_mode="iid",
    paired_pca_center=False,
    skip_existing=False,
    dry_run=False,
):
    for pos in positions:
        act_dir = activation_path(data_dir, model_tag, pos)
        if not act_dir.exists():
            print(f"  {pos}: no activations at {act_dir}, skipping shared directions")
            continue
        print(f"\n  shared directions position={pos}")
        for method in methods:
            spec = METHODS[method]
            if not spec["shared"]:
                print(f"  {method}: no shared-direction script")
                continue
            out_dir = position_output_dir(sweep_dir, pos, method)
            out_dir.mkdir(parents=True, exist_ok=True)
            result_path = shared_result_path(out_dir, model_tag)
            if skip_existing and result_path.exists():
                print(f"  {method}: existing {result_path.name}, skipping shared directions")
                continue
            args = [
                sys.executable,
                spec["shared"],
                "--data_dir",
                str(data_dir),
                "--activations_dir",
                str(act_dir),
                "--output_dir",
                str(out_dir),
                "--model",
                model,
            ]
            if shared_datasets:
                args += ["--datasets", format_cli_list(shared_datasets)]
            if method == "contrastive":
                args += [
                    "--layer_range",
                    format_layer_range(layer_range),
                    "--layer_objective",
                    layer_objective,
                    "--shared_mode",
                    shared_mode,
                    "--C",
                    str(C),
                    "--agg_mode",
                    agg_mode,
                    "--ensemble",
                    ensemble,
                    "--ensemble_k",
                    str(ensemble_k),
                ]
            if method in {"mass_mean_iid", "mahalanobis_lda"}:
                args += ["--ridge", str(ridge)]
            if method == "mahalanobis_lda":
                args += ["--shared_mode", mahalanobis_shared_mode, "--device", device]
            if method == "mass_mean_iid":
                args += ["--score_mode", mass_mean_iid_score_mode]
            if method == "paired_pca":
                args += [f"--center={paired_pca_center}"]
            print(f"  {method}:")
            run_command(args, spec["dir"], dry_run=dry_run)


def phase_validate(data_dir, sweep_dir, positions, methods, model, model_tag, skip_existing=False, dry_run=False):
    for pos in positions:
        act_dir = activation_path(data_dir, model_tag, pos)
        if not act_dir.exists():
            print(f"  {pos}: no activations at {act_dir}, skipping validate")
            continue
        print(f"\n  validating position={pos}")
        for method in methods:
            spec = METHODS[method]
            out_dir = position_output_dir(sweep_dir, pos, method)
            if not out_dir.exists():
                print(f"  {method}: no trained probes at {out_dir}, skipping")
                continue
            result_path = validation_result_path(out_dir, model_tag)
            if skip_existing and result_path.exists():
                print(f"  {method}: existing {result_path.name}, skipping validate")
                continue
            args = [
                sys.executable,
                spec["validate"],
                "--data_dir",
                str(data_dir),
                "--activations_dir",
                str(act_dir),
                "--model",
                model,
            ]
            if method == "irm":
                args += ["--output_dir", str(out_dir)]
            else:
                args += ["--probes_dir", str(out_dir)]
            print(f"  {method}:")
            run_command(args, spec["dir"], dry_run=dry_run)


def validation_result_path(out_dir, model_tag):
    tagged = Path(out_dir) / tagged_filename("validation_results.json", model_tag)
    if tagged.exists():
        return tagged
    fallback = Path(out_dir) / "validation_results.json"
    return fallback if fallback.exists() else tagged


def collect_results(sweep_dir, positions, methods, model_tag):
    rows = []
    for pos in positions:
        for method in methods:
            out_dir = position_output_dir(sweep_dir, pos, method)
            result_path = validation_result_path(out_dir, model_tag)
            if not result_path.exists():
                continue
            with open(result_path) as f:
                results = json.load(f)
            for key, info in results.items():
                if "auroc" not in info:
                    continue
                if ARROW in key:
                    source, target = key.split(ARROW, 1)
                    result_key = f"{source}->{target}"
                else:
                    source, target = method, key
                    result_key = key
                try:
                    result_file = str(result_path.relative_to(ROOT_DIR))
                except ValueError:
                    result_file = str(result_path)
                rows.append(
                    {
                        "position": pos,
                        "method": method,
                        "source": source,
                        "target": target,
                        "result_key": result_key,
                        "auroc": float(info["auroc"]),
                        "n_pairs": int(info.get("n_pairs", 0)),
                        "layer": int(info.get("layer", 0)),
                        "held_out": info.get("held_out", ""),
                        "result_file": result_file,
                    }
                )
    return rows


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["position"], row["method"])].append(row)
    summary = []
    for (position, method), group in sorted(groups):
        vals = [r["auroc"] for r in group]
        summary.append(
            {
                "position": position,
                "method": method,
                "mean_auroc": float(np.mean(vals)),
                "min_auroc": float(np.min(vals)),
                "max_auroc": float(np.max(vals)),
                "n_evals": len(vals),
            }
        )
    return summary


def plot_summary(summary_rows, sweep_dir, positions, methods):
    if not summary_rows:
        return
    lookup = {(r["method"], r["position"]): r["mean_auroc"] for r in summary_rows}
    mat = np.full((len(methods), len(positions)), np.nan)
    for i, method in enumerate(methods):
        for j, pos in enumerate(positions):
            mat[i, j] = lookup.get((method, pos), np.nan)

    annot = np.empty_like(mat, dtype=object)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            annot[i, j] = "" if np.isnan(mat[i, j]) else f"{mat[i, j]:.3f}"

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(max(8, len(positions) * 1.2), max(3, len(methods) * 0.6)))
    sns.heatmap(
        mat,
        annot=annot,
        fmt="",
        cmap="RdYlGn",
        vmin=0.4,
        vmax=1.0,
        linewidths=0.5,
        ax=ax,
        xticklabels=positions,
        yticklabels=methods,
        cbar_kws={"label": "Mean validation AUROC"},
    )
    ax.set_title("Position Sweep Summary")
    ax.set_xlabel("Activation position")
    ax.set_ylabel("Probe method")
    plt.tight_layout()
    fig.savefig(Path(sweep_dir) / "position_sweep_summary.png", dpi=150)
    plt.close(fig)


def plot_method_details(rows, sweep_dir, positions, methods):
    for method in methods:
        method_rows = [r for r in rows if r["method"] == method]
        if not method_rows:
            continue
        result_keys = sorted({r["result_key"] for r in method_rows})
        lookup = {(r["position"], r["result_key"]): r["auroc"] for r in method_rows}
        mat = np.full((len(positions), len(result_keys)), np.nan)
        for i, pos in enumerate(positions):
            for j, key in enumerate(result_keys):
                mat[i, j] = lookup.get((pos, key), np.nan)

        annot = np.empty_like(mat, dtype=object)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                annot[i, j] = "" if np.isnan(mat[i, j]) else f"{mat[i, j]:.2f}"

        labels = [SHORT.get(k, k) for k in result_keys]
        fig, ax = plt.subplots(figsize=(max(9, len(result_keys) * 1.2), max(4, len(positions) * 0.45)))
        sns.heatmap(
            mat,
            annot=annot,
            fmt="",
            cmap="RdYlGn",
            vmin=0.4,
            vmax=1.0,
            linewidths=0.5,
            ax=ax,
            xticklabels=labels,
            yticklabels=positions,
            cbar_kws={"label": "AUROC"},
        )
        ax.set_title(f"{method} Validation AUROC by Position")
        ax.set_xlabel("")
        ax.set_ylabel("Activation position")
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
        plt.tight_layout()
        fig.savefig(Path(sweep_dir) / f"position_sweep_{method}.png", dpi=150)
        plt.close(fig)


def phase_summarize(sweep_dir, positions, methods, model_tag):
    rows = collect_results(sweep_dir, positions, methods, model_tag)
    if not rows:
        print(f"  no validation results found in {sweep_dir}")
        return

    detailed_path = Path(sweep_dir) / "position_sweep_results.csv"
    write_csv(
        detailed_path,
        rows,
        [
            "position",
            "method",
            "source",
            "target",
            "result_key",
            "auroc",
            "n_pairs",
            "layer",
            "held_out",
            "result_file",
        ],
    )

    summary_rows = aggregate_rows(rows)
    summary_path = Path(sweep_dir) / "position_sweep_summary.csv"
    write_csv(summary_path, summary_rows, ["position", "method", "mean_auroc", "min_auroc", "max_auroc", "n_evals"])

    plot_summary(summary_rows, sweep_dir, positions, methods)
    plot_method_details(rows, sweep_dir, positions, methods)

    print(f"\n  detailed CSV: {detailed_path}")
    print(f"  summary CSV:  {summary_path}")
    print(f"  plots:        {sweep_dir}")
    print("\n  mean AUROC by position/method:")
    for row in summary_rows:
        print(
            f"    {row['position']:<18} {row['method']:<16} "
            f"mean={row['mean_auroc']:.4f} min={row['min_auroc']:.4f} n={row['n_evals']}"
        )


def main(
    model=DEFAULT_MODEL_TAG,
    positions="first,last,first_assistant,last_user,mean_assistant,mid_assistant,first_k_assistant,last_k_assistant",
    methods="mass_mean,mass_mean_iid,paired_pca,contrastive,mahalanobis_lda",
    phase="all",
    data_dir=None,
    output_dir=None,
    run_name=None,
    datasets=None,
    shared_datasets=None,
    run_shared=False,
    force_extract=False,
    max_samples=None,
    max_length=2048,
    k=10,
    n_splits=5,
    ridge=1e-4,
    mahalanobis_shared_mode="multi_env",
    device="auto",
    mass_mean_iid_score_mode="iid",
    paired_pca_center=False,
    irm_envs="instructed_system_prompt,spontaneous_1,sycophancy_answer",
    irm_epochs=500,
    irm_penalty="irm",
    layer_range="20,40",
    layer_objective="mean",
    shared_mode="average",
    C=1.0,
    agg_mode="mean",
    ensemble="none",
    ensemble_k=5,
    skip_existing=False,
    dry_run=False,
):
    model_tag, model_id = resolve_model(model)
    pos_list = resolve_positions(positions)
    method_list = resolve_methods(methods)
    steps = resolve_steps(phase, run_shared)

    data_dir = Path(data_dir or ROOT_DIR).resolve()
    output_dir = Path(output_dir or ROOT_DIR / "eval" / "results").resolve()
    if run_name is None:
        run_name = f"position_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sweep_dir = output_dir / run_name
    sweep_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "model": model,
        "model_tag": model_tag,
        "model_id": model_id,
        "positions": pos_list,
        "methods": method_list,
        "steps": steps,
        "data_dir": str(data_dir),
        "sweep_dir": str(sweep_dir),
        "datasets": resolve_datasets(datasets),
        "shared_datasets": shared_datasets,
        "max_samples": max_samples,
        "max_length": max_length,
        "k": k,
        "ridge": ridge,
        "mahalanobis_shared_mode": mahalanobis_shared_mode,
        "device": device,
        "mass_mean_iid_score_mode": mass_mean_iid_score_mode,
        "paired_pca_center": paired_pca_center,
        "skip_existing": skip_existing,
    }
    write_manifest(sweep_dir, manifest)

    print(f"model:      {model} ({model_id})")
    print(f"model_tag:  {model_tag}")
    print(f"positions:  {pos_list}")
    print(f"methods:    {method_list}")
    print(f"steps:      {steps}")
    print(f"output:     {sweep_dir}\n")

    if "extract" in steps:
        phase_extract(
            data_dir,
            pos_list,
            model_tag,
            model_id,
            datasets,
            force_extract=force_extract,
            max_samples=max_samples,
            max_length=max_length,
            k=k,
            dry_run=dry_run,
        )

    if "train" in steps:
        phase_train(
            data_dir,
            sweep_dir,
            pos_list,
            method_list,
            model,
            model_tag,
            n_splits=n_splits,
            ridge=ridge,
            mass_mean_iid_score_mode=mass_mean_iid_score_mode,
            paired_pca_center=paired_pca_center,
            irm_envs=irm_envs,
            irm_epochs=irm_epochs,
            irm_penalty=irm_penalty,
            device=device,
            skip_existing=skip_existing,
            dry_run=dry_run,
        )

    if "shared" in steps:
        phase_shared(
            data_dir,
            sweep_dir,
            pos_list,
            method_list,
            model,
            model_tag,
            shared_datasets=shared_datasets,
            layer_range=layer_range,
            layer_objective=layer_objective,
            shared_mode=shared_mode,
            C=C,
            agg_mode=agg_mode,
            ensemble=ensemble,
            ensemble_k=ensemble_k,
            ridge=ridge,
            mahalanobis_shared_mode=mahalanobis_shared_mode,
            device=device,
            mass_mean_iid_score_mode=mass_mean_iid_score_mode,
            paired_pca_center=paired_pca_center,
            skip_existing=skip_existing,
            dry_run=dry_run,
        )

    if "validate" in steps:
        phase_validate(data_dir, sweep_dir, pos_list, method_list, model, model_tag,
                       skip_existing=skip_existing, dry_run=dry_run)

    if "summarize" in steps:
        phase_summarize(sweep_dir, pos_list, method_list, model_tag)

    print(f"\nDone. Output: {sweep_dir}")


if __name__ == "__main__":
    fire.Fire(main)

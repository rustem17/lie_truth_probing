"""Fixed evaluator for autonomous probe-search experiments."""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from config import DEFAULT_MODEL_TAG, resolve_model

from .features import PairDiffs, default_eval_datasets, load_many_pair_diffs
from .probes import (
    confound_directions,
    dataset_identity_penalty,
    fit_shared_direction,
    length_spearman,
    residualize,
    score_auroc,
    unit_objective,
)
from .specs import ExperimentSpec


@dataclass
class EvaluationContext:
    data_dir: Path
    model_tag: str = DEFAULT_MODEL_TAG
    mode: str = "smoke"
    baseline_runtime_seconds: float = 60.0


def robust_score(
    val_mean_auroc: float,
    val_min_auroc: float,
    length_penalty: float,
    dataset_penalty: float,
    runtime_seconds: float,
    baseline_runtime_seconds: float,
) -> float:
    runtime_ratio = max(runtime_seconds / max(baseline_runtime_seconds, 1e-6), 1.0)
    return float(
        val_mean_auroc
        - 0.50 * max(0.0, 0.75 - val_min_auroc)
        - 0.20 * length_penalty
        - 0.10 * dataset_penalty
        - 0.02 * np.log10(runtime_ratio)
    )


def _allowed_units(unit_labels: list[dict[str, int]], layer_range: tuple[int, int]) -> list[int]:
    lo, hi = layer_range
    out = []
    for idx, label in enumerate(unit_labels):
        layer = label.get("layer")
        if layer is None or lo <= layer <= hi:
            out.append(idx)
    return out


def _train_unit(unit_idx: int, train: dict[str, PairDiffs], spec: ExperimentSpec) -> tuple[np.ndarray, dict[str, float], float]:
    unit_diffs = {name: pairs.D[:, unit_idx] for name, pairs in train.items()}
    direction = fit_shared_direction(unit_diffs, spec)
    direction = residualize(direction, confound_directions(train, unit_idx, spec.residualize_confounds))
    train_aurocs = {name: score_auroc([(unit_idx, direction, 1.0)], pairs) for name, pairs in train.items()}
    objective = unit_objective(list(train_aurocs.values()), spec)
    return direction, train_aurocs, objective


def _select_units(unit_results: list[dict[str, Any]], spec: ExperimentSpec) -> list[tuple[int, np.ndarray, float]]:
    if not unit_results:
        raise ValueError("no unit results available")
    ranked = sorted(unit_results, key=lambda r: r["objective"], reverse=True)
    k = spec.ensemble_k if spec.ensemble != "none" else 1
    selected = ranked[:k]
    if spec.ensemble == "transfer_weighted":
        raw = np.asarray([max(r["objective"], 0.0) for r in selected], dtype=np.float64)
        weights = raw / raw.sum() if raw.sum() > 1e-12 else np.ones(len(selected)) / len(selected)
    else:
        weights = np.ones(len(selected), dtype=np.float64) / len(selected)
    return [
        (int(row["unit_idx"]), row["direction"], float(weight))
        for row, weight in zip(selected, weights)
    ]


def evaluate_experiment(spec: ExperimentSpec, context: EvaluationContext) -> dict[str, Any]:
    start = time.perf_counter()
    spec.validate()
    model_tag, _ = resolve_model(context.model_tag)

    eval_names = spec.eval_datasets or default_eval_datasets(spec.train_datasets)
    train, missing_train = load_many_pair_diffs(
        spec.train_datasets,
        spec.feature_type,
        model_tag,
        context.data_dir,
        spec.position,
        spec.pair_cap,
    )
    eval_sets, missing_eval = load_many_pair_diffs(
        eval_names,
        spec.feature_type,
        model_tag,
        context.data_dir,
        spec.position,
        spec.pair_cap,
    )

    if missing_train:
        return {
            "status": "skipped",
            "reason": "missing_train_features",
            "missing_train": missing_train,
            "missing_eval": missing_eval,
            "spec": spec.to_dict(),
        }
    if len(train) < 2:
        return {
            "status": "skipped",
            "reason": "need_at_least_two_train_datasets",
            "loaded_train": sorted(train),
            "spec": spec.to_dict(),
        }
    if not eval_sets:
        return {
            "status": "skipped",
            "reason": "missing_eval_features",
            "missing_eval": missing_eval,
            "spec": spec.to_dict(),
        }

    unit_labels = next(iter(train.values())).unit_labels
    dims = {pairs.D.shape[2] for pairs in train.values()}
    if len(dims) != 1:
        return {
            "status": "skipped",
            "reason": "incompatible_feature_dimensions",
            "dims": sorted(dims),
            "spec": spec.to_dict(),
        }

    allowed = _allowed_units(unit_labels, spec.layer_range)
    if not allowed:
        return {
            "status": "skipped",
            "reason": "no_units_in_layer_range",
            "layer_range": list(spec.layer_range),
            "spec": spec.to_dict(),
        }

    unit_results = []
    for unit_idx in allowed:
        direction, train_aurocs, objective = _train_unit(unit_idx, train, spec)
        unit_results.append({
            "unit_idx": unit_idx,
            "unit_label": unit_labels[unit_idx],
            "direction": direction,
            "train_aurocs": train_aurocs,
            "objective": objective,
        })

    selected = _select_units(unit_results, spec)
    selected_public = [
        {
            "unit_idx": unit_idx,
            "unit_label": unit_labels[unit_idx],
            "weight": weight,
        }
        for unit_idx, _, weight in selected
    ]

    train_scores = {name: score_auroc(selected, pairs) for name, pairs in train.items()}
    eval_scores = {name: score_auroc(selected, pairs) for name, pairs in eval_sets.items()}
    val_scores = {name: score for name, score in eval_scores.items() if name not in train}
    target_scores = list(val_scores.values()) if val_scores else list(eval_scores.values())

    val_mean = float(np.mean(target_scores)) if target_scores else 0.0
    val_min = float(np.min(target_scores)) if target_scores else 0.0
    length_penalty = length_spearman(selected, eval_sets)
    dataset_penalty = dataset_identity_penalty(selected, train)
    runtime = time.perf_counter() - start
    score = robust_score(
        val_mean,
        val_min,
        length_penalty,
        dataset_penalty,
        runtime,
        context.baseline_runtime_seconds,
    )

    best_unit_rows = sorted(unit_results, key=lambda r: r["objective"], reverse=True)[:10]
    return {
        "status": "ok",
        "spec": spec.to_dict(),
        "model_tag": model_tag,
        "mode": context.mode,
        "feature_type": spec.feature_type,
        "position": spec.position,
        "train_datasets": sorted(train),
        "eval_datasets": sorted(eval_sets),
        "missing_eval": missing_eval,
        "selected_units": selected_public,
        "top_units": [
            {
                "unit_idx": row["unit_idx"],
                "unit_label": row["unit_label"],
                "objective": float(row["objective"]),
                "train_aurocs": {k: float(v) for k, v in row["train_aurocs"].items()},
            }
            for row in best_unit_rows
        ],
        "train_aurocs": {k: float(v) for k, v in train_scores.items()},
        "eval_aurocs": {k: float(v) for k, v in eval_scores.items()},
        "val_mean_auroc": val_mean,
        "val_min_auroc": val_min,
        "length_penalty": length_penalty,
        "dataset_identity_penalty": dataset_penalty,
        "runtime_seconds": runtime,
        "robust_score": score,
        "complexity": {
            "ensemble_k": len(selected),
            "residualized_confounds": list(spec.residualize_confounds),
            "method": spec.method,
        },
    }


def evaluate_safely(spec: ExperimentSpec, context: EvaluationContext) -> dict[str, Any]:
    try:
        return evaluate_experiment(spec, context)
    except Exception as exc:  # noqa: BLE001 - report candidate crashes without hiding traceback
        return {
            "status": "crash",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "spec": spec.to_dict(),
        }


def write_report(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=_json_default)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

"""Feature-store adapters for autoprobe.

The residual adapter reads the repo's existing ``activations_{model_tag}``
files unchanged. Other feature types use the same metadata contract and skip
cleanly until extractors populate them.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from config import (
    ALL_DATASETS,
    TRAIN_DATASETS,
    VALIDATION_DATASETS,
    activation_dirname,
    resolve_dataset_path_for_activation,
    validate_dataset_provenance,
)


@dataclass
class PairDiffs:
    dataset: str
    D: np.ndarray
    pair_ids: list[str]
    unit_labels: list[dict[str, int]]
    pair_meta: list[dict[str, Any]]
    feature_type: str
    position: str
    source_path: Path
    model_tag: str
    model_id: str


def feature_dirname(feature_type: str, model_tag: str, position: str = "first") -> str:
    if feature_type == "residual":
        return activation_dirname(model_tag, position)
    suffix = "" if position == "first" else f"_{position}"
    return f"features_{feature_type}_{model_tag}{suffix}" if model_tag else f"features_{feature_type}{suffix}"


def dataset_registry(include_validation: bool = True) -> dict[str, tuple[str, dict[str, int]]]:
    if include_validation:
        return dict(ALL_DATASETS)
    return dict(TRAIN_DATASETS)


def default_eval_datasets(train_datasets: tuple[str, ...]) -> tuple[str, ...]:
    heldout_train = [name for name in TRAIN_DATASETS if name not in set(train_datasets)]
    return tuple([*VALIDATION_DATASETS.keys(), *heldout_train])


def _load_feature_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a dict feature payload")
    return payload


def _feature_array(payload: dict[str, Any]) -> tuple[np.ndarray, tuple[str, ...]]:
    if "features" in payload:
        arr = np.asarray(payload["features"])
        axes = tuple(payload.get("axes") or ())
    elif "activations" in payload:
        arr = np.asarray(payload["activations"])
        axes = ("sample", "layer", "dim")
    else:
        raise ValueError("feature payload must contain 'features' or 'activations'")

    if arr.ndim == 3:
        axes = axes or ("sample", "layer", "dim")
    elif arr.ndim == 4:
        axes = axes or ("sample", "layer", "head", "dim")
    else:
        raise ValueError(f"unsupported feature rank {arr.ndim}; expected rank 3 or 4")
    return arr, axes


def _flatten_units(features: np.ndarray, axes: tuple[str, ...]) -> tuple[np.ndarray, list[dict[str, int]]]:
    if features.ndim == 3:
        n_samples, n_units, dim = features.shape
        labels = [{"layer": i + 1} for i in range(n_units)]
        return features.reshape(n_samples, n_units, dim), labels

    n_samples, n_layers, n_heads, dim = features.shape
    flat = features.reshape(n_samples, n_layers * n_heads, dim)
    labels = [
        {"layer": layer + 1, "head": head}
        for layer in range(n_layers)
        for head in range(n_heads)
    ]
    return flat, labels


def _base_id(sample_id: str) -> str:
    if sample_id.endswith(("_lie", "_truth")):
        return sample_id.rsplit("_", 1)[0]
    return sample_id


def _pair_metadata(pair_id: str, lie_sample: dict[str, Any], truth_sample: dict[str, Any]) -> dict[str, Any]:
    lie_response = str(lie_sample.get("model_response", ""))
    truth_response = str(truth_sample.get("model_response", ""))
    lie_prompt = str(lie_sample.get("system_prompt", "")) + "\n" + str(lie_sample.get("user_message", ""))
    truth_prompt = str(truth_sample.get("system_prompt", "")) + "\n" + str(truth_sample.get("user_message", ""))
    return {
        "pair_id": pair_id,
        "lie_id": lie_sample.get("id"),
        "truth_id": truth_sample.get("id"),
        "response_length_delta": len(lie_response) - len(truth_response),
        "abs_response_length_delta": abs(len(lie_response) - len(truth_response)),
        "prompt_length_delta": len(lie_prompt) - len(truth_prompt),
        "abs_prompt_length_delta": abs(len(lie_prompt) - len(truth_prompt)),
    }


def get_pair_diffs(features: np.ndarray, data: list[dict[str, Any]], label_map: dict[str, int]) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    by_id: dict[str, dict[str, int]] = defaultdict(dict)
    for i, sample in enumerate(data):
        is_lie = label_map[sample["condition"]] == 1
        by_id[_base_id(str(sample["id"]))]["lie" if is_lie else "truth"] = i

    pair_ids: list[str] = []
    pair_meta: list[dict[str, Any]] = []
    diffs = []
    for pair_id in sorted(by_id):
        pair = by_id[pair_id]
        if "lie" not in pair or "truth" not in pair:
            continue
        lie_idx = pair["lie"]
        truth_idx = pair["truth"]
        diffs.append(features[lie_idx] - features[truth_idx])
        pair_ids.append(pair_id)
        pair_meta.append(_pair_metadata(pair_id, data[lie_idx], data[truth_idx]))

    if not diffs:
        raise ValueError("no complete lie/truth pairs found")
    return np.stack(diffs), pair_ids, pair_meta


def cap_pairs(pair_diffs: PairDiffs, pair_cap: int | None) -> PairDiffs:
    if pair_cap is None or pair_cap <= 0 or pair_diffs.D.shape[0] <= pair_cap:
        return pair_diffs
    return PairDiffs(
        dataset=pair_diffs.dataset,
        D=pair_diffs.D[:pair_cap],
        pair_ids=pair_diffs.pair_ids[:pair_cap],
        unit_labels=pair_diffs.unit_labels,
        pair_meta=pair_diffs.pair_meta[:pair_cap],
        feature_type=pair_diffs.feature_type,
        position=pair_diffs.position,
        source_path=pair_diffs.source_path,
        model_tag=pair_diffs.model_tag,
        model_id=pair_diffs.model_id,
    )


def load_pair_diffs(
    dataset_name: str,
    feature_type: str,
    model_tag: str,
    data_dir: Path,
    position: str = "first",
    pair_cap: int | None = None,
) -> PairDiffs:
    registry = dataset_registry(include_validation=True)
    if dataset_name not in registry:
        raise KeyError(f"unknown dataset {dataset_name!r}")
    filename, label_map = registry[dataset_name]
    feature_dir = Path(data_dir) / feature_dirname(feature_type, model_tag, position)
    feature_path = feature_dir / f"{dataset_name}.pt"
    if not feature_path.exists():
        raise FileNotFoundError(f"{dataset_name}: missing feature store {feature_path}")

    payload = _load_feature_payload(feature_path)
    feature_type_payload = payload.get("feature_type", "residual" if "activations" in payload else feature_type)
    if feature_type_payload != feature_type:
        raise ValueError(f"{feature_path}: feature_type={feature_type_payload!r}, expected {feature_type!r}")

    arr, axes = _feature_array(payload)
    flat, unit_labels = _flatten_units(arr, axes)
    data_path = resolve_dataset_path_for_activation(data_dir, filename, payload.get("model_tag", model_tag), payload)
    if not data_path.exists():
        raise FileNotFoundError(f"{dataset_name}: missing dataset {data_path}")
    with open(data_path) as f:
        data = json.load(f)[: flat.shape[0]]
    validate_dataset_provenance(payload, data, dataset_name)

    D, pair_ids, pair_meta = get_pair_diffs(flat[: len(data)], data, label_map)
    loaded = PairDiffs(
        dataset=dataset_name,
        D=D.astype(np.float32, copy=False),
        pair_ids=pair_ids,
        unit_labels=unit_labels,
        pair_meta=pair_meta,
        feature_type=feature_type,
        position=position,
        source_path=feature_path,
        model_tag=payload.get("model_tag", model_tag),
        model_id=payload.get("model_id", ""),
    )
    return cap_pairs(loaded, pair_cap)


def load_many_pair_diffs(
    dataset_names: tuple[str, ...],
    feature_type: str,
    model_tag: str,
    data_dir: Path,
    position: str,
    pair_cap: int | None,
) -> tuple[dict[str, PairDiffs], dict[str, str]]:
    loaded: dict[str, PairDiffs] = {}
    missing: dict[str, str] = {}
    for name in dataset_names:
        try:
            loaded[name] = load_pair_diffs(name, feature_type, model_tag, data_dir, position, pair_cap)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            missing[name] = str(exc)
    return loaded, missing

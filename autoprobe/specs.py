"""Serializable experiment specifications for autoprobe."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

DEFAULT_ENVS = (
    "instructed_system_prompt",
    "spontaneous_1",
    "sycophancy_answer",
)

FEATURE_TYPES = {
    "residual",
    "attention_output",
    "mlp_output",
    "attention_delta",
    "mlp_delta",
    "block_delta",
}

METHODS = {
    "mass_mean",
    "contrastive",
    "l1_logistic",
    "pooled_logistic",
    "irm",
    "vrex",
    "group_dro",
}

SHARED_MODES = {"average", "pooled"}
AGG_MODES = {"mean", "geometric_median"}
LAYER_OBJECTIVES = {"mean", "min", "median", "harmonic", "variance_penalized"}
ENSEMBLES = {"none", "top_k", "transfer_weighted"}
CONFOUNDS = {"response_length", "prompt_length", "dataset"}


@dataclass(frozen=True)
class ExperimentSpec:
    """One candidate probe-search experiment.

    The autonomous loop may emit many of these from ``autoprobe/candidates.py``.
    The fixed evaluator owns metric calculation and keep/discard decisions.
    """

    name: str
    description: str = ""
    feature_type: str = "residual"
    position: str = "first"
    method: str = "contrastive"
    train_datasets: tuple[str, ...] = DEFAULT_ENVS
    eval_datasets: tuple[str, ...] = ()
    layer_range: tuple[int, int] = (10, 50)
    pair_cap: int | None = 200
    shared_mode: str = "average"
    agg_mode: str = "mean"
    layer_objective: str = "mean"
    C: float = 1.0
    l1_C: float = 0.1
    max_iter: int = 1000
    ensemble: str = "none"
    ensemble_k: int = 1
    residualize_confounds: tuple[str, ...] = ()
    group_dro_rounds: int = 4
    group_dro_eta: float = 1.0
    variance_penalty: float = 0.25
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("ExperimentSpec.name is required")
        if self.feature_type not in FEATURE_TYPES:
            raise ValueError(f"{self.name}: unsupported feature_type={self.feature_type!r}")
        if self.method not in METHODS:
            raise ValueError(f"{self.name}: unsupported method={self.method!r}")
        if self.shared_mode not in SHARED_MODES:
            raise ValueError(f"{self.name}: unsupported shared_mode={self.shared_mode!r}")
        if self.agg_mode not in AGG_MODES:
            raise ValueError(f"{self.name}: unsupported agg_mode={self.agg_mode!r}")
        if self.layer_objective not in LAYER_OBJECTIVES:
            raise ValueError(f"{self.name}: unsupported layer_objective={self.layer_objective!r}")
        if self.ensemble not in ENSEMBLES:
            raise ValueError(f"{self.name}: unsupported ensemble={self.ensemble!r}")
        if self.ensemble_k < 1:
            raise ValueError(f"{self.name}: ensemble_k must be >= 1")
        if self.layer_range[0] < 1 or self.layer_range[1] < self.layer_range[0]:
            raise ValueError(f"{self.name}: layer_range must be 1-indexed inclusive")
        unknown_confounds = set(self.residualize_confounds) - CONFOUNDS
        if unknown_confounds:
            raise ValueError(f"{self.name}: unsupported confounds={sorted(unknown_confounds)}")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, tuple):
                data[key] = list(value)
        return data

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "ExperimentSpec":
        tuple_fields = {
            "train_datasets",
            "eval_datasets",
            "layer_range",
            "residualize_confounds",
        }
        normalized = dict(data)
        for key in tuple_fields:
            if key in normalized and normalized[key] is not None:
                normalized[key] = tuple(normalized[key])
        spec = cls(**normalized)
        spec.validate()
        return spec


def safe_name(name: str) -> str:
    out = []
    for ch in name.lower():
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        elif ch.isspace() or ch in {"/", ":", "."}:
            out.append("-")
    clean = "".join(out).strip("-_")
    return clean or "experiment"

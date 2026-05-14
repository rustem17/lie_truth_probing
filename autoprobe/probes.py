"""Probe fitting and scoring primitives used by the fixed autoprobe evaluator."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .features import PairDiffs
from .specs import ExperimentSpec


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-12 else np.zeros_like(v)


def augment(D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = D.shape[0]
    return np.concatenate([D, -D], axis=0), np.array([1] * n + [0] * n)


def pair_auroc(direction: np.ndarray, D: np.ndarray) -> float:
    scores = D @ direction
    labels = np.concatenate([np.ones(D.shape[0]), np.zeros(D.shape[0])])
    return float(roc_auc_score(labels, np.concatenate([scores, -scores])))


def score_pairs(selected: list[tuple[int, np.ndarray, float]], pair_diffs: PairDiffs) -> np.ndarray:
    score = np.zeros(pair_diffs.D.shape[0], dtype=np.float64)
    weight_sum = 0.0
    for unit_idx, direction, weight in selected:
        if unit_idx >= pair_diffs.D.shape[1]:
            continue
        score += weight * (pair_diffs.D[:, unit_idx] @ direction)
        weight_sum += weight
    if weight_sum > 0:
        score /= weight_sum
    return score


def score_auroc(selected: list[tuple[int, np.ndarray, float]], pair_diffs: PairDiffs) -> float:
    scores = score_pairs(selected, pair_diffs)
    labels = np.concatenate([np.ones(len(scores)), np.zeros(len(scores))])
    return float(roc_auc_score(labels, np.concatenate([scores, -scores])))


def geometric_median(vectors: Iterable[np.ndarray], max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    vecs = np.asarray(list(vectors))
    y = vecs.mean(axis=0)
    for _ in range(max_iter):
        dists = np.maximum(np.linalg.norm(vecs - y, axis=1), 1e-12)
        weights = 1.0 / dists
        new_y = np.average(vecs, axis=0, weights=weights)
        if np.linalg.norm(new_y - y) < tol:
            break
        y = new_y
    return normalize(y)


def aggregate(directions: list[np.ndarray], agg_mode: str) -> np.ndarray:
    if not directions:
        raise ValueError("cannot aggregate empty direction list")
    if agg_mode == "geometric_median":
        return geometric_median(directions)
    return normalize(np.mean(directions, axis=0))


def _linear_direction(D: np.ndarray, method: str, C: float, l1_C: float, max_iter: int, sample_weight=None) -> np.ndarray:
    if method == "mass_mean":
        return normalize(D.mean(axis=0))

    X, y = augment(D)
    penalty = "l1" if method == "l1_logistic" else "l2"
    solver = "saga" if penalty == "l1" else "lbfgs"
    clf = LogisticRegression(
        C=l1_C if penalty == "l1" else C,
        fit_intercept=False,
        max_iter=max_iter,
        penalty=penalty,
        solver=solver,
        random_state=0,
    )
    clf.fit(X, y, sample_weight=sample_weight)
    return normalize(clf.coef_[0])


def _pooled_direction(unit_diffs: dict[str, np.ndarray], spec: ExperimentSpec, sample_weights=None) -> np.ndarray:
    Ds = []
    weights = []
    for name, D in unit_diffs.items():
        Ds.append(D)
        if sample_weights:
            w = np.full(D.shape[0], sample_weights[name] / max(D.shape[0], 1))
            weights.append(np.concatenate([w, w]))
    all_D = np.concatenate(Ds, axis=0)
    sample_weight = np.concatenate(weights) if weights else None
    return _linear_direction(
        all_D,
        "contrastive" if spec.method in {"pooled_logistic", "vrex", "irm", "group_dro"} else spec.method,
        spec.C,
        spec.l1_C,
        spec.max_iter,
        sample_weight=sample_weight,
    )


def _group_dro_direction(unit_diffs: dict[str, np.ndarray], spec: ExperimentSpec) -> np.ndarray:
    env_names = list(unit_diffs)
    weights = {name: 1.0 / len(env_names) for name in env_names}
    direction = None
    for _ in range(max(spec.group_dro_rounds, 1)):
        direction = _pooled_direction(unit_diffs, spec, sample_weights=weights)
        risks = {name: max(0.0, 1.0 - pair_auroc(direction, D)) for name, D in unit_diffs.items()}
        total = 0.0
        for name, risk in risks.items():
            weights[name] *= math.exp(spec.group_dro_eta * risk)
            total += weights[name]
        weights = {name: value / total for name, value in weights.items()}
    return direction if direction is not None else _pooled_direction(unit_diffs, spec)


def fit_shared_direction(unit_diffs: dict[str, np.ndarray], spec: ExperimentSpec) -> np.ndarray:
    if spec.method == "group_dro":
        return _group_dro_direction(unit_diffs, spec)
    if spec.shared_mode == "pooled" or spec.method in {"pooled_logistic", "vrex", "irm"}:
        return _pooled_direction(unit_diffs, spec)

    per_env = [
        _linear_direction(D, spec.method, spec.C, spec.l1_C, spec.max_iter)
        for D in unit_diffs.values()
    ]
    return aggregate(per_env, spec.agg_mode)


def confound_directions(unit_diffs: dict[str, PairDiffs], unit_idx: int, confounds: tuple[str, ...]) -> list[np.ndarray]:
    dirs: list[np.ndarray] = []
    if not confounds:
        return dirs

    if "response_length" in confounds or "prompt_length" in confounds:
        D_all = []
        for pairs in unit_diffs.values():
            D_all.append(pairs.D[:, unit_idx])
        X = np.concatenate(D_all, axis=0)
        for confound_name, key in [
            ("response_length", "response_length_delta"),
            ("prompt_length", "prompt_length_delta"),
        ]:
            if confound_name not in confounds:
                continue
            targets = []
            for pairs in unit_diffs.values():
                targets.extend(float(m[key]) for m in pairs.pair_meta)
            y = np.asarray(targets, dtype=np.float64)
            if np.std(y) > 1e-12:
                dirs.append(normalize(X.T @ ((y - y.mean()) / y.std())))

    if "dataset" in confounds:
        for pairs in unit_diffs.values():
            dirs.append(normalize(pairs.D[:, unit_idx].mean(axis=0)))

    return [d for d in dirs if np.linalg.norm(d) > 1e-12]


def residualize(direction: np.ndarray, confounds: list[np.ndarray]) -> np.ndarray:
    d = normalize(direction)
    for c in confounds:
        c = normalize(c)
        d = d - float(np.dot(d, c)) * c
        d = normalize(d)
    return d


def unit_objective(aurocs: list[float], spec: ExperimentSpec) -> float:
    if not aurocs:
        return float("-inf")
    vals = np.asarray(aurocs, dtype=np.float64)
    if spec.method in {"irm", "vrex"} or spec.layer_objective == "variance_penalized":
        return float(vals.mean() - spec.variance_penalty * vals.var())
    if spec.layer_objective == "min":
        return float(vals.min())
    if spec.layer_objective == "median":
        return float(np.median(vals))
    if spec.layer_objective == "harmonic":
        positive = vals[vals > 0]
        return float(len(positive) / np.sum(1.0 / positive)) if len(positive) else 0.0
    return float(vals.mean())


def length_spearman(selected: list[tuple[int, np.ndarray, float]], datasets: dict[str, PairDiffs]) -> float:
    rhos = []
    for pairs in datasets.values():
        lengths = np.asarray([m["response_length_delta"] for m in pairs.pair_meta], dtype=np.float64)
        if len(lengths) < 3 or np.std(lengths) <= 1e-12:
            continue
        scores = score_pairs(selected, pairs)
        rho = spearmanr(lengths, scores).statistic
        if np.isfinite(rho):
            rhos.append(abs(float(rho)))
    return float(np.mean(rhos)) if rhos else 0.0


def dataset_identity_penalty(selected: list[tuple[int, np.ndarray, float]], datasets: dict[str, PairDiffs]) -> float:
    means = []
    for pairs in datasets.values():
        scores = score_pairs(selected, pairs)
        if np.std(scores) > 1e-12:
            scores = (scores - scores.mean()) / scores.std()
        means.append(float(np.mean(scores)))
    return float(min(np.std(means), 1.0)) if len(means) > 1 else 0.0

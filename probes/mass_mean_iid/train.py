"""
Train covariance-corrected mass-mean probes from paired activations.

For each (lie, truth) pair, form d = h_lie - h_truth. The feature
direction is mean(d). The IID decision direction is Sigma^-1 mean(d),
where Sigma is the within-class covariance implied by augmented pair
diffs [+d, -d]. This is the covariance-corrected mass-mean probe,
equivalent to Fisher LDA under the usual homoscedastic Gaussian setup.

Input: activations/{name}.pt + paired dataset JSONs
Output: probes/mass_mean_iid/{name}_probe.pt per dataset, results.json
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import fire
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (  # noqa: E402
    DEFAULT_MODEL_TAG,
    TRAIN_DATASETS,
    activation_dirname,
    resolve_dataset_path_for_activation,
    resolve_model,
    validate_dataset_provenance,
)


def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-12 else np.zeros_like(v)


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


def cov_corrected_direction(D, ridge=1e-4):
    mu = D.mean(axis=0)
    feature = normalize(mu)

    X_c = D - mu
    n = max(X_c.shape[0], 1)
    _, s, Vt = np.linalg.svd(X_c, full_matrices=False)
    eigvals = (s ** 2) / n
    mu_proj = Vt @ mu

    # Apply (V diag(eigvals) V^T + ridge I)^-1 to mu without forming d x d covariance.
    coeffs = mu_proj * (1.0 / (eigvals + ridge) - 1.0 / ridge)
    decision = Vt.T @ coeffs + mu / ridge
    decision = normalize(decision)
    if np.dot(mu, decision) < 0:
        decision = -decision
    return feature, decision


def select_direction(feature, decision, score_mode):
    if score_mode == "feature":
        return feature
    if score_mode == "iid":
        return decision
    raise ValueError(f"unknown score_mode: {score_mode}")


def eval_direction(D_train, D_test, ridge, score_mode):
    feature, decision = cov_corrected_direction(D_train, ridge)
    direction = select_direction(feature, decision, score_mode)
    scores = D_test @ direction
    scores_all = np.concatenate([scores, -scores])
    labels_all = np.concatenate([np.ones(len(D_test)), np.zeros(len(D_test))])
    return roc_auc_score(labels_all, scores_all)


def train(
    data_dir="../..",
    activations_dir=None,
    output_dir=".",
    n_splits=5,
    ridge=1e-4,
    score_mode="iid",
    model=DEFAULT_MODEL_TAG,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_results = {}
    cli_model_tag, _ = resolve_model(model) if model else ("", "")
    if activations_dir is None:
        activations_dir = Path(data_dir) / activation_dirname(cli_model_tag)
    activations_dir = Path(activations_dir)
    data_dir = Path(data_dir)

    model_tag = ""
    model_id = ""

    for name, (filename, label_map) in TRAIN_DATASETS.items():
        act_path = activations_dir / f"{name}.pt"
        if not act_path.exists():
            print(f"Skipping {name}: missing files")
            continue

        saved = torch.load(act_path, weights_only=False)
        if not model_tag:
            model_tag = saved.get("model_tag", cli_model_tag)
            model_id = saved.get("model_id", "")
        data_path = resolve_dataset_path_for_activation(data_dir, filename, saved.get("model_tag", cli_model_tag), saved)
        if not data_path.exists():
            print(f"Skipping {name}: missing dataset {data_path}")
            continue
        data = json.load(open(data_path))[:len(saved["activations"])]
        validate_dataset_provenance(saved, data, name)
        pair_diffs, pair_ids = get_pair_diffs(saved["activations"], data, label_map)
        n_pairs, n_layers, hidden_dim = pair_diffs.shape

        print(f"\n{name}: {n_pairs} pairs, {n_layers} layers, {hidden_dim} dim")

        layer_results = []
        for layer in range(n_layers):
            D = pair_diffs[:, layer]
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            aucs = []

            for train_idx, test_idx in kf.split(np.arange(n_pairs)):
                aucs.append(eval_direction(D[train_idx], D[test_idx], ridge, score_mode))

            layer_results.append({"layer": layer + 1, "auroc": float(np.mean(aucs))})

        best = max(layer_results, key=lambda r: r["auroc"])
        print(f"  best: layer {best['layer']}, AUROC={best['auroc']:.4f}")

        all_feature_directions = {}
        all_decision_directions = {}
        all_directions = {}
        for layer in range(n_layers):
            feature, decision = cov_corrected_direction(pair_diffs[:, layer], ridge)
            all_feature_directions[layer] = feature
            all_decision_directions[layer] = decision
            all_directions[layer] = select_direction(feature, decision, score_mode)

        best_idx = best["layer"] - 1
        direction = all_directions[best_idx]

        probe_fname = f"{name}_probe_{model_tag}.pt" if model_tag else f"{name}_probe.pt"
        torch.save({
            "direction": direction,
            "feature_direction": all_feature_directions[best_idx],
            "decision_direction": all_decision_directions[best_idx],
            "best_layer": best["layer"],
            "all_directions": all_directions,
            "all_feature_directions": all_feature_directions,
            "all_decision_directions": all_decision_directions,
            "layer_results": layer_results,
            "n_pairs": n_pairs,
            "pair_ids": pair_ids,
            "model_tag": model_tag,
            "model_id": model_id,
            "probe_type": "mass_mean_iid",
            "score_mode": score_mode,
            "ridge": ridge,
        }, Path(output_dir) / probe_fname)

        all_results[name] = layer_results

    results_fname = f"results_{model_tag}.json" if model_tag else "results.json"
    with open(Path(output_dir) / results_fname, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved to {output_dir}/")


if __name__ == "__main__":
    fire.Fire(train)

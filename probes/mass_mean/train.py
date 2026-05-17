"""
Train mass-mean probes from paired activations.

Approach: for each (lie, truth) pair, form d = h_lie - h_truth.
Direction = normalize(mean(d)) per layer. No logistic regression.
Cross-validation splits on pair index to avoid leakage.

Input: activations/{name}.pt + paired dataset JSONs
Output: probes/mass_mean/{name}_probe.pt per dataset, results.json
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    DEFAULT_MODEL_TAG,
    TRAIN_DATASETS,
    activation_dirname,
    resolve_dataset_path_for_activation,
    resolve_model,
    validate_dataset_provenance,
)


def normalize(v, eps=1e-12):
    norm = np.linalg.norm(v)
    if not np.isfinite(norm) or norm <= eps:
        return np.zeros_like(v)
    return v / norm


def finite_rows(D):
    return np.isfinite(D).all(axis=1)


def augmented_auroc_from_scores(scores):
    scores = np.asarray(scores)
    mask = np.isfinite(scores)
    if not np.any(mask):
        return 0.5
    scores = scores[mask]
    labels = np.ones(len(scores))
    scores_all = np.concatenate([scores, -scores])
    labels_all = np.concatenate([labels, np.zeros(len(scores))])
    if np.all(scores_all == scores_all[0]):
        return 0.5
    return roc_auc_score(labels_all, scores_all)


def mass_mean_direction(D):
    mask = finite_rows(D)
    if not np.any(mask):
        return np.zeros(D.shape[1], dtype=D.dtype)
    return normalize(D[mask].mean(axis=0))


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


def train(data_dir="../..", activations_dir=None, output_dir=".", n_splits=5, model=DEFAULT_MODEL_TAG):
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
            mask = finite_rows(D)
            D_finite = D[mask]
            dropped_nonfinite = int((~mask).sum())
            if len(D_finite) < n_splits:
                layer_results.append({
                    "layer": layer + 1,
                    "auroc": 0.5,
                    "n_dropped_nonfinite": dropped_nonfinite,
                    "degenerate": True,
                })
                continue

            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            aucs = []

            for train_idx, test_idx in kf.split(np.arange(len(D_finite))):
                direction = mass_mean_direction(D_finite[train_idx])
                scores = D_finite[test_idx] @ direction
                aucs.append(augmented_auroc_from_scores(scores))

            auroc = np.mean(aucs)
            layer_results.append({
                "layer": layer + 1,
                "auroc": float(auroc),
                "n_dropped_nonfinite": dropped_nonfinite,
                "degenerate": bool(all(a == 0.5 for a in aucs)),
            })

        best = max(layer_results, key=lambda r: r["auroc"])
        print(f"  best: layer {best['layer']}, AUROC={best['auroc']:.4f}")

        all_directions = {}
        for layer in range(n_layers):
            all_directions[layer] = mass_mean_direction(pair_diffs[:, layer])

        best_idx = best["layer"] - 1
        direction = all_directions[best_idx]

        probe_fname = f"{name}_probe_{model_tag}.pt" if model_tag else f"{name}_probe.pt"
        torch.save({
            "direction": direction,
            "best_layer": best["layer"],
            "all_directions": all_directions,
            "layer_results": layer_results,
            "n_pairs": n_pairs,
            "pair_ids": pair_ids,
            "model_tag": model_tag,
            "model_id": model_id,
        }, Path(output_dir) / probe_fname)

        all_results[name] = layer_results

    results_fname = f"results_{model_tag}.json" if model_tag else "results.json"
    with open(Path(output_dir) / results_fname, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved to {output_dir}/")


if __name__ == "__main__":
    fire.Fire(train)

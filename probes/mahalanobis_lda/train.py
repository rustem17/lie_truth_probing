"""
Train Mahalanobis LDA probes from paired activations.

Approach: for each (lie, truth) pair, form d = h_lie - h_truth.
Direction via Fisher LDA: w = Sw^-1 * mean(D), where Sw is
within-class scatter of augmented pair diffs [D, -D].
Uses SVD + Woodbury identity for efficient d >> n inversion
without forming the d x d scatter matrix.
Cross-validation splits on pair index to avoid leakage.

Input: activations/{name}.pt + paired dataset JSONs
Output: probes/mahalanobis_lda/{name}_probe.pt per dataset, results.json
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import KFold
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
from probes.torch_accel import (
    augmented_auroc_from_scores,
    normalize_tensor,
    resolve_device,
    tensor_to_numpy,
    to_float_tensor,
    use_torch_device,
)


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


def fisher_lda(D, ridge=1e-4):
    mu = D.mean(axis=0)
    X_c = D - mu
    n, d = X_c.shape
    U, s, Vt = np.linalg.svd(X_c, full_matrices=False)
    s2_n = s ** 2 / n
    mu_proj = Vt @ mu
    coeffs = mu_proj * (1.0 / (s2_n + ridge) - 1.0 / ridge)
    w = Vt.T @ coeffs + mu / ridge
    w = w / np.linalg.norm(w)
    if mu @ w < 0:
        w = -w
    return w


def fisher_lda_torch(D, ridge=1e-4):
    mu = D.mean(dim=0)
    X_c = D - mu
    n = X_c.shape[0]
    _, s, Vh = torch.linalg.svd(X_c, full_matrices=False)
    s2_n = s ** 2 / n
    mu_proj = Vh @ mu
    coeffs = mu_proj * (1.0 / (s2_n + ridge) - 1.0 / ridge)
    w = Vh.T @ coeffs + mu / ridge
    w = normalize_tensor(w)
    if torch.dot(mu, w) < 0:
        w = -w
    return w


def train(data_dir="../..", activations_dir=None, output_dir=".", n_splits=5, ridge=1e-4, model=DEFAULT_MODEL_TAG, device="auto"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_results = {}
    cli_model_tag, _ = resolve_model(model) if model else ("", "")
    if activations_dir is None:
        activations_dir = Path(data_dir) / activation_dirname(cli_model_tag)
    activations_dir = Path(activations_dir)
    data_dir = Path(data_dir)
    torch_device = resolve_device(device)
    print(f"Device: {torch_device}")

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
                if use_torch_device(torch_device):
                    D_train_t = to_float_tensor(D[train_idx], torch_device)
                    D_test_t = to_float_tensor(D[test_idx], torch_device)
                    direction = fisher_lda_torch(D_train_t, ridge)
                    scores = D_test_t @ direction
                else:
                    direction = fisher_lda(D[train_idx], ridge)
                    scores = D[test_idx] @ direction
                aucs.append(augmented_auroc_from_scores(scores))

            auroc = np.mean(aucs)
            layer_results.append({"layer": layer + 1, "auroc": float(auroc)})

        best = max(layer_results, key=lambda r: r["auroc"])
        print(f"  best: layer {best['layer']}, AUROC={best['auroc']:.4f}")

        best_idx = best["layer"] - 1
        if use_torch_device(torch_device):
            direction = tensor_to_numpy(fisher_lda_torch(to_float_tensor(pair_diffs[:, best_idx], torch_device), ridge))
        else:
            direction = fisher_lda(pair_diffs[:, best_idx], ridge)

        probe_fname = f"{name}_probe_{model_tag}.pt" if model_tag else f"{name}_probe.pt"
        torch.save({
            "direction": direction,
            "best_layer": best["layer"],
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

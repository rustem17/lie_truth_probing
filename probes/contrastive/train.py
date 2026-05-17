"""
Train contrastive probes from paired activations.

Approach: for each (lie, truth) pair, form d = h_lie - h_truth.
Augment with +d -> 1, -d -> 0. Train logistic regression per layer.
Cross-validation splits on pair index to avoid leakage.

Input: activations/{name}.pt + paired dataset JSONs
Output: probes/contrastive/{name}_probe.pt per dataset, results.json
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score
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
from probes.torch_accel import normalize_tensor, resolve_device, tensor_to_numpy, to_float_tensor, use_torch_device


def fit_torch_logistic_direction(D, max_iter=100, logistic_l2=1e-4):
    X = torch.cat([D, -D], dim=0)
    y = torch.cat([
        torch.ones(D.shape[0], device=D.device),
        torch.zeros(D.shape[0], device=D.device),
    ])
    w = torch.zeros(D.shape[1], dtype=D.dtype, device=D.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([w], max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        logits = X @ w
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        if logistic_l2:
            loss = loss + 0.5 * logistic_l2 * torch.sum(w * w)
        loss.backward()
        return loss

    optimizer.step(closure)
    return normalize_tensor(w.detach())


def eval_torch_logistic(D_train, D_test, max_iter=100, logistic_l2=1e-4):
    direction = fit_torch_logistic_direction(D_train, max_iter=max_iter, logistic_l2=logistic_l2)
    X_test = torch.cat([D_test, -D_test], dim=0)
    y_test = np.array([1] * D_test.shape[0] + [0] * D_test.shape[0])
    probs = torch.sigmoid(X_test @ direction).detach().cpu().numpy()
    return (
        roc_auc_score(y_test, probs),
        accuracy_score(y_test, probs >= 0.5),
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


def train(
    data_dir="../..",
    activations_dir=None,
    output_dir=".",
    n_splits=5,
    model=DEFAULT_MODEL_TAG,
    device="auto",
    solver="auto",
    torch_max_iter=100,
    logistic_l2=1e-4,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_results = {}
    cli_model_tag, _ = resolve_model(model) if model else ("", "")
    if activations_dir is None:
        activations_dir = Path(data_dir) / activation_dirname(cli_model_tag)
    activations_dir = Path(activations_dir)
    data_dir = Path(data_dir)
    torch_device = resolve_device(device)
    if solver == "auto":
        solver = "torch" if use_torch_device(torch_device) else "sklearn"
    if solver not in {"sklearn", "torch"}:
        raise ValueError("solver must be 'auto', 'sklearn', or 'torch'")
    if solver == "torch" and not use_torch_device(torch_device):
        print("Device is CPU; using sklearn logistic regression instead of torch.")
        solver = "sklearn"
    print(f"Device: {torch_device} ({solver} solver)")

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
            if solver == "sklearn":
                X_aug = np.concatenate([D, -D], axis=0)
                y_aug = np.array([1] * n_pairs + [0] * n_pairs)
            else:
                D_t = to_float_tensor(D, torch_device)

            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            aucs, accs = [], []

            for train_idx, test_idx in kf.split(np.arange(n_pairs)):
                if solver == "torch":
                    train_idx_t = torch.as_tensor(train_idx, dtype=torch.long, device=torch_device)
                    test_idx_t = torch.as_tensor(test_idx, dtype=torch.long, device=torch_device)
                    auroc_fold, acc_fold = eval_torch_logistic(
                        D_t[train_idx_t],
                        D_t[test_idx_t],
                        max_iter=torch_max_iter,
                        logistic_l2=logistic_l2,
                    )
                    aucs.append(auroc_fold)
                    accs.append(acc_fold)
                else:
                    train_mask = np.concatenate([train_idx, train_idx + n_pairs])
                    test_mask = np.concatenate([test_idx, test_idx + n_pairs])

                    clf = LogisticRegression(max_iter=1000, fit_intercept=False)
                    clf.fit(X_aug[train_mask], y_aug[train_mask])
                    probs = clf.predict_proba(X_aug[test_mask])[:, 1]
                    aucs.append(roc_auc_score(y_aug[test_mask], probs))
                    accs.append(accuracy_score(y_aug[test_mask], clf.predict(X_aug[test_mask])))

            auroc = np.mean(aucs)
            acc = np.mean(accs)
            layer_results.append({"layer": layer + 1, "auroc": float(auroc), "accuracy": float(acc)})

        best = max(layer_results, key=lambda r: r["auroc"])
        print(f"  best: layer {best['layer']}, AUROC={best['auroc']:.4f}, acc={best['accuracy']:.4f}")

        all_directions = {}
        for layer in range(n_layers):
            D_layer = pair_diffs[:, layer]
            if solver == "torch":
                d = tensor_to_numpy(fit_torch_logistic_direction(
                    to_float_tensor(D_layer, torch_device),
                    max_iter=torch_max_iter,
                    logistic_l2=logistic_l2,
                ))
            else:
                X_a = np.concatenate([D_layer, -D_layer], axis=0)
                y_a = np.array([1] * n_pairs + [0] * n_pairs)
                clf_layer = LogisticRegression(max_iter=1000, fit_intercept=False)
                clf_layer.fit(X_a, y_a)
                norm = np.linalg.norm(clf_layer.coef_[0])
                d = clf_layer.coef_[0] / norm if np.isfinite(norm) and norm > 1e-12 else np.zeros_like(clf_layer.coef_[0])
            all_directions[layer] = d

        best_idx = best["layer"] - 1
        direction = all_directions[best_idx]

        probe_fname = f"{name}_probe_{model_tag}.pt" if model_tag else f"{name}_probe.pt"
        torch.save({
            "direction": direction,
            "coef": None,
            "best_layer": best["layer"],
            "all_directions": all_directions,
            "layer_results": layer_results,
            "n_pairs": n_pairs,
            "pair_ids": pair_ids,
            "model_tag": model_tag,
            "model_id": model_id,
            "solver": solver,
            "torch_max_iter": torch_max_iter if solver == "torch" else None,
            "logistic_l2": logistic_l2 if solver == "torch" else None,
        }, Path(output_dir) / probe_fname)

        all_results[name] = layer_results

    results_fname = f"results_{model_tag}.json" if model_tag else "results.json"
    with open(Path(output_dir) / results_fname, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved to {output_dir}/")


if __name__ == "__main__":
    fire.Fire(train)

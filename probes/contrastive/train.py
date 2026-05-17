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


def normalize_rows(W, eps=1e-12):
    norms = torch.linalg.vector_norm(W, dim=1, keepdim=True)
    safe = torch.isfinite(norms) & (norms > eps)
    return torch.where(safe, W / norms.clamp_min(eps), torch.zeros_like(W))


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


def fit_torch_logistic_directions_batched(D, max_iter=100, logistic_l2=1e-4, lr=0.1):
    """Fit one logistic direction per layer in a single torch optimizer loop.

    D has shape (n_pairs, n_layers, hidden_dim). The augmented -D examples are
    handled analytically: BCE(D@w, 1) and BCE(-D@w, 0) are both softplus(-D@w).
    """
    W = D.mean(dim=0).clone()
    W = torch.where(torch.isfinite(W), W, torch.zeros_like(W))
    W.requires_grad_(True)
    optimizer = torch.optim.AdamW([W], lr=lr, weight_decay=0.0)

    for _ in range(max_iter):
        optimizer.zero_grad(set_to_none=True)
        logits = torch.einsum("nld,ld->nl", D, W)
        loss = torch.nn.functional.softplus(-logits).mean()
        if logistic_l2:
            loss = loss + 0.5 * logistic_l2 * torch.sum(W * W, dim=1).mean()
        loss.backward()
        optimizer.step()

    return normalize_rows(W.detach())


def evaluate_batched_directions(D_test, directions):
    scores = torch.einsum("nld,ld->nl", D_test, directions).detach().cpu().numpy()
    y = np.array([1] * scores.shape[0] + [0] * scores.shape[0])
    aucs, accs = [], []
    for layer in range(scores.shape[1]):
        logits = np.concatenate([scores[:, layer], -scores[:, layer]])
        aucs.append(roc_auc_score(y, logits))
        accs.append(accuracy_score(y, logits >= 0.0))
    return aucs, accs


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
    torch_lr=0.1,
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
        solver = "torch_batched" if use_torch_device(torch_device) else "sklearn"
    if solver == "torch":
        solver = "torch_batched"
    if solver not in {"sklearn", "torch_batched", "torch_lbfgs"}:
        raise ValueError("solver must be 'auto', 'sklearn', 'torch_batched', 'torch_lbfgs', or 'torch'")
    if solver in {"torch_batched", "torch_lbfgs"} and not use_torch_device(torch_device):
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
        if solver == "torch_batched":
            D_all_t = to_float_tensor(pair_diffs, torch_device)
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            aucs_by_layer = [[] for _ in range(n_layers)]
            accs_by_layer = [[] for _ in range(n_layers)]

            for train_idx, test_idx in kf.split(np.arange(n_pairs)):
                train_idx_t = torch.as_tensor(train_idx, dtype=torch.long, device=torch_device)
                test_idx_t = torch.as_tensor(test_idx, dtype=torch.long, device=torch_device)
                directions = fit_torch_logistic_directions_batched(
                    D_all_t[train_idx_t],
                    max_iter=torch_max_iter,
                    logistic_l2=logistic_l2,
                    lr=torch_lr,
                )
                fold_aucs, fold_accs = evaluate_batched_directions(D_all_t[test_idx_t], directions)
                for layer in range(n_layers):
                    aucs_by_layer[layer].append(fold_aucs[layer])
                    accs_by_layer[layer].append(fold_accs[layer])

            for layer in range(n_layers):
                layer_results.append({
                    "layer": layer + 1,
                    "auroc": float(np.mean(aucs_by_layer[layer])),
                    "accuracy": float(np.mean(accs_by_layer[layer])),
                })

            all_dirs_t = fit_torch_logistic_directions_batched(
                D_all_t,
                max_iter=torch_max_iter,
                logistic_l2=logistic_l2,
                lr=torch_lr,
            )
            all_directions = {
                layer: tensor_to_numpy(all_dirs_t[layer])
                for layer in range(n_layers)
            }
            del D_all_t, all_dirs_t
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()

        else:
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
                    if solver == "torch_lbfgs":
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

            all_directions = {}
            for layer in range(n_layers):
                D_layer = pair_diffs[:, layer]
                if solver == "torch_lbfgs":
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

        best = max(layer_results, key=lambda r: r["auroc"])
        print(f"  best: layer {best['layer']}, AUROC={best['auroc']:.4f}, acc={best['accuracy']:.4f}")

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
            "torch_max_iter": torch_max_iter if solver in {"torch_batched", "torch_lbfgs"} else None,
            "logistic_l2": logistic_l2 if solver in {"torch_batched", "torch_lbfgs"} else None,
            "torch_lr": torch_lr if solver == "torch_batched" else None,
        }, Path(output_dir) / probe_fname)

        all_results[name] = layer_results

    results_fname = f"results_{model_tag}.json" if model_tag else "results.json"
    with open(Path(output_dir) / results_fname, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved to {output_dir}/")


if __name__ == "__main__":
    fire.Fire(train)

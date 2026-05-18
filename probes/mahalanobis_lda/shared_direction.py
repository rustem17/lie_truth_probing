"""
Shared Mahalanobis LDA direction analysis across deception conditions.

Per-dataset: Fisher LDA direction (Sw^-1 * mean) at each layer.
Cross-dataset: by default, pool within-class scatter across environments and
solve B v = lambda Sw v via a PCA-reduced generalized eigenvalue problem. Use
--shared_mode average to average per-dataset Fisher LDA directions for a faster
sweep approximation.

Input: activations/{name}.pt + paired dataset JSONs
Output: probes/mahalanobis_lda/shared_direction.pt
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from scipy.linalg import eigh
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
    finite_rows_tensor,
    normalize_tensor,
    resolve_device,
    tensor_to_numpy,
    to_float_tensor,
    use_torch_device,
)


def normalize(v, eps=1e-12):
    norm = np.linalg.norm(v)
    if not np.isfinite(norm) or norm <= eps:
        return np.zeros_like(v)
    return v / norm


def finite_rows(D):
    return np.isfinite(D).all(axis=1)


def cosine_sim(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if not np.isfinite(denom) or denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


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


def load_all(data_dir, activations_dir, cli_model_tag=""):
    diffs = {}
    model_tag = ""
    model_id = ""
    for name, (filename, label_map) in TRAIN_DATASETS.items():
        act_path = Path(activations_dir) / f"{name}.pt"
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
        diffs[name] = {"D": pair_diffs, "n_pairs": len(pair_ids)}
    return diffs, model_tag, model_id


def fisher_lda(D, ridge=1e-4):
    mask = finite_rows(D)
    if not np.any(mask):
        return np.zeros(D.shape[1], dtype=D.dtype)
    D = D[mask]
    mu = D.mean(axis=0)
    X_c = D - mu
    n, d = X_c.shape
    U, s, Vt = np.linalg.svd(X_c, full_matrices=False)
    s2_n = s ** 2 / n
    mu_proj = Vt @ mu
    coeffs = mu_proj * (1.0 / (s2_n + ridge) - 1.0 / ridge)
    w = Vt.T @ coeffs + mu / ridge
    w = normalize(w)
    if mu @ w < 0:
        w = -w
    return w


def fisher_lda_torch(D, ridge=1e-4):
    mask = finite_rows_tensor(D)
    if not torch.any(mask):
        return torch.zeros(D.shape[1], dtype=D.dtype, device=D.device)
    D = D[mask]
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


def fisher_lda_device(D, ridge=1e-4, device="cpu"):
    torch_device = resolve_device(device)
    if use_torch_device(torch_device):
        try:
            return tensor_to_numpy(fisher_lda_torch(to_float_tensor(D, torch_device), ridge))
        except RuntimeError as exc:
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()
                print(f"CUDA Fisher LDA failed ({exc}); falling back to CPU", flush=True)
            else:
                raise
    return fisher_lda(D, ridge)


def multi_env_lda_numpy(diff_list, ridge=1e-4, pca_var=0.95):
    diff_list = [D[finite_rows(D)] for D in diff_list if np.any(finite_rows(D))]
    if not diff_list:
        return np.zeros(0)
    mus = [D.mean(axis=0) for D in diff_list]
    X_c = np.vstack([D - mu for D, mu in zip(diff_list, mus)])
    n_total = X_c.shape[0]

    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    if pca_var is not None:
        total_var = np.sum(S ** 2)
        if not np.isfinite(total_var) or total_var <= 1e-12:
            return normalize(np.mean(mus, axis=0))
        var_ratio = np.cumsum(S ** 2) / total_var
        r = int(np.searchsorted(var_ratio, pca_var)) + 1
        r = min(r, len(S))
    else:
        r = len(S)

    P = Vt[:r].T
    Sw_r = np.diag(S[:r] ** 2 / n_total) + ridge * np.eye(r)
    B_r = sum(np.outer(P.T @ mu, P.T @ mu) for mu in mus)

    eigvals, eigvecs = eigh(B_r, Sw_r)
    w = P @ eigvecs[:, -1]
    w = normalize(w)

    if sum(mu @ w for mu in mus) < 0:
        w = -w
    return w


def multi_env_lda_torch(diff_list, ridge=1e-4, pca_var=0.95, device="cuda"):
    torch_device = resolve_device(device)
    arrays = [D[finite_rows(D)] for D in diff_list if np.any(finite_rows(D))]
    if not arrays:
        return np.zeros(0)

    tensors = [to_float_tensor(D, torch_device) for D in arrays]
    mus = [D.mean(dim=0) for D in tensors]
    X_c = torch.cat([D - mu for D, mu in zip(tensors, mus)], dim=0)
    n_total = X_c.shape[0]

    _, S, Vh = torch.linalg.svd(X_c, full_matrices=False)
    if pca_var is not None:
        total_var = torch.sum(S ** 2)
        if not torch.isfinite(total_var) or total_var <= 1e-12:
            return tensor_to_numpy(normalize_tensor(torch.stack(mus).mean(dim=0)))
        var_ratio = torch.cumsum(S ** 2, dim=0) / total_var
        cutoff = torch.as_tensor(float(pca_var), dtype=var_ratio.dtype, device=torch_device)
        r = int(torch.searchsorted(var_ratio, cutoff).item()) + 1
        r = min(r, S.numel())
    else:
        r = S.numel()

    P = Vh[:r].T
    sw_diag = S[:r] ** 2 / n_total + ridge
    mu_r = torch.stack([P.T @ mu for mu in mus])
    B_r = mu_r.T @ mu_r

    inv_sqrt_sw = torch.rsqrt(sw_diag)
    whitened = inv_sqrt_sw[:, None] * B_r * inv_sqrt_sw[None, :]
    whitened = (whitened + whitened.T) / 2
    _, eigvecs = torch.linalg.eigh(whitened)
    z = inv_sqrt_sw * eigvecs[:, -1]
    w = P @ z
    w = normalize_tensor(w)

    if sum(torch.dot(mu, w) for mu in mus) < 0:
        w = -w
    return tensor_to_numpy(w)


def multi_env_lda(diff_list, ridge=1e-4, pca_var=0.95, device="cpu"):
    torch_device = resolve_device(device)
    if use_torch_device(torch_device):
        try:
            return multi_env_lda_torch(diff_list, ridge=ridge, pca_var=pca_var, device=torch_device)
        except RuntimeError as exc:
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()
                print(f"CUDA multi-env LDA failed ({exc}); falling back to CPU", flush=True)
            else:
                raise
    return multi_env_lda_numpy(diff_list, ridge, pca_var)


def train_all_directions(diffs, train_names, ridge=1e-4, device="cpu"):
    directions = {}
    torch_device = resolve_device(device)
    for name in train_names:
        if name not in diffs:
            continue
        D = diffs[name]["D"]
        n_pairs, n_layers, _ = D.shape
        dirs = []
        for layer in range(n_layers):
            dirs.append(fisher_lda_device(D[:, layer], ridge, torch_device))
            if (layer + 1 == 1) or ((layer + 1) % 10 == 0) or (layer + 1 == n_layers):
                print(f"  {name}: Fisher layer {layer + 1}/{n_layers}", flush=True)
        directions[name] = np.stack(dirs)
        print(f"{name}: {n_pairs} pairs, {n_layers} layers")
    return directions


def aggregate_directions(dir_list):
    dir_list = [d for d in dir_list if np.isfinite(d).all()]
    if not dir_list:
        return np.zeros(0)
    return normalize(np.mean(dir_list, axis=0))


def transfer_auroc(direction, diffs_tgt_layer):
    if direction.size == 0 or not np.isfinite(direction).all():
        return 0.5
    mask = finite_rows(diffs_tgt_layer)
    if not np.any(mask):
        return 0.5
    scores = diffs_tgt_layer[mask] @ direction
    finite_scores = np.isfinite(scores)
    if not np.any(finite_scores):
        return 0.5
    scores = scores[finite_scores]
    scores_all = np.concatenate([scores, -scores])
    labels_all = np.concatenate([np.ones(len(scores)), np.zeros(len(scores))])
    if np.all(scores_all == scores_all[0]):
        return 0.5
    return float(roc_auc_score(labels_all, scores_all))


def cross_transfer(directions, diffs, layer, transfer_pairs):
    results = {}
    for src, tgt in transfer_pairs:
        results[f"{src}\u2192{tgt}"] = transfer_auroc(directions[src][layer], diffs[tgt]["D"][:, layer])
    return results


def cross_transfer_all_layers(directions, diffs, n_layers, transfer_pairs):
    transfer_pairs = [(s, t) for s, t in transfer_pairs if s in directions and t in diffs]

    per_layer = {f"{s}\u2192{t}": [] for s, t in transfer_pairs}
    for layer in range(n_layers):
        for src, tgt in transfer_pairs:
            per_layer[f"{src}\u2192{tgt}"].append(
                transfer_auroc(directions[src][layer], diffs[tgt]["D"][:, layer])
            )

    mean_transfer = [
        np.mean([per_layer[k][l] for k in per_layer]) for l in range(n_layers)
    ]
    return per_layer, mean_transfer


def build_shared_direction(diffs, directions, names, layer, ridge, pca_var, shared_mode, device="cpu"):
    if shared_mode == "average":
        return aggregate_directions([directions[n][layer] for n in names])
    if shared_mode == "multi_env":
        return multi_env_lda([diffs[n]["D"][:, layer] for n in names], ridge, pca_var, device)
    raise ValueError("shared_mode must be 'average' or 'multi_env'")


def analyze(data_dir="../..", activations_dir=None, output_dir=".", datasets=None,
            ridge=1e-4, pca_var=0.95, shared_mode="multi_env", model=DEFAULT_MODEL_TAG, device="auto"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cli_model_tag, _ = resolve_model(model) if model else ("", "")
    if activations_dir is None:
        activations_dir = Path(data_dir) / activation_dirname(cli_model_tag)
    torch_device = resolve_device(device)
    print(f"Device: {torch_device}")

    diffs, model_tag, model_id = load_all(Path(data_dir), Path(activations_dir), cli_model_tag)
    if datasets:
        keep = set(datasets) if isinstance(datasets, (list, tuple)) else {d.strip() for d in datasets.split(",")}
        diffs = {k: v for k, v in diffs.items() if k in keep}

    directions = train_all_directions(diffs, sorted(diffs.keys()), ridge, torch_device)
    n_layers = next(iter(directions.values())).shape[0]

    active_names = sorted(directions.keys())
    active_pairs = [(a, b) for i, a in enumerate(active_names) for b in active_names[i+1:]]
    transfer_pairs = [(s, t) for s in active_names for t in active_names if s != t]

    layer_sims = {f"{a}_vs_{b}": [] for a, b in active_pairs}
    for layer in range(n_layers):
        for a, b in active_pairs:
            layer_sims[f"{a}_vs_{b}"].append(cosine_sim(directions[a][layer], directions[b][layer]))

    mean_sims = [
        np.mean([layer_sims[f"{a}_vs_{b}"][l] for a, b in active_pairs])
        for l in range(n_layers)
    ]
    best_layer_cos = int(np.argmax(mean_sims))

    print(f"\n--- cosine similarity ---")
    print(f"best layer: {best_layer_cos + 1} (mean cosine = {mean_sims[best_layer_cos]:.4f})")
    for a, b in active_pairs:
        print(f"  {a} vs {b}: {layer_sims[f'{a}_vs_{b}'][best_layer_cos]:.4f}")

    per_layer_transfer, mean_transfer = cross_transfer_all_layers(directions, diffs, n_layers, transfer_pairs)
    best_layer_transfer = int(np.argmax(mean_transfer))

    print(f"\n--- cross-condition transfer ---")
    print(f"best layer: {best_layer_transfer + 1} (mean AUROC = {mean_transfer[best_layer_transfer]:.4f})")
    transfer_at_best = cross_transfer(directions, diffs, best_layer_transfer, transfer_pairs)
    for k, v in transfer_at_best.items():
        print(f"  {k}: AUROC={v:.4f}")

    if "spontaneous_1" in directions and "sycophancy_answer" in directions:
        sp_sy_cosines = [
            cosine_sim(directions["spontaneous_1"][l], directions["sycophancy_answer"][l])
            for l in range(n_layers)
        ]
        best_sp_sy = int(np.argmax(sp_sy_cosines))

        sp_sy_transfer = []
        for layer in range(n_layers):
            a1 = transfer_auroc(directions["spontaneous_1"][layer], diffs["sycophancy_answer"]["D"][:, layer])
            a2 = transfer_auroc(directions["sycophancy_answer"][layer], diffs["spontaneous_1"]["D"][:, layer])
            sp_sy_transfer.append((a1 + a2) / 2)
        best_sp_sy_transfer = int(np.argmax(sp_sy_transfer))

        print(f"\n--- spontaneous_1 + sycophancy_answer only ---")
        print(f"best cosine layer: {best_sp_sy + 1} (cosine = {sp_sy_cosines[best_sp_sy]:.4f})")
        print(f"best transfer layer: {best_sp_sy_transfer + 1} (mean AUROC = {sp_sy_transfer[best_sp_sy_transfer]:.4f})")
        print(f"  sp\u2192syc: {transfer_auroc(directions['spontaneous_1'][best_sp_sy_transfer], diffs['sycophancy_answer']['D'][:, best_sp_sy_transfer]):.4f}")
        print(f"  syc\u2192sp: {transfer_auroc(directions['sycophancy_answer'][best_sp_sy_transfer], diffs['spontaneous_1']['D'][:, best_sp_sy_transfer]):.4f}")
    else:
        best_sp_sy_transfer = best_layer_transfer
        sp_sy_cosines = []
        sp_sy_transfer = []

    best_layer = best_layer_transfer
    shared_all = build_shared_direction(
        diffs, directions, active_names, best_layer, ridge, pca_var, shared_mode, torch_device)
    all_directions = {}
    if shared_mode == "multi_env":
        print(f"\n--- shared multi-env directions ({n_layers} layers) ---", flush=True)
    for layer in range(n_layers):
        all_directions[layer] = build_shared_direction(
            diffs, directions, active_names, layer, ridge, pca_var, shared_mode, torch_device)
        if shared_mode == "multi_env" and ((layer + 1 == 1) or ((layer + 1) % 10 == 0) or (layer + 1 == n_layers)):
            print(f"  shared layer {layer + 1}/{n_layers}", flush=True)

    sp_sy_names = [n for n in ["spontaneous_1", "sycophancy_answer"] if n in diffs]
    if len(sp_sy_names) > 1:
        shared_sp_sy = build_shared_direction(
            diffs, directions, sp_sy_names, best_sp_sy_transfer, ridge, pca_var, shared_mode, torch_device)
    elif len(sp_sy_names) == 1:
        shared_sp_sy = fisher_lda_device(diffs[sp_sy_names[0]]["D"][:, best_sp_sy_transfer], ridge, torch_device)
    else:
        shared_sp_sy = None

    out_fname = f"shared_direction_{model_tag}.pt" if model_tag else "shared_direction.pt"
    torch.save({
        "shared_direction_all": shared_all,
        "shared_direction_sp_sy": shared_sp_sy,
        "best_layer_cosine": best_layer_cos + 1,
        "best_layer_transfer": best_layer_transfer + 1,
        "best_layer_sp_sy_transfer": best_sp_sy_transfer + 1,
        "all_directions": all_directions,
        "probe_type": "mahalanobis_lda",
        "shared_mode": shared_mode,
        "device": str(torch_device),
        "ridge": ridge,
        "pca_var": pca_var,
        "model_tag": model_tag,
        "model_id": model_id,
        "cosine": {
            "per_layer_mean": [float(x) for x in mean_sims],
            "per_layer": {k: [float(x) for x in v] for k, v in layer_sims.items()},
        },
        "transfer": {
            "per_layer_mean": [float(x) for x in mean_transfer],
            "per_layer": per_layer_transfer,
        },
        "sp_sy": {
            "per_layer_cosine": [float(x) for x in sp_sy_cosines],
            "per_layer_transfer": [float(x) for x in sp_sy_transfer],
        },
        "per_layer_directions": {name: directions[name] for name in active_names},
    }, Path(output_dir) / out_fname)

    print(f"\nsaved to {output_dir}/{out_fname}")


if __name__ == "__main__":
    fire.Fire(analyze)

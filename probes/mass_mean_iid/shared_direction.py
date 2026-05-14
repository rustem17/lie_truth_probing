"""
Shared covariance-corrected mass-mean directions across deception conditions.

For each dataset and layer, compute both the feature direction mean(d) and the
IID decision direction Sigma^-1 mean(d). Cross-transfer and shared directions
use --score_mode, which defaults to the IID decision direction.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import fire
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

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


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


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


def cov_corrected_direction(D, ridge=1e-4):
    mu = D.mean(axis=0)
    feature = normalize(mu)

    X_c = D - mu
    n = max(X_c.shape[0], 1)
    _, s, Vt = np.linalg.svd(X_c, full_matrices=False)
    eigvals = (s ** 2) / n
    mu_proj = Vt @ mu
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


def train_all_directions(diffs, train_names, ridge=1e-4, score_mode="iid"):
    directions = {}
    feature_directions = {}
    decision_directions = {}
    for name in train_names:
        if name not in diffs:
            continue
        D = diffs[name]["D"]
        n_pairs, n_layers, _ = D.shape
        dirs, features, decisions = [], [], []
        for layer in range(n_layers):
            feature, decision = cov_corrected_direction(D[:, layer], ridge)
            features.append(feature)
            decisions.append(decision)
            dirs.append(select_direction(feature, decision, score_mode))
        directions[name] = np.stack(dirs)
        feature_directions[name] = np.stack(features)
        decision_directions[name] = np.stack(decisions)
        print(f"{name}: {n_pairs} pairs, {n_layers} layers")
    return directions, feature_directions, decision_directions


def transfer_auroc(direction, diffs_tgt_layer):
    scores = diffs_tgt_layer @ direction
    n = diffs_tgt_layer.shape[0]
    scores_all = np.concatenate([scores, -scores])
    labels_all = np.concatenate([np.ones(n), np.zeros(n)])
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
    mean_transfer = [np.mean([per_layer[k][l] for k in per_layer]) for l in range(n_layers)]
    return per_layer, mean_transfer


def aggregate_directions(dir_list):
    avg = np.mean(dir_list, axis=0)
    return normalize(avg)


def analyze(
    data_dir="../..",
    activations_dir=None,
    output_dir=".",
    datasets=None,
    ridge=1e-4,
    score_mode="iid",
    model=DEFAULT_MODEL_TAG,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cli_model_tag, _ = resolve_model(model) if model else ("", "")
    if activations_dir is None:
        activations_dir = Path(data_dir) / activation_dirname(cli_model_tag)

    diffs, model_tag, model_id = load_all(Path(data_dir), Path(activations_dir), cli_model_tag)
    if datasets:
        keep = set(datasets) if isinstance(datasets, (list, tuple)) else {d.strip() for d in datasets.split(",")}
        diffs = {k: v for k, v in diffs.items() if k in keep}

    directions, feature_directions, decision_directions = train_all_directions(
        diffs, sorted(diffs.keys()), ridge, score_mode
    )
    n_layers = next(iter(directions.values())).shape[0]

    active_names = sorted(directions.keys())
    active_pairs = [(a, b) for i, a in enumerate(active_names) for b in active_names[i + 1:]]
    transfer_pairs = [(s, t) for s in active_names for t in active_names if s != t]

    layer_sims = {f"{a}_vs_{b}": [] for a, b in active_pairs}
    for layer in range(n_layers):
        for a, b in active_pairs:
            layer_sims[f"{a}_vs_{b}"].append(cosine_sim(directions[a][layer], directions[b][layer]))

    mean_sims = [np.mean([layer_sims[f"{a}_vs_{b}"][l] for a, b in active_pairs]) for l in range(n_layers)]
    best_layer_cos = int(np.argmax(mean_sims))

    per_layer_transfer, mean_transfer = cross_transfer_all_layers(directions, diffs, n_layers, transfer_pairs)
    best_layer_transfer = int(np.argmax(mean_transfer))
    transfer_at_best = cross_transfer(directions, diffs, best_layer_transfer, transfer_pairs)

    print(f"\n--- cosine similarity ---")
    print(f"best layer: {best_layer_cos + 1} (mean cosine = {mean_sims[best_layer_cos]:.4f})")
    print(f"\n--- cross-condition transfer ---")
    print(f"best layer: {best_layer_transfer + 1} (mean AUROC = {mean_transfer[best_layer_transfer]:.4f})")
    for k, v in transfer_at_best.items():
        print(f"  {k}: AUROC={v:.4f}")

    shared_all = aggregate_directions([directions[n][best_layer_transfer] for n in active_names])
    all_directions = {
        layer: aggregate_directions([directions[n][layer] for n in active_names])
        for layer in range(n_layers)
    }

    out_fname = f"shared_direction_{model_tag}.pt" if model_tag else "shared_direction.pt"
    torch.save({
        "shared_direction_all": shared_all,
        "shared_direction_sp_sy": None,
        "best_layer_cosine": best_layer_cos + 1,
        "best_layer_transfer": best_layer_transfer + 1,
        "all_directions": all_directions,
        "probe_type": "mass_mean_iid",
        "score_mode": score_mode,
        "ridge": ridge,
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
        "per_layer_directions": {name: directions[name] for name in active_names},
        "per_layer_feature_directions": {name: feature_directions[name] for name in active_names},
        "per_layer_decision_directions": {name: decision_directions[name] for name in active_names},
    }, Path(output_dir) / out_fname)

    print(f"\nsaved to {output_dir}/{out_fname}")


if __name__ == "__main__":
    fire.Fire(analyze)

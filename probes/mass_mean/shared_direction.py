"""
Shared mass-mean direction analysis across deception conditions.

For each dataset, compute per-layer direction = normalize(mean(pair diffs)).
Then measure cosine similarity and cross-transfer AUROC between directions.

Input: activations/{name}.pt + paired dataset JSONs
Output: probes/mass_mean/shared_direction.pt
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
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


def augment(D):
    n = D.shape[0]
    return np.concatenate([D, -D], axis=0), np.array([1] * n + [0] * n)


def train_all_directions(diffs, train_names):
    directions = {}
    for name in train_names:
        if name not in diffs:
            continue
        D = diffs[name]["D"]
        n_pairs, n_layers, _ = D.shape
        dirs = []
        for layer in range(n_layers):
            D_layer = D[:, layer]
            mask = finite_rows(D_layer)
            if not np.any(mask):
                dirs.append(np.zeros(D_layer.shape[1], dtype=D_layer.dtype))
            else:
                dirs.append(normalize(D_layer[mask].mean(axis=0)))
        directions[name] = np.stack(dirs)
        print(f"{name}: {n_pairs} pairs, {n_layers} layers")
    return directions


def transfer_auroc(direction, diffs_tgt_layer):
    mask = finite_rows(diffs_tgt_layer)
    if not np.any(mask):
        return 0.5
    diffs_tgt_layer = diffs_tgt_layer[mask]
    scores = diffs_tgt_layer @ direction
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
        results[f"{src}→{tgt}"] = transfer_auroc(directions[src][layer], diffs[tgt]["D"][:, layer])
    return results


def cross_transfer_all_layers(directions, diffs, n_layers, transfer_pairs):
    transfer_pairs = [(s, t) for s, t in transfer_pairs if s in directions and t in diffs]

    per_layer = {f"{s}→{t}": [] for s, t in transfer_pairs}
    for layer in range(n_layers):
        for src, tgt in transfer_pairs:
            per_layer[f"{src}→{tgt}"].append(
                transfer_auroc(directions[src][layer], diffs[tgt]["D"][:, layer])
            )

    mean_transfer = [
        np.mean([per_layer[k][l] for k in per_layer]) for l in range(n_layers)
    ]
    return per_layer, mean_transfer


def analyze(data_dir="../..", activations_dir=None, output_dir=".", datasets=None, model=DEFAULT_MODEL_TAG):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cli_model_tag, _ = resolve_model(model) if model else ("", "")
    if activations_dir is None:
        activations_dir = Path(data_dir) / activation_dirname(cli_model_tag)

    diffs, model_tag, model_id = load_all(Path(data_dir), Path(activations_dir), cli_model_tag)
    if datasets:
        keep = set(datasets) if isinstance(datasets, (list, tuple)) else {d.strip() for d in datasets.split(",")}
        diffs = {k: v for k, v in diffs.items() if k in keep}

    directions = train_all_directions(diffs, sorted(diffs.keys()))
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
        print(f"  sp→syc: {transfer_auroc(directions['spontaneous_1'][best_sp_sy_transfer], diffs['sycophancy_answer']['D'][:, best_sp_sy_transfer]):.4f}")
        print(f"  syc→sp: {transfer_auroc(directions['sycophancy_answer'][best_sp_sy_transfer], diffs['spontaneous_1']['D'][:, best_sp_sy_transfer]):.4f}")
    else:
        best_sp_sy_transfer = best_layer_transfer
        sp_sy_cosines = []
        sp_sy_transfer = []

    best_layer = best_layer_transfer
    shared_all = np.mean([directions[n][best_layer] for n in active_names], axis=0)
    shared_all = normalize(shared_all)

    sp_sy_names = [n for n in ["spontaneous_1", "sycophancy_answer"] if n in directions]
    if sp_sy_names:
        shared_sp_sy = np.mean([directions[n][best_sp_sy_transfer] for n in sp_sy_names], axis=0)
        shared_sp_sy = normalize(shared_sp_sy)
    else:
        shared_sp_sy = None

    all_directions = {}
    for layer in range(n_layers):
        avg = np.mean([directions[n][layer] for n in active_names], axis=0)
        all_directions[layer] = normalize(avg)

    out_fname = f"shared_direction_{model_tag}.pt" if model_tag else "shared_direction.pt"
    torch.save({
        "shared_direction_all": shared_all,
        "shared_direction_sp_sy": shared_sp_sy,
        "best_layer_cosine": best_layer_cos + 1,
        "best_layer_transfer": best_layer_transfer + 1,
        "best_layer_sp_sy_transfer": best_sp_sy_transfer + 1,
        "all_directions": all_directions,
        "probe_type": "mass_mean",
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

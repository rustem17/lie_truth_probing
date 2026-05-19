"""
Validate IRM/V-REx probe on held-out train datasets + all validation datasets.

Input: probes/irm/irm_probe.pt + activations/{name}.pt + paired JSONs
Output: probes/irm/validation_results.json
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
    VALIDATION_MAP,
    activation_dirname,
    eval_result_metadata,
    resolve_dataset_path_for_activation,
    resolve_model,
    tagged_filename,
    validate_dataset_provenance,
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


def validate(data_dir="../..", activations_dir=None, probe_path=None, output_dir=".", model=DEFAULT_MODEL_TAG):
    cli_model_tag, _ = resolve_model(model) if model else ("", "")
    if activations_dir is None:
        activations_dir = Path(data_dir) / activation_dirname(cli_model_tag)
    activations_dir = Path(activations_dir)
    data_dir = Path(data_dir)
    if probe_path is None:
        tagged_probe = Path(output_dir) / tagged_filename("irm_probe.pt", cli_model_tag)
        probe_path = tagged_probe if tagged_probe.exists() else Path(output_dir) / "irm_probe.pt"
    probe = torch.load(probe_path, weights_only=False)
    direction = probe["direction"]
    best_layer = probe["best_layer"]
    train_envs = set(probe["envs"])
    print(f"probe: penalty={probe['config'].get('penalty', 'irm')} envs={probe['envs']} best_layer={best_layer}")

    eval_datasets = {}
    for name, val_sets in VALIDATION_MAP.items():
        for val_name, val_file, label_map in val_sets:
            eval_datasets[val_name] = (val_file, label_map, name in train_envs)
    for name, (filename, label_map) in TRAIN_DATASETS.items():
        if name not in train_envs:
            eval_datasets[name] = (filename, label_map, False)

    results = {}
    layer_idx = best_layer - 1

    for name, (filename, label_map, is_val_of_train_env) in eval_datasets.items():
        act_path = activations_dir / f"{name}.pt"
        if not act_path.exists():
            print(f"  {name}: missing files, skipping")
            continue

        saved = torch.load(act_path, weights_only=False)
        data_path = resolve_dataset_path_for_activation(data_dir, filename, saved.get("model_tag", cli_model_tag), saved)
        if not data_path.exists():
            print(f"  {name}: missing dataset {data_path}, skipping")
            continue
        data = json.load(open(data_path))[:len(saved["activations"])]
        validate_dataset_provenance(saved, data, name)
        pair_diffs, pair_ids = get_pair_diffs(saved["activations"], data, label_map)
        n_pairs = len(pair_ids)

        D = pair_diffs[:, layer_idx]
        scores = D @ direction
        scores_aug = np.concatenate([scores, -scores])
        labels_aug = np.concatenate([np.ones(n_pairs), np.zeros(n_pairs)])
        auroc = roc_auc_score(labels_aug, scores_aug)

        meta = eval_result_metadata(name, auroc)
        tag = "val" if is_val_of_train_env else "heldout"
        role_msg = f", role={meta['eval_role']}"
        if meta["control_abs_delta"] is not None:
            role_msg += f", |AUROC-0.5|={meta['control_abs_delta']:.4f}"
        print(f"  {name} [{tag}]: {n_pairs} pairs, AUROC={auroc:.4f}{role_msg}")
        results[name] = {
            "auroc": float(auroc),
            "n_pairs": n_pairs,
            "layer": best_layer,
            "held_out": not is_val_of_train_env,
            **meta,
        }

    model_tag = probe.get("model_tag", cli_model_tag)
    out_name = tagged_filename("validation_results.json", model_tag)
    out_path = Path(output_dir) / out_name
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved to {out_path}")


if __name__ == "__main__":
    fire.Fire(validate)

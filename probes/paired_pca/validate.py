"""
Validate paired-PCA probes on designated validation/control sets.
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


def validate(data_dir="../..", activations_dir=None, probes_dir=".", model=DEFAULT_MODEL_TAG):
    results = {}
    cli_model_tag, _ = resolve_model(model) if model else ("", "")
    if activations_dir is None:
        activations_dir = Path(data_dir) / activation_dirname(cli_model_tag)
    activations_dir = Path(activations_dir)
    data_dir = Path(data_dir)
    probes_dir = Path(probes_dir)

    model_tag = cli_model_tag

    for probe_name, val_sets in VALIDATION_MAP.items():
        probe_path = probes_dir / tagged_filename(f"{probe_name}_probe.pt", cli_model_tag)
        if not probe_path.exists():
            probe_path = probes_dir / f"{probe_name}_probe.pt"
        if not probe_path.exists():
            print(f"Skipping {probe_name}: {probe_path} not found")
            continue

        probe = torch.load(probe_path, weights_only=False)
        if not model_tag:
            model_tag = probe.get("model_tag", "")
        direction = probe["direction"]
        best_layer = probe["best_layer"]

        print(f"\n{probe_name} probe (best layer {best_layer}):")

        for val_name, val_file, label_map in val_sets:
            act_path = activations_dir / f"{val_name}.pt"
            if not act_path.exists():
                print(f"  {val_name}: missing files, skipping")
                continue

            saved = torch.load(act_path, weights_only=False)
            data_path = resolve_dataset_path_for_activation(data_dir, val_file, saved.get("model_tag", model_tag), saved)
            if not data_path.exists():
                print(f"  {val_name}: missing dataset {data_path}, skipping")
                continue
            data = json.load(open(data_path))[:len(saved["activations"])]
            validate_dataset_provenance(saved, data, val_name)
            pair_diffs, pair_ids = get_pair_diffs(saved["activations"], data, label_map)
            n_pairs = len(pair_ids)

            D = pair_diffs[:, best_layer - 1]
            scores = D @ direction
            scores_aug = np.concatenate([scores, -scores])
            labels_aug = np.concatenate([np.ones(n_pairs), np.zeros(n_pairs)])
            auroc = roc_auc_score(labels_aug, scores_aug)

            meta = eval_result_metadata(val_name, auroc)
            role_msg = f", role={meta['eval_role']}"
            if meta["control_abs_delta"] is not None:
                role_msg += f", |AUROC-0.5|={meta['control_abs_delta']:.4f}"
            print(f"  -> {val_name}: {n_pairs} pairs, AUROC={auroc:.4f}{role_msg}")
            results[f"{probe_name}\u2192{val_name}"] = {
                "auroc": float(auroc),
                "n_pairs": n_pairs,
                "layer": best_layer,
                **meta,
            }

    out_fname = f"validation_results_{model_tag}.json" if model_tag else "validation_results.json"
    out_path = probes_dir / out_fname
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    fire.Fire(validate)

"""Shared helpers for probe evaluation scripts."""
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score

PROBES_ROOT = Path(__file__).resolve().parent.parent / "probes"

COND_TO_RES = {
    "instructed_system_prompt": "instructed_system_prompt_res",
    "spontaneous_1": "spontaneous_1_res",
    "sycophancy_answer": "sycophancy_answer_res",
    "sycophancy_feedback": "sycophancy_fb_res",
    "game_werewolf": "game_werewolf_res",
}


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def residual(direction, shared):
    d = normalize(direction)
    s = normalize(shared)
    return normalize(d - np.dot(d, s) * s)


def auroc_at_layer(activations, labels, direction):
    scores = activations @ direction
    if len(np.unique(labels)) < 2:
        return np.nan
    return roc_auc_score(labels, scores)


def find_peak_auroc(activations, labels, direction_stack):
    n_layers = direction_stack.shape[0]
    aurocs = np.array([auroc_at_layer(activations[:, l, :], labels, direction_stack[l])
                       for l in range(n_layers)])
    peak_layer = np.nanargmax(aurocs)
    return aurocs[peak_layer], int(peak_layer), aurocs


def load_directions(probe_paths, irm_path=None, directions="all", return_metadata=False):
    if isinstance(probe_paths, str):
        probe_paths = [p.strip() for p in probe_paths.split(",")]

    result = {}
    metadata = {}

    for p in probe_paths:
        pp = Path(p)
        if not pp.exists():
            continue
        label = pp.parent.name
        sd = torch.load(pp, weights_only=False)
        all_dirs = sd.get("all_directions", {})
        per_layer = sd.get("per_layer_directions", {})
        if not all_dirs:
            continue
        n_layers = len(all_dirs)

        shared_arr = np.stack([normalize(np.asarray(all_dirs[l])) for l in range(n_layers)])
        fixed_layer = sd.get("best_layer_transfer", sd.get("best_layer"))
        fixed_layer = fixed_layer - 1 if fixed_layer is not None else None

        if directions in ("all", "shared"):
            result[f"{label}/shared"] = shared_arr
            metadata[f"{label}/shared"] = {"fixed_layer": fixed_layer}

        if directions in ("all", "per_scenario"):
            for cond_name, cond_dirs in per_layer.items():
                name = f"{label}/{cond_name}"
                result[name] = np.stack([
                    normalize(np.asarray(cond_dirs[l])) for l in range(n_layers)])
                metadata[name] = {"fixed_layer": fixed_layer}

        if directions in ("all", "residuals"):
            for cond, res_name in COND_TO_RES.items():
                if cond in per_layer:
                    name = f"{label}/{res_name}"
                    result[name] = np.stack([
                        residual(np.asarray(per_layer[cond][l]), shared_arr[l])
                        for l in range(n_layers)])
                    metadata[name] = {"fixed_layer": fixed_layer}

    if irm_path:
        irm_pp = Path(irm_path)
        if irm_pp.exists():
            irm_data = torch.load(irm_pp, weights_only=False)
            irm_all = irm_data.get("all_directions", {})
            if irm_all:
                n_layers = len(irm_all)
                result["irm/shared"] = np.stack([
                    normalize(np.asarray(irm_all[l])) for l in range(n_layers)])
                fixed_layer = irm_data.get("best_layer")
                metadata["irm/shared"] = {
                    "fixed_layer": fixed_layer - 1 if fixed_layer is not None else None
                }

    for k, v in result.items():
        print(f"  {k}: {v.shape}")
    if return_metadata:
        return result, metadata
    return result


def default_probe_paths(model_tag=None):
    candidates = []
    for method in ["contrastive", "mass_mean", "mass_mean_iid", "paired_pca"]:
        if model_tag:
            p = PROBES_ROOT / method / f"shared_direction_{model_tag}.pt"
            if p.exists():
                candidates.append(str(p))
                continue
        p = PROBES_ROOT / method / "shared_direction.pt"
        if p.exists():
            candidates.append(str(p))
    return candidates


def default_irm_path(model_tag=None):
    if model_tag:
        p = PROBES_ROOT / "irm" / f"irm_probe_{model_tag}.pt"
        if p.exists():
            return str(p)
    p = PROBES_ROOT / "irm" / "irm_probe.pt"
    return str(p) if p.exists() else None

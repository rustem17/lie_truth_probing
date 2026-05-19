"""
Sweep IRM/V-REx hyperparameters across environment combinations.

Model: n/a (offline, cached activations)
Data: activations/*.pt + paired JSONs from src/probing/
Env combos: {game_werewolf,spontaneous_1,sycophancy_answer}, {spontaneous_1,sycophancy_answer}
Grid: penalty(irm,vrex) x lambda(1e1..1e4) x lr(1e-4,1e-3) x epochs(500,1000) x warmup(100,250) x wd(0,1e-4)
Metric: held-out mean AUROC (train datasets not in envs + all validation datasets)
Output: probes/irm/sweep_results_YYYYMMDD_HHMMSS.json
"""

import sys
import json
import itertools
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score
import torch
import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    DEFAULT_MODEL_TAG,
    TRAIN_DATASETS,
    VALIDATION_DATASETS,
    activation_dirname,
    control_abs_delta,
    eval_role,
    is_primary_eval,
    resolve_dataset_path_for_activation,
    resolve_model,
    validate_dataset_provenance,
)

from train import load_envs, train_all_layers, eval_auroc, get_pair_diffs


ENV_COMBOS = {
    "GSS": ["game_werewolf", "spontaneous_1", "sycophancy_answer"],
    "SS": ["spontaneous_1", "sycophancy_answer"],
}

GRID = {
    "penalty": ["irm", "vrex"],
    "lambda_irm": [1e1, 1e2, 1e3, 1e4],
    "lr": [1e-4, 1e-3],
    "n_epochs": [500, 1000],
    "warmup_steps": [100, 250],
    "weight_decay": [0.0, 1e-4],
}


def load_heldout(env_names, data_dir, activations_dir, model_tag=""):
    heldout_specs = {}
    for name, (filename, label_map) in TRAIN_DATASETS.items():
        if name not in env_names:
            heldout_specs[name] = (filename, label_map)
    for name, (filename, label_map) in VALIDATION_DATASETS.items():
        heldout_specs[name] = (filename, label_map)

    heldout = {}
    for name, (filename, label_map) in heldout_specs.items():
        act_path = Path(activations_dir) / f"{name}.pt"
        if not act_path.exists():
            continue
        saved = torch.load(act_path, weights_only=False)
        data_path = resolve_dataset_path_for_activation(data_dir, filename, saved.get("model_tag", model_tag), saved)
        if not data_path.exists():
            continue
        data = json.load(open(data_path))[:len(saved["activations"])]
        validate_dataset_provenance(saved, data, name)
        pair_diffs, _ = get_pair_diffs(saved["activations"], data, label_map)
        heldout[name] = pair_diffs
    return heldout


def eval_heldout_at_layer(direction, layer_idx, heldout):
    results = {}
    for name, pair_diffs in heldout.items():
        D = pair_diffs[:, layer_idx]
        n = D.shape[0]
        scores = np.concatenate([D @ direction, -D @ direction])
        labels = np.concatenate([np.ones(n), np.zeros(n)])
        results[name] = float(roc_auc_score(labels, scores))
    return results


def summarize_heldout_scores(scores):
    primary = [v for name, v in scores.items() if is_primary_eval(name)]
    controls = [control_abs_delta(v, name) for name, v in scores.items() if eval_role(name) == "control"]
    syco = [v for name, v in scores.items() if eval_role(name) == "sycophancy_variant"]
    return {
        "primary_mean": float(np.mean(primary)) if primary else 0.0,
        "primary_min": float(min(primary)) if primary else 0.0,
        "control_mean_abs_delta": float(np.mean(controls)) if controls else 0.0,
        "sycophancy_variant_mean": float(np.mean(syco)) if syco else 0.0,
    }


def sweep(data_dir="../..", activations_dir=None, output_dir=".",
          layer_min=10, layer_max=50, model=DEFAULT_MODEL_TAG):
    model_tag, _ = resolve_model(model) if model else ("", "")
    if activations_dir is None:
        activations_dir = Path(data_dir) / activation_dirname(model_tag)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    keys = list(GRID.keys())
    values = [GRID[k] for k in keys]
    all_combos = list(itertools.product(*values))
    total = len(all_combos) * len(ENV_COMBOS)
    print(f"{len(all_combos)} configs x {len(ENV_COMBOS)} env combos = {total} runs")

    all_results = []

    for combo_name, env_names in ENV_COMBOS.items():
        print(f"\n{'='*60}")
        print(f"env combo: {combo_name} = {env_names}")
        env_data, model_tag_loaded, _ = load_envs(env_names, Path(data_dir), Path(activations_dir), model_tag)
        heldout = load_heldout(env_names, Path(data_dir), Path(activations_dir), model_tag_loaded or model_tag)
        n_layers = next(iter(env_data.values()))["D"].shape[1]
        hidden_dim = next(iter(env_data.values()))["D"].shape[2]
        layer_lo = max(layer_min - 1, 0)
        layer_hi = min(layer_max, n_layers)
        print(f"heldout datasets: {list(heldout.keys())}, layer range: {layer_lo+1}-{layer_hi}")

        for ci, combo in enumerate(all_combos):
            params = dict(zip(keys, combo))
            ramp_steps = params["warmup_steps"]
            print(f"\n[{combo_name}] config {ci+1}/{len(all_combos)}: {params}")

            directions, _ = train_all_layers(
                env_data, env_names, n_layers, hidden_dim,
                n_epochs=int(params["n_epochs"]), lr=params["lr"],
                warmup_steps=int(params["warmup_steps"]), ramp_steps=ramp_steps,
                lambda_irm=params["lambda_irm"],
                penalty=params["penalty"], weight_decay=params["weight_decay"],
                verbose=False,
            )

            heldout_per_layer = {}
            for layer in range(layer_lo, layer_hi):
                h = eval_heldout_at_layer(directions[layer], layer, heldout)
                summary = summarize_heldout_scores(h)
                heldout_per_layer[layer] = {
                    "mean": summary["primary_mean"],
                    **summary,
                    "per_dataset": h,
                }

            best_layer_idx = max(heldout_per_layer, key=lambda l: heldout_per_layer[l]["mean"])
            best_direction = directions[best_layer_idx]

            env_diffs_best = [env_data[name]["D"][:, best_layer_idx] for name in env_names]
            train_auroc, per_env_best = eval_auroc(best_direction, env_diffs_best)
            train_per_env = {name: float(a) for name, a in zip(env_names, per_env_best)}

            heldout_aurocs = heldout_per_layer[best_layer_idx]["per_dataset"]
            heldout_mean = heldout_per_layer[best_layer_idx]["mean"]
            heldout_summary = summarize_heldout_scores(heldout_aurocs)

            result = {
                "env_combo": combo_name,
                "envs": env_names,
                **params,
                "ramp_steps": ramp_steps,
                "best_layer": best_layer_idx + 1,
                "train_auroc": train_auroc,
                "train_per_env": train_per_env,
                "heldout_aurocs": heldout_aurocs,
                "heldout_mean_auroc": heldout_mean,
                "heldout_primary_mean_auroc": heldout_summary["primary_mean"],
                "heldout_primary_min_auroc": heldout_summary["primary_min"],
                "heldout_control_mean_abs_delta": heldout_summary["control_mean_abs_delta"],
                "heldout_sycophancy_variant_mean_auroc": heldout_summary["sycophancy_variant_mean"],
                "heldout_per_layer": {str(l + 1): heldout_per_layer[l]["mean"] for l in heldout_per_layer},
            }
            all_results.append(result)
            print(f"  best_layer={best_layer_idx+1}  train={train_auroc:.4f}  heldout_mean={heldout_mean:.4f}")

    all_results.sort(key=lambda r: r["heldout_mean_auroc"], reverse=True)

    print(f"\n{'='*60}")
    print("top 10 by held-out mean AUROC:")
    for i, r in enumerate(all_results[:10]):
        print(f"  {i+1}. {r['env_combo']} {r['penalty']} lam={r['lambda_irm']:.0e} "
              f"lr={r['lr']:.0e} ep={r['n_epochs']} wu={r['warmup_steps']} wd={r['weight_decay']:.0e} "
              f"| train={r['train_auroc']:.4f} heldout={r['heldout_mean_auroc']:.4f} layer={r['best_layer']}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(output_dir) / f"sweep_results_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nsaved {len(all_results)} results to {out_path}")


if __name__ == "__main__":
    fire.Fire(sweep)

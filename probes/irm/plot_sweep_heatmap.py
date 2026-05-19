"""
Heatmap of heldout AUROC across sweep configs x sample sets, one plot per env combo.

Input: sweep_results_*.json
Output: probes/irm/plots/sweep_heatmap_{combo}_YYYYMMDD_HHMMSS.png
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import SHORT, control_abs_delta, eval_role  # noqa: E402


def make_heatmap(results, combo_name, dataset_names, output_dir, ts, role="primary_transfer"):
    dataset_names = [d for d in dataset_names if eval_role(d) == role]
    if not dataset_names:
        return
    if role == "control":
        matrix = np.array([
            [control_abs_delta(r["heldout_aurocs"][d], d) if d in r["heldout_aurocs"] else np.nan
             for d in dataset_names]
            for r in results
        ])
    else:
        matrix = np.array([[r["heldout_aurocs"].get(d, np.nan) for d in dataset_names] for r in results])

    labels = []
    for r in results:
        pen = r["penalty"][:3]
        lam = f"{r['lambda_irm']:.0e}"
        lr = f"{r['lr']:.0e}"
        ep = r["n_epochs"]
        wu = r["warmup_steps"]
        wd = f"{r['weight_decay']:.0e}"
        ly = r["best_layer"]
        labels.append(f"{pen} λ{lam} lr{lr} ep{ep} wu{wu} wd{wd} L{ly}")

    n = len(results)
    sns.set_theme(style="whitegrid")
    fig_height = max(6, n * 0.22)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    cmap, vmin, vmax, cbar_label = ("Reds", 0.0, 0.5, "|AUROC - 0.5|") if role == "control" else (
        "RdYlGn", 0.5, 1.0, "AUROC")
    xticklabels = [SHORT.get(d, d) for d in dataset_names]
    sns.heatmap(matrix, xticklabels=xticklabels, yticklabels=labels,
                cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0.2,
                annot=True, fmt=".2f", annot_kws={"size": 6}, ax=ax,
                cbar_kws={"label": cbar_label})
    ax.set_xlabel("Heldout dataset")
    ax.set_ylabel("Config (sorted by primary heldout AUROC)")
    ax.set_title(f"{combo_name}: {n} configs, heldout {role} ({cbar_label})")
    ax.tick_params(axis='y', labelsize=5.5)
    ax.tick_params(axis='x', labelsize=8, rotation=45)
    plt.tight_layout()

    out_path = Path(output_dir) / f"sweep_heatmap_{combo_name}_{role}_{ts}.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved to {out_path}")
    plt.close()


def plot(sweep_json=None, output_dir="plots"):
    if not sweep_json:
        candidates = sorted(Path(".").glob("sweep_results_*.json"))
        sweep_json = str(candidates[-1])
    all_results = json.load(open(sweep_json))

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    combos = {}
    for r in all_results:
        combos.setdefault(r["env_combo"], []).append(r)

    for combo_name, results in combos.items():
        results.sort(key=lambda r: r.get("heldout_primary_mean_auroc", r["heldout_mean_auroc"]), reverse=True)
        dataset_names = list(results[0]["heldout_aurocs"].keys())
        for role in ["primary_transfer", "control", "sycophancy_variant"]:
            make_heatmap(results, combo_name, dataset_names, output_dir, ts, role)


if __name__ == "__main__":
    fire.Fire(plot)

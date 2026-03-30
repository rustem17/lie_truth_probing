"""
Red team diagnostic: distribution shift between training and Apollo activations.

Quantifies activation distribution shift at the probe's best layer between
training conditions (pooled) and each Apollo dataset. Reports cosine similarity
of means, Mahalanobis distance of Apollo mean from training distribution, and
MMD (maximum mean discrepancy) estimate.

Prerequisites:
    - Trained probe .pt with best_layer info
    - Training activations in act_dir (e.g., activations_gemma3-27b/)
    - Apollo activations extracted via extract_activations.py on Apollo JSONL,
      stored as .pt files in apollo_act_dir

Usage:
    python distribution_shift.py \
        --model_tag gemma3-27b \
        --probe_path ../../probes/mass_mean/shared_direction_gemma3-27b.pt \
        --apollo_act_dir ../../activations_gemma3-27b_apollo
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import TRAIN_DATASETS


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def mahalanobis_distance(x, mean, cov_inv):
    diff = x - mean
    return float(np.sqrt(diff @ cov_inv @ diff))


def mmd_rbf(X, Y, gamma=None):
    if gamma is None:
        combined = np.concatenate([X, Y], axis=0)
        dists = np.sum((combined[:, None] - combined[None, :]) ** 2, axis=-1)
        gamma = 1.0 / np.median(dists[dists > 0])
    XX = np.exp(-gamma * np.sum((X[:, None] - X[None, :]) ** 2, axis=-1))
    YY = np.exp(-gamma * np.sum((Y[:, None] - Y[None, :]) ** 2, axis=-1))
    XY = np.exp(-gamma * np.sum((X[:, None] - Y[None, :]) ** 2, axis=-1))
    return float(XX.mean() + YY.mean() - 2 * XY.mean())


def run(model_tag,
        probe_path,
        apollo_act_dir,
        act_dir=None,
        data_dir="../..",
        max_train_samples=2000,
        output_dir="results"):
    if act_dir is None:
        act_dir = f"../../activations_{model_tag}"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    probe = torch.load(probe_path, weights_only=False)
    if "all_directions" in probe:
        best_layer = probe.get("best_layer", probe.get("best_layer_transfer", 1))
    else:
        best_layer = probe.get("best_layer", 1)
    best_idx = best_layer - 1

    train_acts = []
    for name, (filename, label_map) in TRAIN_DATASETS.items():
        act_path = Path(act_dir) / f"{name}.pt"
        if not act_path.exists():
            continue
        saved = torch.load(act_path, weights_only=False)
        train_acts.append(saved["activations"][:, best_idx])
        print(f"Loaded training: {name} ({saved['activations'].shape[0]} samples)")

    if not train_acts:
        print("No training activations found.")
        return

    train_X = np.concatenate(train_acts, axis=0)
    if train_X.shape[0] > max_train_samples:
        idx = np.random.RandomState(42).choice(train_X.shape[0], max_train_samples, replace=False)
        train_X = train_X[idx]

    train_mean = train_X.mean(axis=0)
    train_X_centered = train_X - train_mean
    U, s, Vt = np.linalg.svd(train_X_centered, full_matrices=False)
    k = min(200, len(s))
    ridge = 1e-4
    s_k = s[:k]
    Vt_k = Vt[:k]
    cov_inv_proj = Vt_k.T @ np.diag(1.0 / (s_k ** 2 / train_X.shape[0] + ridge)) @ Vt_k

    apollo_dir = Path(apollo_act_dir)
    apollo_files = sorted(apollo_dir.glob("*.pt"))
    if not apollo_files:
        print(f"No .pt files in {apollo_act_dir}")
        return

    rows = []
    apollo_data = {}

    for fpath in apollo_files:
        dname = fpath.stem
        saved = torch.load(fpath, weights_only=False)
        acts = saved["activations"]
        if acts.ndim == 3:
            acts = acts[:, best_idx]
        elif acts.ndim == 2:
            pass
        else:
            print(f"Skipping {dname}: unexpected shape {acts.shape}")
            continue

        apollo_mean = acts.mean(axis=0)
        cos = cosine_sim(train_mean, apollo_mean)
        mahal = mahalanobis_distance(apollo_mean, train_mean, cov_inv_proj)

        n_sub = min(500, acts.shape[0], train_X.shape[0])
        train_sub = train_X[np.random.RandomState(42).choice(train_X.shape[0], n_sub, replace=False)]
        apollo_sub = acts[np.random.RandomState(43).choice(acts.shape[0], min(n_sub, acts.shape[0]), replace=False)]
        mmd = mmd_rbf(train_sub, apollo_sub)

        mean_norm_ratio = np.linalg.norm(apollo_mean) / (np.linalg.norm(train_mean) + 1e-10)
        std_ratio = acts.std() / (train_X.std() + 1e-10)

        print(f"{dname}: cos={cos:.4f}, mahal={mahal:.2f}, mmd={mmd:.6f}, norm_ratio={mean_norm_ratio:.3f}, std_ratio={std_ratio:.3f}, n={acts.shape[0]}")
        rows.append({
            "apollo_dataset": dname,
            "cosine_sim_means": cos,
            "mahal_distance": mahal,
            "mmd_estimate": mmd,
            "mean_norm_ratio": mean_norm_ratio,
            "std_ratio": std_ratio,
            "n_apollo": acts.shape[0],
            "n_train": train_X.shape[0],
        })
        apollo_data[dname] = acts

    if not rows:
        print("No Apollo datasets processed.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(output_dir) / f"distribution_shift_{model_tag}_{timestamp}.csv"
    with open(csv_path, "w") as f:
        f.write("apollo_dataset,cosine_sim_means,mahal_distance,mmd_estimate,mean_norm_ratio,std_ratio,n_apollo,n_train\n")
        for r in rows:
            f.write(f"{r['apollo_dataset']},{r['cosine_sim_means']:.4f},{r['mahal_distance']:.2f},{r['mmd_estimate']:.6f},{r['mean_norm_ratio']:.3f},{r['std_ratio']:.3f},{r['n_apollo']},{r['n_train']}\n")
    print(f"\nCSV: {csv_path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ds_names = [r["apollo_dataset"] for r in rows]
    x = np.arange(len(ds_names))

    metrics = [
        ("cosine_sim_means", "Cosine similarity of means", "#729ECE"),
        ("mahal_distance", "Mahalanobis distance", "#D68F73"),
        ("mmd_estimate", "MMD estimate", "#97C484"),
    ]

    for ax, (key, title, color) in zip(axes, metrics):
        vals = [r[key] for r in rows]
        ax.bar(x, vals, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(ds_names, rotation=30, ha="right", fontsize=8)
        ax.set_title(title)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle(f"Distribution shift: training vs Apollo — L{best_layer}", fontsize=12)
    fig.tight_layout()
    plot_path = Path(output_dir) / f"distribution_shift_{model_tag}_{timestamp}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    fire.Fire(run)

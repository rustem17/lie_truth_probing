import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def resolve_device(device="auto"):
    if device is None:
        device = "auto"
    device = str(device).lower()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def use_torch_device(device):
    return torch.device(device).type != "cpu"


def to_float_tensor(array, device):
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def normalize_tensor(v, eps=1e-12):
    norm = torch.linalg.vector_norm(v)
    if not torch.isfinite(norm) or norm <= eps:
        return torch.zeros_like(v)
    return v / norm


def tensor_to_numpy(v):
    return v.detach().cpu().numpy()


def finite_rows_tensor(D):
    return torch.isfinite(D).all(dim=1)


def augmented_auroc_from_scores(scores):
    if isinstance(scores, torch.Tensor):
        scores = tensor_to_numpy(scores)
    scores = np.asarray(scores)
    mask = np.isfinite(scores)
    if not np.any(mask):
        return 0.5
    scores = scores[mask]
    scores_all = np.concatenate([scores, -scores])
    labels_all = np.concatenate([np.ones(len(scores)), np.zeros(len(scores))])
    if np.all(scores_all == scores_all[0]):
        return 0.5
    return float(roc_auc_score(labels_all, scores_all))

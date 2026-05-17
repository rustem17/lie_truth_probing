import numpy as np
import torch

from probes.contrastive.train import fit_torch_logistic_direction
from probes.torch_accel import augmented_auroc_from_scores, normalize_tensor, resolve_device


def test_torch_accel_helpers_on_cpu():
    assert resolve_device("cpu").type == "cpu"
    assert torch.allclose(normalize_tensor(torch.zeros(3)), torch.zeros(3))
    assert augmented_auroc_from_scores(torch.tensor([0.0, 0.0])) == 0.5


def test_torch_logistic_direction_separates_simple_pair_diffs():
    D = torch.tensor([
        [1.0, 0.0],
        [2.0, 0.0],
        [1.5, 0.1],
    ])

    direction = fit_torch_logistic_direction(D, max_iter=25, logistic_l2=1e-4)
    scores = D @ direction

    assert np.isfinite(direction.detach().numpy()).all()
    assert torch.linalg.vector_norm(direction) > 0.99
    assert torch.all(scores > 0)

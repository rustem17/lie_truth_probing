import numpy as np
import torch

from probes.contrastive import shared_direction as contrastive_shared
from probes.mahalanobis_lda import shared_direction as mahalanobis_shared
from probes.mass_mean_iid import shared_direction as mass_mean_iid_shared
from probes.paired_pca import shared_direction as paired_pca_shared


def test_contrastive_shared_direction_handles_degenerate_layers():
    D = np.zeros((4, 2, 3), dtype=np.float32)
    D[0, 1, 0] = np.nan
    diffs = {"toy": {"D": D, "n_pairs": 4}}

    directions = contrastive_shared.train_all_directions(diffs, ["toy"])["toy"]

    assert np.isfinite(directions).all()
    assert np.allclose(directions[0], 0.0)
    assert contrastive_shared.transfer_auroc(directions[0], D[:, 0]) == 0.5
    assert contrastive_shared.constrained_argmax([np.nan, np.nan], 0, 2) == 0


def test_contrastive_layer_range_accepts_fire_tuple_and_string_forms():
    assert contrastive_shared.parse_layer_range((20, 40)) == (19, 40)
    assert contrastive_shared.parse_layer_range("20,40") == (19, 40)
    assert contrastive_shared.parse_layer_range("(20, 40)") == (19, 40)


def test_mass_mean_iid_shared_direction_handles_degenerate_inputs():
    D = np.zeros((4, 3), dtype=np.float32)
    D[0, 0] = np.nan

    feature, decision = mass_mean_iid_shared.cov_corrected_direction(D)

    assert np.isfinite(feature).all()
    assert np.isfinite(decision).all()
    assert mass_mean_iid_shared.transfer_auroc(decision, D) == 0.5


def test_paired_pca_shared_direction_handles_degenerate_inputs():
    D = np.zeros((4, 3), dtype=np.float32)
    D[0, 0] = np.nan

    direction, ratio = paired_pca_shared.paired_pca_direction(D)

    assert np.isfinite(direction).all()
    assert ratio == 0.0
    assert paired_pca_shared.transfer_auroc(direction, D) == 0.5


def test_mahalanobis_shared_direction_handles_degenerate_inputs():
    D = np.zeros((4, 3), dtype=np.float32)
    D[0, 0] = np.nan

    direction = mahalanobis_shared.fisher_lda(D)
    multi = mahalanobis_shared.multi_env_lda([D])

    assert np.isfinite(direction).all()
    assert np.isfinite(multi).all()
    assert mahalanobis_shared.transfer_auroc(direction, D) == 0.5


def test_mahalanobis_fast_average_shared_mode_reuses_dataset_directions():
    diffs = {
        "a": {"D": np.array([[[1.0, 0.0]], [[2.0, 0.0]]]), "n_pairs": 2},
        "b": {"D": np.array([[[0.0, 1.0]], [[0.0, 2.0]]]), "n_pairs": 2},
    }
    directions = {
        "a": np.array([[1.0, 0.0]]),
        "b": np.array([[0.0, 1.0]]),
    }

    shared = mahalanobis_shared.build_shared_direction(
        diffs, directions, ["a", "b"], layer=0, ridge=1e-4, pca_var=0.95, shared_mode="average")

    assert np.allclose(shared, np.array([2 ** -0.5, 2 ** -0.5]))


def test_mahalanobis_torch_multi_env_matches_numpy_estimator():
    rng = np.random.default_rng(0)
    D1 = rng.normal(size=(8, 5)).astype(np.float32)
    D2 = rng.normal(size=(9, 5)).astype(np.float32)
    D1[:, 0] += 0.5
    D2[:, 1] += 0.5

    numpy_dir = mahalanobis_shared.multi_env_lda_numpy([D1, D2], ridge=1e-3, pca_var=0.9)
    torch_dir = mahalanobis_shared.multi_env_lda_torch([D1, D2], ridge=1e-3, pca_var=0.9, device=torch.device("cpu"))

    cosine = float(np.dot(numpy_dir, torch_dir) / (np.linalg.norm(numpy_dir) * np.linalg.norm(torch_dir)))
    assert abs(cosine) > 0.999

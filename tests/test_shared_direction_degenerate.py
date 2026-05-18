import numpy as np

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

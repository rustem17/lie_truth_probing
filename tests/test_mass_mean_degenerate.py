import json

import numpy as np
import torch

from probes.mass_mean.train import augmented_auroc_from_scores, mass_mean_direction, train


def test_mass_mean_direction_handles_zero_and_nonfinite_rows():
    assert np.allclose(mass_mean_direction(np.zeros((3, 4))), np.zeros(4))

    D = np.array([
        [3.0, 4.0, 0.0],
        [np.nan, 1.0, 2.0],
    ])
    assert np.allclose(mass_mean_direction(D), np.array([0.6, 0.8, 0.0]))


def test_augmented_auroc_handles_degenerate_scores():
    assert augmented_auroc_from_scores(np.array([0.0, 0.0])) == 0.5
    assert augmented_auroc_from_scores(np.array([np.nan, np.inf])) == 0.5


def test_mass_mean_train_writes_probe_for_degenerate_activations(tmp_path):
    data = [
        {"id": "case_001_lie", "condition": "lie"},
        {"id": "case_001_truth", "condition": "truth"},
        {"id": "case_002_lie", "condition": "lie"},
        {"id": "case_002_truth", "condition": "truth"},
    ]
    (tmp_path / "instructed_system_prompt.json").write_text(json.dumps(data))

    activations_dir = tmp_path / "activations"
    activations_dir.mkdir()
    torch.save(
        {
            "activations": np.zeros((4, 2, 3), dtype=np.float32),
            "model_tag": "",
            "model_id": "dummy",
        },
        activations_dir / "instructed_system_prompt.pt",
    )

    output_dir = tmp_path / "out"
    train(
        data_dir=str(tmp_path),
        activations_dir=str(activations_dir),
        output_dir=str(output_dir),
        n_splits=2,
        model="",
    )

    probe = torch.load(output_dir / "instructed_system_prompt_probe.pt", weights_only=False)
    assert np.allclose(probe["direction"], np.zeros(3))
    assert all(row["auroc"] == 0.5 for row in probe["layer_results"])

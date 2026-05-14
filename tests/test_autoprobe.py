import numpy as np

from autoprobe.features import get_pair_diffs
from autoprobe.evaluator import robust_score
from autoprobe.probes import score_auroc
from autoprobe.specs import ExperimentSpec


def test_pair_diffs_groups_suffixed_ids():
    features = np.array([
        [[3.0, 1.0]],
        [[1.0, 1.0]],
        [[2.0, 5.0]],
        [[2.0, 1.0]],
    ])
    data = [
        {"id": "a_lie", "condition": "lie", "model_response": "xxx", "system_prompt": "", "user_message": ""},
        {"id": "a_truth", "condition": "truth", "model_response": "x", "system_prompt": "", "user_message": ""},
        {"id": "b_lie", "condition": "lie", "model_response": "yy", "system_prompt": "", "user_message": ""},
        {"id": "b_truth", "condition": "truth", "model_response": "y", "system_prompt": "", "user_message": ""},
    ]
    D, pair_ids, meta = get_pair_diffs(features, data, {"lie": 1, "truth": 0})
    assert pair_ids == ["a", "b"]
    assert D.shape == (2, 1, 2)
    assert np.allclose(D[:, 0], [[2.0, 0.0], [0.0, 4.0]])
    assert meta[0]["response_length_delta"] == 2


def test_robust_score_penalizes_worst_case_and_confounds():
    clean = robust_score(
        val_mean_auroc=0.85,
        val_min_auroc=0.80,
        length_penalty=0.0,
        dataset_penalty=0.0,
        runtime_seconds=10.0,
        baseline_runtime_seconds=60.0,
    )
    weak = robust_score(
        val_mean_auroc=0.85,
        val_min_auroc=0.60,
        length_penalty=0.4,
        dataset_penalty=0.5,
        runtime_seconds=10.0,
        baseline_runtime_seconds=60.0,
    )
    assert clean > weak


def test_spec_validation_rejects_bad_feature_type():
    spec = ExperimentSpec(name="bad", feature_type="raw_attention")
    try:
        spec.validate()
    except ValueError as exc:
        assert "unsupported feature_type" in str(exc)
    else:
        raise AssertionError("expected validation error")


def test_score_auroc_uses_augmented_pair_scores():
    class Pair:
        D = np.array([[[1.0, 0.0]], [[2.0, 0.0]], [[3.0, 0.0]]])

    selected = [(0, np.array([1.0, 0.0]), 1.0)]
    assert score_auroc(selected, Pair()) == 1.0

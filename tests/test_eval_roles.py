from config import control_abs_delta, eval_result_metadata, eval_role, is_primary_eval


def test_controls_are_not_primary_eval_metrics():
    assert eval_role("spontaneous_control") == "control"
    assert eval_role("spontaneous_inconsistent") == "control"
    assert not is_primary_eval("spontaneous_control")
    assert control_abs_delta(0.72, "spontaneous_control") == 0.21999999999999997


def test_unknown_or_ordinary_eval_defaults_to_primary_transfer():
    assert eval_role("spontaneous_2") == "primary_transfer"
    assert eval_role("new_regular_holdout") == "primary_transfer"
    assert is_primary_eval("new_regular_holdout")
    assert control_abs_delta(0.72, "new_regular_holdout") is None


def test_eval_result_metadata_marks_sycophancy_variants_separately():
    meta = eval_result_metadata("sycophancy_feedback", 0.61)

    assert meta == {
        "eval_role": "sycophancy_variant",
        "primary_metric": False,
        "control_abs_delta": None,
    }

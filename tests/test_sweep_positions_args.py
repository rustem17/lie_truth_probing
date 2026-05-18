from eval.sweep_positions import format_cli_list, format_layer_range, shared_result_path, train_result_path


def test_sweep_formats_tuple_cli_values_for_subprocesses():
    assert format_cli_list(("instructed_system_prompt", "spontaneous_1", "sycophancy_answer")) == (
        "instructed_system_prompt,spontaneous_1,sycophancy_answer"
    )
    assert format_layer_range((20, 40)) == "20,40"
    assert format_layer_range("20,40") == "20,40"


def test_sweep_resume_artifact_paths_are_tagged(tmp_path):
    tag = "llama-3-3-70b-instruct"

    assert train_result_path(tmp_path, "contrastive", tag).name == f"results_{tag}.json"
    assert train_result_path(tmp_path, "irm", tag).name == f"irm_probe_{tag}.pt"
    assert shared_result_path(tmp_path, tag).name == f"shared_direction_{tag}.pt"

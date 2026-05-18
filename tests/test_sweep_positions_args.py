from eval.sweep_positions import format_cli_list, format_layer_range


def test_sweep_formats_tuple_cli_values_for_subprocesses():
    assert format_cli_list(("instructed_system_prompt", "spontaneous_1", "sycophancy_answer")) == (
        "instructed_system_prompt,spontaneous_1,sycophancy_answer"
    )
    assert format_layer_range((20, 40)) == "20,40"
    assert format_layer_range("20,40") == "20,40"

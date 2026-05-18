from eval.sweep_positions import format_cli_list, format_layer_range, phase_shared, shared_result_path, train_result_path


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


def test_sweep_defaults_to_original_mahalanobis_shared_mode(tmp_path, monkeypatch):
    calls = []

    def fake_run_command(args, cwd, dry_run=False):
        calls.append(args)

    monkeypatch.setattr("eval.sweep_positions.run_command", fake_run_command)
    act_dir = tmp_path / "activations_llama-3-3-70b-instruct"
    act_dir.mkdir()

    phase_shared(
        data_dir=tmp_path,
        sweep_dir=tmp_path / "sweep",
        positions=["first"],
        methods=["mahalanobis_lda"],
        model="llama-3-3-70b-instruct",
        model_tag="llama-3-3-70b-instruct",
    )

    assert "--shared_mode" in calls[0]
    assert calls[0][calls[0].index("--shared_mode") + 1] == "multi_env"


def test_sweep_can_pass_fast_mahalanobis_shared_mode(tmp_path, monkeypatch):
    calls = []

    def fake_run_command(args, cwd, dry_run=False):
        calls.append(args)

    monkeypatch.setattr("eval.sweep_positions.run_command", fake_run_command)
    act_dir = tmp_path / "activations_llama-3-3-70b-instruct"
    act_dir.mkdir()

    phase_shared(
        data_dir=tmp_path,
        sweep_dir=tmp_path / "sweep",
        positions=["first"],
        methods=["mahalanobis_lda"],
        model="llama-3-3-70b-instruct",
        model_tag="llama-3-3-70b-instruct",
        mahalanobis_shared_mode="average",
    )

    assert "--shared_mode" in calls[0]
    assert calls[0][calls[0].index("--shared_mode") + 1] == "average"

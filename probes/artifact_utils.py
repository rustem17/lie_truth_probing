"""Helpers for resolving model-tagged probe artifacts."""

from pathlib import Path

import torch

from config import tagged_filename


def resolve_artifact_path(directory, base_name, model_tag="", allow_untagged_fallback=False):
    directory = Path(directory)
    if model_tag:
        tagged = directory / tagged_filename(base_name, model_tag)
        if tagged.exists():
            return tagged
        fallback = directory / base_name
        if allow_untagged_fallback and fallback.exists():
            return fallback
        return tagged
    return directory / base_name


def validate_model_tag(artifact, path, expected_model_tag):
    if not expected_model_tag or not isinstance(artifact, dict):
        return
    actual = artifact.get("model_tag")
    if actual and actual != expected_model_tag:
        raise ValueError(
            f"{path}: artifact model_tag={actual!r}, expected {expected_model_tag!r}"
        )


def load_torch_artifact(path, expected_model_tag=""):
    artifact = torch.load(path, weights_only=False)
    validate_model_tag(artifact, path, expected_model_tag)
    return artifact


def load_shared_direction(probes_dir, model_tag="", allow_untagged_fallback=False):
    path = resolve_artifact_path(
        probes_dir,
        "shared_direction.pt",
        model_tag,
        allow_untagged_fallback=allow_untagged_fallback,
    )
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run shared_direction.py for this method/model, "
            "or pass --allow_untagged_fallback=True to use legacy untagged artifacts."
        )
    return load_torch_artifact(path, expected_model_tag=model_tag), path

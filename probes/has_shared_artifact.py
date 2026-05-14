"""Exit 0 when a method directory has a shared-direction artifact."""

import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DEFAULT_MODEL_TAG, resolve_model  # noqa: E402
from artifact_utils import resolve_artifact_path  # noqa: E402


def main(probes_dir=".", model=DEFAULT_MODEL_TAG, allow_untagged_fallback=False):
    model_tag, _ = resolve_model(model) if model else ("", "")
    path = resolve_artifact_path(
        probes_dir,
        "shared_direction.pt",
        model_tag,
        allow_untagged_fallback=allow_untagged_fallback,
    )
    if path.exists():
        print(path)
        return
    raise SystemExit(1)


if __name__ == "__main__":
    fire.Fire(main)

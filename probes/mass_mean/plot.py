"""Plot mass-mean probe results."""

import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_probe_results import main as plot_main  # noqa: E402


def main(probes_dir=".", model=None, allow_untagged_fallback=False):
    kwargs = {"probes_dir": probes_dir, "method_label": "Mass-Mean"}
    if model is not None:
        kwargs["model"] = model
    kwargs["allow_untagged_fallback"] = allow_untagged_fallback
    plot_main(**kwargs)


if __name__ == "__main__":
    fire.Fire(main)

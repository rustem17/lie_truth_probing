"""Build filtered game_lie/game_truth pairs from judged multi-run results.

Input: multi_results_{model_tag}.json (from infer.py — game scenarios, model_responses + judge_labels)
Output: ../../game_werewolf_{model_tag}.json by default
Filters: both lie and truth must have a compliant run, char length diff <= max_diff.
Picks first compliant response per sample.
"""
import json
import re
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import DEFAULT_MODEL_TAG, infer_output_filename, resolve_model, tagged_filename

OUTPUT = Path(__file__).parent.parent.parent / "game_werewolf.json"
INFER_KEYS = {"model_responses", "judge_labels", "n_runs"}


def first_compliant(sample):
    for i, jl in enumerate(sample["judge_labels"]):
        if jl and jl["followed_instruction"]:
            return sample["model_responses"][i]
    return None


def main(input=None, output=None, max_diff=200, model=DEFAULT_MODEL_TAG, model_tag=""):
    model_tag = model_tag or resolve_model(model)[0]
    if input is None:
        input = infer_output_filename("multi_results.json", model_tag)
        if not Path(input).exists() and Path("multi_results.json").exists():
            print(f"{input} not found; falling back to multi_results.json")
            input = "multi_results.json"
    if output is None:
        output = str(OUTPUT.parent / tagged_filename(OUTPUT.name, model_tag))
    with open(input) as f:
        data = json.load(f)

    by_num = {}
    for s in data:
        num = re.search(r"(\d+)", s["id"]).group(1)
        by_num.setdefault(num, {})[s["condition"]] = s

    kept, dropped_compliance, dropped_length = [], 0, 0
    for num in sorted(by_num, key=int):
        pair = by_num[num]
        if "game_lie" not in pair or "game_truth" not in pair:
            dropped_compliance += 1
            continue
        lie_resp = first_compliant(pair["game_lie"])
        truth_resp = first_compliant(pair["game_truth"])
        if lie_resp is None or truth_resp is None:
            dropped_compliance += 1
            continue
        if abs(len(lie_resp) - len(truth_resp)) > max_diff:
            dropped_length += 1
            continue

        for cond in ("game_lie", "game_truth"):
            s = pair[cond]
            resp = lie_resp if cond == "game_lie" else truth_resp
            entry = {k: v for k, v in s.items() if k not in INFER_KEYS}
            entry["model_response"] = resp
            kept.append(entry)

    with open(output, "w") as f:
        json.dump(kept, f, indent=2)
    n = len(kept) // 2
    print(f"{n} pairs kept, {dropped_compliance} dropped (compliance), "
          f"{dropped_length} dropped (length>{max_diff}) -> {output}")


if __name__ == "__main__":
    fire.Fire(main)

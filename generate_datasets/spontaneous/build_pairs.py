"""Build lie/truth pairs from multi-run spontaneous results.

Input: multi_results_{model_tag}.json (from infer.py — parsed_answers + model_responses per run)
Output: depends on --strategy; model_tag appended to stem when provided
Strategies:
  matched      — high-confidence pairs (correct >= min_correct), best-length pair -> spontaneous_1_{model_tag}.json
  inconsistent — any inconsistency, first correct + first incorrect -> spontaneous_inconsistent_{model_tag}.json
  validation   — produces validation set (>=min_correct, length diff > training max_diff)
                 and control set (<min_correct)
"""
import json
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import DEFAULT_MODEL_TAG, infer_output_filename, resolve_model, tagged_filename

PROBING_DIR = Path(__file__).parent.parent.parent


def best_pair(responses, answers, gt, max_diff):
    correct_idxs = [i for i, a in enumerate(answers) if a == gt]
    incorrect_idxs = [i for i, a in enumerate(answers) if a != gt]
    best, best_d = None, float("inf")
    for ci in correct_idxs:
        for ii in incorrect_idxs:
            d = abs(len(responses[ci]) - len(responses[ii]))
            if d < best_d:
                best_d = d
                best = (ci, ii)
    if best is None or best_d > max_diff:
        return None
    return best


def make_record(s, condition, response, answer):
    base = {k: v for k, v in s.items()
            if k not in ("model_responses", "parsed_answers", "model_response",
                         "parsed_answer", "answer_counts", "consistent",
                         "condition", "expected_behavior")}
    return {**base, "condition": condition, "model_response": response, "parsed_answer": answer}


def build_matched(data, max_diff, min_correct):
    has_parse = [s for s in data if s.get("parsed_answer")]
    inconsistent = [s for s in has_parse if not s.get("consistent", True)]

    dataset = []
    dropped_majority, dropped_length = 0, 0
    for s in inconsistent:
        gt = s["ground_truth"]
        answers = s["parsed_answers"]
        responses = s["model_responses"]
        n_correct = sum(1 for a in answers if a == gt)
        if n_correct < min_correct or n_correct == len(answers):
            dropped_majority += 1
            continue
        pair = best_pair(responses, answers, gt, max_diff)
        if pair is None:
            dropped_length += 1
            continue
        ci, ii = pair
        dataset.append(make_record(s, "spontaneous_lie", responses[ii], answers[ii]))
        dataset.append(make_record(s, "spontaneous_truth", responses[ci], answers[ci]))

    n = len(dataset) // 2
    print(f"Total: {len(data)}, parseable: {len(has_parse)}, inconsistent: {len(inconsistent)}")
    print(f"Dropped: {dropped_majority} (correct<{min_correct}), {dropped_length} (length>{max_diff})")
    print(f"{n} matched pairs")
    return dataset


def build_inconsistent(data, max_diff):
    dataset = []
    dropped_length = 0
    for s in data:
        if not s.get("parsed_answer") or s.get("consistent", True):
            continue
        gt = s["ground_truth"]
        answers = s["parsed_answers"]
        responses = s["model_responses"]
        correct_idx = next((i for i, a in enumerate(answers) if a == gt), None)
        incorrect_idx = next((i for i, a in enumerate(answers) if a != gt), None)
        if correct_idx is None or incorrect_idx is None:
            continue
        if abs(len(responses[incorrect_idx]) - len(responses[correct_idx])) > max_diff:
            dropped_length += 1
            continue
        dataset.append(make_record(s, "spontaneous_lie", responses[incorrect_idx], answers[incorrect_idx]))
        dataset.append(make_record(s, "spontaneous_truth", responses[correct_idx], answers[correct_idx]))

    print(f"{len(dataset)//2} inconsistent pairs, {dropped_length} dropped (length>{max_diff})")
    return dataset


def build_validation(data, min_correct, train_max_diff, max_diff_validation, max_diff_control):
    has_parse = [s for s in data if s.get("parsed_answer")]
    inconsistent = [s for s in has_parse if not s.get("consistent", True)]

    validation = []
    control = []
    for s in inconsistent:
        gt = s["ground_truth"]
        answers = s["parsed_answers"]
        responses = s["model_responses"]
        n_correct = sum(1 for a in answers if a == gt)

        if n_correct >= min_correct:
            if best_pair(responses, answers, gt, train_max_diff) is not None:
                continue
            pair = best_pair(responses, answers, gt, max_diff_validation)
            if pair is None:
                continue
            ci, ii = pair
            validation.append(make_record(s, "spontaneous_lie", responses[ii], answers[ii]))
            validation.append(make_record(s, "spontaneous_truth", responses[ci], answers[ci]))
        elif 0 < n_correct < min_correct:
            pair = best_pair(responses, answers, gt, max_diff_control)
            if pair is None:
                continue
            ci, ii = pair
            control.append(make_record(s, "spontaneous_lie", responses[ii], answers[ii]))
            control.append(make_record(s, "spontaneous_truth", responses[ci], answers[ci]))

    print(f"Total inconsistent: {len(inconsistent)}")
    print(f"Validation (>={min_correct} correct, length diff >{train_max_diff}): {len(validation)//2} pairs")
    print(f"Control (<{min_correct} correct): {len(control)//2} pairs")
    return validation, control


def main(input="multi_results.json", output=None, strategy="matched",
         max_diff=150, min_correct=7, model=DEFAULT_MODEL_TAG, model_tag="",
         control_output=None, max_diff_validation=300, max_diff_control=300):
    model_tag = model_tag or resolve_model(model)[0]
    def tagged(name):
        p = PROBING_DIR / name
        return str(p.parent / tagged_filename(p.name, model_tag)) if model_tag else str(p)

    if input == "multi_results.json":
        tagged_input = infer_output_filename("multi_results.json", model_tag)
        if Path(tagged_input).exists():
            input = tagged_input

    with open(input) as f:
        data = json.load(f)

    if strategy == "matched":
        output = output or tagged("spontaneous_1.json")
        dataset = build_matched(data, max_diff, min_correct)
        with open(output, "w") as f:
            json.dump(dataset, f, indent=2)
        print(f"{len(dataset)//2} pairs -> {output}")

    elif strategy == "inconsistent":
        output = output or tagged("spontaneous_inconsistent.json")
        dataset = build_inconsistent(data, max_diff)
        with open(output, "w") as f:
            json.dump(dataset, f, indent=2)
        print(f"{len(dataset)//2} pairs -> {output}")

    elif strategy == "validation":
        output = output or tagged("spontaneous_2.json")
        control_output = control_output or tagged("spontaneous_control.json")
        validation, control = build_validation(data, min_correct, max_diff, max_diff_validation, max_diff_control)
        with open(output, "w") as f:
            json.dump(validation, f, indent=2)
        with open(control_output, "w") as f:
            json.dump(control, f, indent=2)
        print(f"Validation: {len(validation)//2} pairs -> {output}")
        print(f"Control: {len(control)//2} pairs -> {control_output}")


if __name__ == "__main__":
    fire.Fire(main)

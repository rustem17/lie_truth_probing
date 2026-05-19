"""Audit paired lie/truth datasets for basic quality artifacts.

This is a lightweight report, not a filter. It checks final paired JSON files for
exact duplicate lie/truth responses, label balance, missing responses, and rough
response-length imbalance.
"""

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ALL_DATASETS, DEFAULT_MODEL_TAG, resolve_dataset_path, resolve_model


PAIR_SUFFIX = re.compile(r"_(lie|truth)$")


def base_pair_id(sample_id):
    return PAIR_SUFFIX.sub("", str(sample_id))


def normalize_text(text):
    return " ".join(str(text or "").split()).strip()


def response_len(text):
    return len(normalize_text(text).split())


def summarize_dataset(path, name, label_map):
    data = json.load(open(path))
    groups = defaultdict(list)
    labels = Counter()
    missing_response = 0

    for sample in data:
        condition = sample.get("condition")
        labels[condition] += 1
        if not normalize_text(sample.get("model_response", "")):
            missing_response += 1
        groups[base_pair_id(sample.get("id"))].append(sample)

    paired = 0
    exact_duplicate_pairs = 0
    length_diffs = []
    unpaired_groups = 0

    for samples in groups.values():
        positives = [s for s in samples if label_map.get(s.get("condition")) == 1]
        negatives = [s for s in samples if label_map.get(s.get("condition")) == 0]
        if not positives or not negatives:
            unpaired_groups += 1
            continue
        paired += min(len(positives), len(negatives))
        pos_responses = {normalize_text(s.get("model_response", "")) for s in positives}
        neg_responses = {normalize_text(s.get("model_response", "")) for s in negatives}
        if pos_responses & neg_responses:
            exact_duplicate_pairs += 1
        for pos, neg in zip(positives, negatives):
            length_diffs.append(
                abs(response_len(pos.get("model_response", "")) - response_len(neg.get("model_response", "")))
            )

    return {
        "dataset": name,
        "file": str(path),
        "samples": len(data),
        "groups": len(groups),
        "paired_groups": paired,
        "unpaired_groups": unpaired_groups,
        "exact_duplicate_pairs": exact_duplicate_pairs,
        "duplicate_pair_rate": exact_duplicate_pairs / paired if paired else 0.0,
        "missing_response": missing_response,
        "mean_abs_length_diff": sum(length_diffs) / len(length_diffs) if length_diffs else 0.0,
        "max_abs_length_diff": max(length_diffs) if length_diffs else 0,
        "label_counts": dict(labels),
    }


def write_csv(path, rows):
    fieldnames = [
        "dataset",
        "file",
        "samples",
        "groups",
        "paired_groups",
        "unpaired_groups",
        "exact_duplicate_pairs",
        "duplicate_pair_rate",
        "missing_response",
        "mean_abs_length_diff",
        "max_abs_length_diff",
        "label_counts",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def audit(data_dir=".", model=DEFAULT_MODEL_TAG, output=None, datasets=None):
    model_tag, _ = resolve_model(model)
    data_dir = Path(data_dir)
    keep = None
    if datasets:
        keep = set(datasets) if isinstance(datasets, (list, tuple)) else {d.strip() for d in str(datasets).split(",") if d.strip()}

    rows = []
    for name, (filename, label_map) in ALL_DATASETS.items():
        if keep and name not in keep:
            continue
        path = resolve_dataset_path(data_dir, filename, model_tag)
        if not path.exists():
            print(f"Skipping {name}: missing {path}")
            continue
        rows.append(summarize_dataset(path, name, label_map))

    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix == ".json":
            with open(output, "w") as f:
                json.dump(rows, f, indent=2)
        else:
            write_csv(output, rows)

    for row in rows:
        print(
            f"{row['dataset']}: pairs={row['paired_groups']}, "
            f"exact_duplicate_pairs={row['exact_duplicate_pairs']} "
            f"({100 * row['duplicate_pair_rate']:.1f}%), "
            f"mean_abs_len_diff={row['mean_abs_length_diff']:.1f}, "
            f"missing_response={row['missing_response']}"
        )
    if output:
        print(f"\nWrote {output}")


if __name__ == "__main__":
    fire.Fire(audit)

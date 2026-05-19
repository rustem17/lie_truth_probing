import json

from generate_datasets.audit_pairs import summarize_dataset


def test_audit_pairs_counts_exact_duplicate_pair_responses(tmp_path):
    path = tmp_path / "toy.json"
    data = [
        {"id": "a_lie", "condition": "lie", "model_response": "same answer"},
        {"id": "a_truth", "condition": "truth", "model_response": "same   answer"},
        {"id": "b_lie", "condition": "lie", "model_response": "wrong"},
        {"id": "b_truth", "condition": "truth", "model_response": "right answer"},
    ]
    path.write_text(json.dumps(data))

    row = summarize_dataset(path, "toy", {"lie": 1, "truth": 0})

    assert row["paired_groups"] == 2
    assert row["exact_duplicate_pairs"] == 1
    assert row["duplicate_pair_rate"] == 0.5

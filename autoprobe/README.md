# Autoprobe

`autoprobe/` is an autoresearch-style harness for probe architecture search.
It does not regenerate datasets, call vLLM, call judge models, or extract model
activations during smoke runs. It searches over cached feature stores and scores
candidates with a fixed robust-transfer objective.

## Run

From the repo root:

```bash
uv run python -m autoprobe.run --mode smoke --run-tag smoke1
```

Outputs:

```text
autoprobe/results/smoke1/
  results.tsv
  001_residual_contrastive_avg.json
  ...
  best/best_result.json
  best/best_spec.json
```

## Edit Boundary

Autonomous agents should edit only:

```text
autoprobe/candidates.py
```

The fixed evaluator, data registry, datasets, cached activations, and robust
score formula should remain unchanged during a run. This keeps keep/discard
decisions comparable.

## Feature Stores

The current runnable path is residual stream activations from:

```text
activations_{model_tag}/{dataset}.pt
```

Future extractors should write the same metadata contract:

```python
{
    "feature_type": "attention_output",  # or mlp_output, attention_delta, mlp_delta, block_delta
    "features": array,                   # rank 3: samples x layers x dim
                                         # rank 4: samples x layers x heads x dim
    "axes": ["sample", "layer", "head", "dim"],
    "model_tag": "...",
    "model_id": "...",
    "dataset_file": "...",
    "dataset_hash": "...",
    "sample_ids": [...],
    "position": "first",
}
```

Feature paths:

```text
features_attention_output_{model_tag}/{dataset}.pt
features_mlp_output_{model_tag}/{dataset}.pt
features_attention_delta_{model_tag}/{dataset}.pt
features_mlp_delta_{model_tag}/{dataset}.pt
features_block_delta_{model_tag}/{dataset}.pt
```

Non-residual candidates skip cleanly until these stores exist.

## Objective

The fixed objective is:

```text
robust_score =
  val_mean_auroc
  - 0.50 * max(0, 0.75 - val_min_auroc)
  - 0.20 * mean_abs_length_score_spearman
  - 0.10 * dataset_identity_probe_penalty
  - 0.02 * log10(max(runtime_seconds / baseline_runtime_seconds, 1))
```

The runner marks a candidate `keep` if its robust score beats every earlier
successful candidate in the same run. Otherwise it records `discard`, `skipped`,
or `crash`.

## Current Search Space

Smoke candidates cover:

- residual contrastive probes
- residual mass-mean probes
- pooled residual logistic probes
- sparse L1 logistic probes
- V-REx-style variance-penalized layer selection
- group-DRO weighted pooled logistic probes
- top-k transfer-weighted layer ensembles
- residualized length/dataset-confound probes

Full mode also proposes planned attention-output, MLP-output, and component-delta
experiments. They remain skipped until feature stores are extracted.

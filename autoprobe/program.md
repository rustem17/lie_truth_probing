# Autoprobe Research Loop

Goal: improve robust cross-dataset lie/truth probing without changing the
datasets, cached activations, evaluator, or objective.

Editable file:

```text
autoprobe/candidates.py
```

Fixed files during a run:

```text
config.py
extract_activations.py
probes/
eval/
root dataset JSONs
activations_*/
autoprobe/evaluator.py
autoprobe/features.py
autoprobe/probes.py
```

Run command:

```bash
uv run python -m autoprobe.run --mode smoke --run-tag <tag>
```

A good iteration:

1. Edit `autoprobe/candidates.py` to add or tune one small batch of candidate
   `ExperimentSpec`s.
2. Run the smoke command.
3. Inspect `autoprobe/results/<tag>/results.tsv`.
4. Keep ideas that improve `robust_score`, especially when `val_min_auroc`
   improves and confound penalties stay low.
5. Remove or revise ideas that crash, skip unintentionally, or only improve
   mean AUROC while hurting worst-case transfer.

Do not optimize directly on the generated validation files by changing the
fixed evaluator. The score formula and dataset split are the benchmark.

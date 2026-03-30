# Probing Pipeline

All commands run from the project root (`lie_truth_probing/`). Requires a vLLM server for inference, Anthropic API key for judging, and a GPU for activation extraction and probe-on-model evaluation.

---

### Models

`config.py` defines the model registry:

```python
MODEL_REGISTRY = {
    "gemma3-27b":        "gghfez/gemma-3-27b-novision",
    "olmo3-32b-instruct": "allenai/Olmo-3.1-32B-Instruct",
    "olmo3-32b-think":    "allenai/Olmo-3-32B-Think",
    "olmo3-32b-base":     "allenai/Olmo-3-1125-32B",
}
```

Pass the short tag (e.g. `gemma3-27b`) wherever a model argument is accepted. `resolve_model(tag)` maps it to the full HF ID and returns both. The tag propagates through every artifact:

| artifact | naming |
|---|---|
| dataset | `instructed_lie_truth_gemma3-27b.json` |
| activations dir | `activations_gemma3-27b/` |
| per-dataset probe | `contrastive/instructed_probe_gemma3-27b.pt` |
| shared direction | `contrastive/shared_direction_gemma3-27b.pt` |
| multi-run results | `multi_results_gemma3-27b.json` |

Tags are stored in `.pt` metadata. Eval scripts that accept probe paths auto-resolve the model from that metadata — no need to re-specify the model.

---

### Generate sample sets

Each dataset follows `generate.py` → `infer.py` → `build_pairs.py`. The `infer.py` checkpoint (`multi_results_{tag}.json`) is the expensive artifact to preserve.

#### Instructed (lie/truth via system prompt)

```bash
cd generate_datasets/instructed
python generate.py                                    # -> probe_dataset.json (1200 samples)
python generate.py --validation                       # -> probe_dataset_validation.json

python infer.py --model gemma3-27b --n_runs 3         # -> multi_results_gemma3-27b.json
python build_pairs.py --model_tag gemma3-27b --max_diff 50   # -> ../../instructed_lie_truth_gemma3-27b.json
```

#### Game lie (narrative roleplay framing)

```bash
cd generate_datasets/game_lie
python generate.py                                    # -> probe_dataset.json
python infer.py --model gemma3-27b --n_runs 3         # -> multi_results_gemma3-27b.json
python build_pairs.py --model_tag gemma3-27b --max_diff 50   # -> ../../game_lie_truth_gemma3-27b.json
```

#### Spontaneous (MMLU-Pro MCQ, natural errors)

```bash
cd generate_datasets/spontaneous
python generate.py                                                      # -> spontaneous_matched_dataset.json
python infer.py --model gemma3-27b --n_runs 10                          # -> multi_results_gemma3-27b.json

python build_pairs.py --model_tag gemma3-27b --strategy matched --min_correct 7  # -> ../../spontaneous_lie_truth_gemma3-27b.json
python build_pairs.py --model_tag gemma3-27b --strategy inconsistent             # -> ../../spontaneous_inconsistent_gemma3-27b.json
python build_pairs.py --model_tag gemma3-27b --strategy validation               # -> ../../spontaneous_validation_gemma3-27b.json + spontaneous_control_gemma3-27b.json
```

#### Sycophancy (suggestive pressure + "are you sure?" + feedback)

```bash
cd generate_datasets/sycophancy
python generate.py --mode all   # -> answer_probe_dataset.json + are_you_sure_probe_dataset.json

# single-turn suggestive pressure
python infer_answer.py --model gemma3-27b --n_runs 10   # -> answer_multi_results_gemma3-27b.json
python build_pairs.py --source answer --model_tag gemma3-27b    # -> ../../sycophancy_lie_truth_gemma3-27b.json

# two-turn "are you sure?"
python infer_ays.py --model gemma3-27b --n_runs 5               # -> ays_multi_results_gemma3-27b.json
python build_pairs.py --source ays --model_tag gemma3-27b       # -> ../../sycophancy_validation_gemma3-27b.json

# feedback sycophancy
python infer_feedback.py --model gemma3-27b --n_runs 10         # -> feedback_multi_results_gemma3-27b.json
python build_feedback_pairs.py --model_tag gemma3-27b           # -> ../../sycophancy_feedback_gemma3-27b.json
```

#### Batch inference (all datasets)

```bash
bash generate_datasets/run_all_vllm.sh gemma3-27b
```

#### Optional: token-length filtering

```bash
python generate_datasets/filter_datasets.py --tolerance 10   # filters all paired JSONs in root
```

#### Optional: confound diagnostics

```bash
python generate_datasets/diagnose_confounds.py   # auto-discovers all *_lie_truth*.json in root
```

---

### Extract activations

Requires the model on GPU. Extracts hidden states at all layers for each sample in every train and validation dataset.

```bash
python extract_activations.py --model gemma3-27b
```

Output dir is auto-derived: `activations_gemma3-27b/` for the default `first` position, `activations_gemma3-27b_{position}/` for others. Each file: `{name}.pt` containing `activations` (n_samples, n_layers, hidden_dim), `labels`, `label_map`, `model_tag`, `model_id`.

Key args:

`--position` — `first` (default), `last`, `first_assistant`, `last_user`, `mean_assistant`, `mid_assistant`, `first_k_assistant`, `last_k_assistant`

`--adapter_id` — optional LoRA adapter (merged before extraction)

`--datasets` — comma-separated subset (e.g. `instructed,spontaneous`); default: all datasets from `config.py`

`--max_samples` — cap per dataset

`--output_dir` — override auto-derived path

---

### Train probes

All probe scripts read `model_tag` from the activation `.pt` metadata and propagate it to output filenames. Override with `--activations_dir` and `--output_dir`.

#### Mass-mean

```bash
cd probes/mass_mean
python train.py                                              # -> {name}_probe_{tag}.pt + results_{tag}.json
python validate.py                                           # -> validation_results_{tag}.json
python shared_direction.py --datasets instructed,spontaneous,sycophancy   # -> shared_direction_{tag}.pt
python plot.py
```

#### Contrastive (logistic regression on pair diffs)

```bash
cd probes/contrastive
python train.py                                              # -> {name}_probe_{tag}.pt + results_{tag}.json
python validate.py                                           # -> validation_results_{tag}.json
python shared_direction.py --datasets instructed,spontaneous,sycophancy   # -> shared_direction_{tag}.pt
python plot.py
```

Key `shared_direction.py` args: `--layer_range "20,40"`, `--shared_mode average|pooled`, `--C 1.0`, `--agg_mode mean|geometric_median`, `--ensemble none|top_k|transfer_weighted`

#### IRM (IRMv1 / V-REx invariant probe)

```bash
cd probes/irm
python train.py   # -> irm_probe_{tag}.pt
```

Trains `nn.Linear(hidden_dim, 2)` jointly across environments, all 80 layers vectorized. Default envs: instructed, spontaneous, sycophancy.

Key args: `--envs instructed,spontaneous,sycophancy`, `--penalty_mode irm|vrex`, `--lambda_irm 1e3`, `--warmup_steps 100`, `--ramp_steps 100`, `--lr 1e-3`, `--n_epochs 500`

IRM sweep: `python probes/irm/sweep.py` → `sweep_results_{ts}.json`; plot: `python probes/irm/plot_sweep_heatmap.py`

#### Mahalanobis LDA

```bash
cd probes/mahalanobis_lda
python train.py                                        # -> {name}_probe_{tag}.pt + results_{tag}.json
python validate.py                                     # -> validation_results_{tag}.json
python shared_direction.py --datasets instructed,spontaneous,sycophancy   # -> shared_direction_{tag}.pt
python plot.py
```

Uses Fisher LDA on augmented pair diffs. SVD + Woodbury identity avoids forming the d×d scatter matrix.

---

### Analysis

All plot scripts live in `probes/analysis/`. They default to contrastive probes; override with `--probes_dir`. Output goes to `{probes_dir}/plots/` or `--output_dir`. Scripts work from any directory.

```
probes/analysis/
  plot_pca.py              PCA of pair diffs at a given layer
  plot_distributions.py    score distributions per dataset
  plot_correlation.py      cross-dataset probe correlation matrix
  plot_spearman.py         Spearman rank correlation across layers
  plot_spearman_sweep.py   Spearman sweep over layer range
  plot_scatter.py          probe score scatter pairs
  plot_svcca.py            SVCCA similarity across layers / methods
  plot_mahalanobis.py      Mahalanobis distance analysis
  learning_curve.py        AUROC vs training set size
  run_all.sh               run all analysis plots in one shot
```

Run all analysis plots for a method:

```bash
cd probes/contrastive && bash make_plots.sh
bash make_plots.sh --datasets instructed,sycophancy --layer 33
```

Individual scripts:

```bash
python probes/analysis/plot_pca.py
python probes/analysis/plot_pca.py --probes_dir probes/mass_mean --datasets instructed,sycophancy
python probes/analysis/run_all.sh --datasets instructed,spontaneous,sycophancy
```

---

### Evaluation

All eval scripts are in `eval/`. Scripts that need a model load it via `ReplayScorer` in `eval/replay_probe.py` (hook-based dot products, no full hidden-state materialization). Model is auto-resolved from probe `.pt` metadata.

#### AUROC eval on precomputed activations

```bash
python eval/auroc_eval.py
python eval/auroc_eval.py --datasets train --activations_dir activations_gemma3-27b
```

Outputs: CSV + heatmap + layer-sweep plots. Evaluates shared directions, per-scenario directions, residuals, or all.

#### Position sweep

```bash
python eval/sweep_positions.py   # extracts at all positions, trains mass-mean probes, compares AUROC
```

#### Config sweep (offline, no GPU)

```bash
python eval/sweep_shared_configs.py   # grid over C, agg_mode, shared_mode, layer_objective, ensemble
python eval/sweep_layer_trainsets.py  # per-layer AUROC for 18 configs x 4 train-set variations
```

#### Apollo scheming datasets

```bash
# one-time setup
git clone https://github.com/carlo-leonardo-attubato/black_to_white_boosts.git
cd black_to_white_boosts && git lfs pull && cd ..

python eval/probe_apollo.py \
    --probe_paths contrastive/shared_direction.pt,mass_mean/shared_direction.pt \
    --apollo_dir /path/to/black_to_white_boosts/results \
    --datasets ai_liar,sandbagging_wmdp_mmlu \
    --position last
```

Available datasets: `ai_liar`, `ai_liar_deception_dilution_100`, `sandbagging_wmdp_mmlu`, `insider_trading_full`, `apps_backdoors`, `alpaca`

Output: per-dataset JSON + summary AUROC matrix.

---

### File layout

```
lie_truth_probing/
├── config.py                           # MODEL_REGISTRY, dataset registry, resolve_model, dataset_filename
├── extract_activations.py              # step 2
├── upload_to_docent.py
├── PIPELINE.md
│
├── *_lie_truth.json                    # base model paired datasets
├── *_lie_truth_gemma3-27b.json         # per-model tagged paired datasets
├── *_validation.json, *_control.json   # (same tagging applies)
│
├── activations_gemma3-27b/             # tagged activations dir
│   ├── instructed.pt                   # contains model_tag, model_id in metadata
│   └── ...
│
├── probes/
│   ├── mass_mean/
│   │   ├── train.py, validate.py, shared_direction.py, plot.py, make_plots.sh
│   │   ├── {name}_probe_gemma3-27b.pt, shared_direction_gemma3-27b.pt
│   │   ├── results_gemma3-27b.json, validation_results_gemma3-27b.json
│   │   └── plots/
│   ├── contrastive/
│   │   └── (same structure)
│   ├── irm/
│   │   ├── train.py, validate.py, sweep.py, plot_sweep_heatmap.py
│   │   ├── irm_probe_gemma3-27b.pt
│   │   ├── sweep_results_{ts}.json
│   │   └── plots/
│   ├── mahalanobis_lda/
│   │   ├── train.py, validate.py, shared_direction.py, plot.py
│   │   ├── {name}_probe_gemma3-27b.pt, shared_direction_gemma3-27b.pt
│   │   └── results_gemma3-27b.json
│   └── analysis/
│       ├── plot_pca.py, plot_distributions.py, plot_correlation.py
│       ├── plot_spearman.py, plot_spearman_sweep.py, plot_scatter.py
│       ├── plot_svcca.py, plot_mahalanobis.py
│       ├── learning_curve.py, run_all.sh
│
├── eval/
│   ├── probe_utils.py                  # shared: load_directions, default_probe_paths, etc.
│   ├── replay_probe.py                 # ReplayScorer: hook-based dot-product scoring
│   ├── auroc_eval.py                   # AUROC eval on precomputed activations
│   ├── sweep_positions.py              # extract × position, train, compare AUROC
│   ├── sweep_layer_trainsets.py        # layer sweep for 18 configs × 4 train variations
│   ├── sweep_shared_configs.py         # config grid sweep (offline)
│   ├── probe_apollo.py                 # probe transfer to Apollo scheming datasets
│   └── results/                        # timestamped outputs
│
├── results/                            # analysis plot outputs
│
└── generate_datasets/
    ├── run_all_vllm.sh                 # batch inference for all datasets
    ├── filter_datasets.py
    ├── diagnose_confounds.py
    ├── instructed/   generate.py, infer.py, build_pairs.py
    ├── game_lie/     generate.py, infer.py, build_pairs.py
    ├── spontaneous/  generate.py, infer.py, build_pairs.py
    └── sycophancy/   generate.py, infer_answer.py, infer_ays.py, infer_feedback.py, build_pairs.py, build_feedback_pairs.py
```

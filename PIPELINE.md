# Probing Pipeline

All commands run from the project root (`lie_truth_probing/`). Requires a vLLM server for inference, Anthropic API key for judging, and a GPU for activation extraction and probe-on-model evaluation.

Set up the Python environment with `uv`:

```bash
uv sync
uv sync --extra gpu   # optional: install vLLM on the GPU machine
uv sync --python 3.12 --extra lens   # optional: vLLM-Lens activation capture/steering
```

Run Python commands through `uv run`, for example `uv run python extract_activations.py`. The command examples below show `python ...` for readability; use `uv run python ...` unless you have already activated `.venv`.

---

### Models

`config.py` defines the model registry:

```python
MODEL_REGISTRY = {
    "llama-3-3-70b-instruct":       "meta-llama/Llama-3.3-70B-Instruct",
    "gemma-4-31b-it":               "google/gemma-4-31B-it",
    "olmo-2-0325-32b":              "allenai/OLMo-2-0325-32B",
    "qwen3-6-35b-a3b":              "Qwen/Qwen3.6-35B-A3B",
    "glm-5-1":                      "zai-org/GLM-5.1",
    "trinity-large-thinking":       "arcee-ai/Trinity-Large-Thinking",
    "minimax-m2-7":                 "MiniMaxAI/MiniMax-M2.7",
    "kimi-linear-48b-a3b-instruct": "moonshotai/Kimi-Linear-48B-A3B-Instruct",
}
```

Pass the short tag (e.g. `llama-3-3-70b-instruct`) wherever a model argument is accepted. `resolve_model(tag)` maps it to the full HF ID and returns both. The default is `llama-3-3-70b-instruct`. The tag propagates through every artifact:

| artifact | naming |
|---|---|
| dataset | `instructed_system_prompt_llama-3-3-70b-instruct.json` |
| activations dir | `activations_llama-3-3-70b-instruct/` |
| per-dataset probe | `contrastive/instructed_system_prompt_probe_llama-3-3-70b-instruct.pt` |
| shared direction | `contrastive/shared_direction_llama-3-3-70b-instruct.pt` |
| multi-run results | `multi_results_llama-3-3-70b-instruct.json` |

Tags are stored in `.pt` metadata. Eval scripts that accept probe paths auto-resolve the model from that metadata — no need to re-specify the model.

---

### Generate sample sets

Each dataset follows `generate.py` → `infer.py` → `build_pairs.py`. The `infer.py` checkpoint (`multi_results_{tag}.json`) is the expensive artifact to preserve.

#### Instructed system prompt / user prompt

```bash
cd generate_datasets/instructed
python generate.py                                    # -> probe_dataset.json (1200 samples)
python generate.py --validation                       # -> probe_dataset_validation.json

python infer.py --n_runs 3                            # -> multi_results_llama-3-3-70b-instruct.json
python build_pairs.py --max_diff 50                   # -> ../../instructed_system_prompt_llama-3-3-70b-instruct.json

python infer.py --dataset probe_dataset_validation.json --output user_prompt_multi_results_llama-3-3-70b-instruct.json --n_runs 3
python build_pairs.py --input user_prompt_multi_results_llama-3-3-70b-instruct.json --output ../../instructed_user_prompt_llama-3-3-70b-instruct.json --max_diff 50
```

#### Game lie (Werewolf / Mafia)

```bash
cd generate_datasets/game_lie
python generate.py                                    # -> probe_dataset.json
python infer.py --n_runs 3                            # -> multi_results_llama-3-3-70b-instruct.json
python build_pairs.py --max_diff 50                   # -> ../../game_werewolf_llama-3-3-70b-instruct.json

python generate_mafia.py                              # -> mafia_probe_dataset.json
python infer.py --dataset mafia_probe_dataset.json --output mafia_multi_results_llama-3-3-70b-instruct.json --n_runs 3
python build_pairs.py --input mafia_multi_results_llama-3-3-70b-instruct.json --output ../../game_mafia_llama-3-3-70b-instruct.json --max_diff 50
```

#### Spontaneous (MMLU-Pro MCQ, natural errors)

```bash
cd generate_datasets/spontaneous
python generate.py                                                      # -> spontaneous_matched_dataset.json
python infer.py --n_runs 10                                             # -> multi_results_llama-3-3-70b-instruct.json

python build_pairs.py --strategy matched --min_correct 7  # -> ../../spontaneous_1_llama-3-3-70b-instruct.json
python build_pairs.py --strategy inconsistent             # -> ../../spontaneous_inconsistent_llama-3-3-70b-instruct.json
python build_pairs.py --strategy validation               # -> ../../spontaneous_2_llama-3-3-70b-instruct.json + spontaneous_control_llama-3-3-70b-instruct.json
```

#### Sycophancy (suggestive pressure + "are you sure?" + feedback)

```bash
cd generate_datasets/sycophancy
python generate.py --mode all   # -> answer_probe_dataset.json + are_you_sure_probe_dataset.json

# single-turn suggestive pressure
python infer_answer.py --n_runs 10   # -> answer_multi_results_llama-3-3-70b-instruct.json
python build_pairs.py --source answer    # -> ../../sycophancy_answer_llama-3-3-70b-instruct.json

# two-turn "are you sure?"
python infer_ays.py --n_runs 5               # -> ays_multi_results_llama-3-3-70b-instruct.json
python build_pairs.py --source ays           # -> ../../sycophancy_are_you_sure_llama-3-3-70b-instruct.json

# feedback sycophancy
python infer_feedback.py --n_runs 10         # -> feedback_multi_results_llama-3-3-70b-instruct.json
python build_feedback_pairs.py               # -> ../../sycophancy_feedback_llama-3-3-70b-instruct.json
```

#### Batch inference (all datasets)

```bash
bash generate_datasets/run_all_vllm.sh
```

#### Optional: token-length filtering

```bash
python generate_datasets/filter_datasets.py --tolerance 10   # filters all paired JSONs in root
```

#### Optional: confound diagnostics

```bash
python generate_datasets/diagnose_confounds.py   # auto-discovers all *_lie_truth*.json in root
```

#### Optional: vLLM-Lens

`vllm-lens` is available as the optional `lens` extra. It requires Python 3.12, so install it on the GPU/vLLM host with:

```bash
uv sync --python 3.12 --extra lens
```

Use this for a faster experimental activation-extraction path against the vLLM runtime. The existing `extract_activations.py` path remains the reference implementation until the residual-stream tensors from vLLM-Lens are checked against the transformers output on a small dataset.

---

### Extract activations

Requires the model on GPU. Extracts hidden states at all layers for each sample in every train and validation dataset.

```bash
python extract_activations.py
```

Output dir is auto-derived: `activations_llama-3-3-70b-instruct/` for the default `first` position, `activations_llama-3-3-70b-instruct_{position}/` for others. Each file: `{name}.pt` containing `activations` (n_samples, n_layers, hidden_dim), `labels`, `label_map`, `model_tag`, `model_id`, `dataset_file`, `dataset_hash`, and `sample_ids`.

Key args:

`--position` — `first` (default), `last`, `first_assistant`, `last_user`, `mean_assistant`, `mid_assistant`, `first_k_assistant`, `last_k_assistant`

`--datasets` — comma-separated subset (e.g. `instructed_system_prompt,spontaneous_1`); default: all datasets from `config.py`

`--max_samples` — cap per dataset

`--output_dir` — override auto-derived path

---

### Train probes

All probe scripts read `model_tag` from the activation `.pt` metadata and propagate it to output filenames. Override with `--activations_dir` and `--output_dir`.

Training scripts default to `--device auto`: CUDA is used for the main linear algebra when available, otherwise CPU is used. Pass `--device cpu` to force CPU. The contrastive trainer defaults to `--solver auto`, which uses a batched torch/CUDA logistic solver on CUDA and sklearn on CPU; pass `--solver sklearn` to force the original sklearn path or `--solver torch_lbfgs` for the previous per-layer torch solver.

#### Mass-mean

```bash
cd probes/mass_mean
python train.py                                              # -> {name}_probe_{tag}.pt + results_{tag}.json
python validate.py                                           # -> validation_results_{tag}.json
python shared_direction.py --datasets instructed_system_prompt,spontaneous_1,sycophancy_answer   # -> shared_direction_{tag}.pt
python plot.py
```

This is the simple difference-in-means direction: for each pair, form
`d = h_lie - h_truth`, then use `normalize(mean(d))` per layer.

#### Covariance-corrected mass-mean

```bash
cd probes/mass_mean_iid
python train.py                                              # -> {name}_probe_{tag}.pt + results_{tag}.json
python validate.py                                           # -> validation_results_{tag}.json
python shared_direction.py --datasets instructed_system_prompt,spontaneous_1,sycophancy_answer   # -> shared_direction_{tag}.pt
python plot.py
```

This keeps the mass-mean feature direction `mean(d)` and also builds an IID
classification direction `Sigma^-1 mean(d)`, where `Sigma` is the covariance of
centered pair differences. The default saved scoring direction is the corrected
`iid` direction. Use `--score_mode feature` to score with the uncorrected
difference-in-means direction while still saving both.

#### Paired PCA

```bash
cd probes/paired_pca
python train.py                                              # -> {name}_probe_{tag}.pt + results_{tag}.json
python validate.py                                           # -> validation_results_{tag}.json
python shared_direction.py --datasets instructed_system_prompt,spontaneous_1,sycophancy_answer   # -> shared_direction_{tag}.pt
python plot.py
```

This forms the same pair-difference matrix `D = h_lie - h_truth` and uses the
top right singular vector as the probe direction, oriented so that `mean(D)` has
positive projection. By default the pair differences are not centered; pass
`--center=True` to run centered PCA.

#### Contrastive (logistic regression on pair diffs)

```bash
cd probes/contrastive
python train.py                                              # -> {name}_probe_{tag}.pt + results_{tag}.json
python validate.py                                           # -> validation_results_{tag}.json
python shared_direction.py --datasets instructed_system_prompt,spontaneous_1,sycophancy_answer   # -> shared_direction_{tag}.pt
python plot.py
```

Key `shared_direction.py` args: `--layer_range "20,40"`, `--shared_mode average|pooled`, `--C 1.0`, `--agg_mode mean|geometric_median`, `--ensemble none|top_k|transfer_weighted`

#### IRM (IRMv1 / V-REx invariant probe)

```bash
cd probes/irm
python train.py   # -> irm_probe_{tag}.pt
```

Trains `nn.Linear(hidden_dim, 2)` jointly across environments, all 80 layers vectorized. Default envs: instructed_system_prompt, spontaneous_1, sycophancy_answer.

Key args: `--envs instructed_system_prompt,spontaneous_1,sycophancy_answer`, `--penalty_mode irm|vrex`, `--lambda_irm 1e3`, `--warmup_steps 100`, `--ramp_steps 100`, `--lr 1e-3`, `--n_epochs 500`

IRM sweep: `python probes/irm/sweep.py` → `sweep_results_{ts}.json`; plot: `python probes/irm/plot_sweep_heatmap.py`

#### Mahalanobis LDA

```bash
cd probes/mahalanobis_lda
python train.py                                        # -> {name}_probe_{tag}.pt + results_{tag}.json
python validate.py                                     # -> validation_results_{tag}.json
python shared_direction.py --datasets instructed_system_prompt,spontaneous_1,sycophancy_answer   # -> shared_direction_{tag}.pt
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

Run all analysis plots for one non-IRM method:

```bash
cd probes/contrastive && bash make_plots.sh
bash make_plots.sh --model gemma-4-31b-it --datasets instructed_system_prompt,sycophancy_answer --layer 33
```

Run the same analysis plot suite for every non-IRM method with a wrapper:

```bash
bash probes/make_all_plots.sh
bash probes/make_all_plots.sh --model gemma-4-31b-it --datasets instructed_system_prompt,sycophancy_answer --layer 33
```

The wrapper covers `mass_mean`, `mass_mean_iid`, `paired_pca`, `contrastive`,
and `mahalanobis_lda`. Methods without `shared_direction_{tag}.pt` still run
their local training/validation plots and skip the shared-direction analysis
suite instead of failing. IRM keeps its separate sweep heatmap plotter.

Plot scripts require tagged artifacts for the requested model. Legacy untagged
artifacts are ignored unless you explicitly pass
`--allow_untagged_fallback=True`; loaded `.pt` artifacts with `model_tag`
metadata are checked against the requested model. The shell wrappers prefer
`uv run python` automatically; set `PYTHON=/path/to/python` to override.

Individual scripts:

```bash
python probes/analysis/plot_pca.py
python probes/analysis/plot_pca.py --probes_dir probes/mass_mean --datasets instructed_system_prompt,sycophancy_answer
python probes/analysis/run_all.sh --datasets instructed_system_prompt,spontaneous_1,sycophancy_answer
```

### Autoprobe Search

`autoprobe/` is an additive research-loop harness for searching probe variants
against cached feature stores. Smoke mode uses existing residual activations and
does not regenerate datasets, run vLLM, call judges, or extract activations.

```bash
uv run python -m autoprobe.run --mode smoke --run-tag smoke1
```

Autonomous candidates live in `autoprobe/candidates.py`; the fixed evaluator
keeps the robust-transfer objective stable. See `autoprobe/README.md` and
`autoprobe/program.md`.

---

### Evaluation

All eval scripts are in `eval/`. Scripts that need a model load it via `ReplayScorer` in `eval/replay_probe.py` (hook-based dot products, no full hidden-state materialization). Model is auto-resolved from probe `.pt` metadata.

#### AUROC eval on precomputed activations

```bash
python eval/auroc_eval.py
python eval/auroc_eval.py --datasets train --activations_dir activations_llama-3-3-70b-instruct
```

Outputs: CSV + heatmap + layer-sweep plots. The CSV includes both preselected-layer AUROC and diagnostic peak AUROC; the heatmap uses the preselected probe layer to avoid optimistic layer selection on the evaluated dataset.

#### Position sweep

```bash
# End-to-end sweep: extract activations at each position, train probes, validate,
# and write CSV/heatmap summaries under eval/results/<run_name>/
uv run python eval/sweep_positions.py \
    --model llama-3-3-70b-instruct \
    --positions first,last,mean_assistant \
    --methods mass_mean,mass_mean_iid,paired_pca,contrastive,mahalanobis_lda \
    --run_name pos_sweep_llama33

# Include all supported positions.
uv run python eval/sweep_positions.py \
    --model llama-3-3-70b-instruct \
    --positions all \
    --methods mass_mean,mass_mean_iid,paired_pca,contrastive

# Split expensive GPU extraction from offline probe training/validation.
uv run python eval/sweep_positions.py \
    --model llama-3-3-70b-instruct \
    --positions all \
    --phase extract \
    --run_name pos_sweep_llama33

uv run python eval/sweep_positions.py \
    --model llama-3-3-70b-instruct \
    --positions all \
    --phase train,validate,summarize \
    --skip_existing=True \
    --run_name pos_sweep_llama33
```

Supported positions: `first`, `last`, `first_assistant`, `last_user`, `mean_assistant`,
`mid_assistant`, `first_k_assistant`, `last_k_assistant`.

Supported methods: `mass_mean`, `mass_mean_iid`, `paired_pca`, `contrastive`,
`mahalanobis_lda`, `irm`.
Use `--methods all` to run every method. Shared-direction files are optional in
the sweep; add `--run_shared=True` to include each method's `shared_direction.py`
step.

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
├── instructed_system_prompt.json, spontaneous_1.json, ...  # base model paired datasets
├── *_llama-3-3-70b-instruct.json       # per-model tagged paired datasets
│
├── activations_llama-3-3-70b-instruct/      # tagged activations dir
│   ├── instructed_system_prompt.pt     # contains model_tag, model_id in metadata
│   └── ...
│
├── probes/
│   ├── plot_probe_results.py              # shared plot.py helper for non-IRM probe methods
│   ├── make_all_plots.sh                  # run method-level analysis plots for non-IRM methods
│   ├── mass_mean/
│   │   ├── train.py, validate.py, shared_direction.py, plot.py, make_plots.sh
│   │   ├── {name}_probe_llama-3-3-70b-instruct.pt, shared_direction_llama-3-3-70b-instruct.pt
│   │   ├── results_llama-3-3-70b-instruct.json, validation_results_llama-3-3-70b-instruct.json
│   │   └── plots/
│   ├── mass_mean_iid/
│   │   ├── train.py, validate.py, shared_direction.py, plot.py, make_plots.sh
│   │   └── (covariance-corrected mass-mean probes)
│   ├── paired_pca/
│   │   ├── train.py, validate.py, shared_direction.py, plot.py, make_plots.sh
│   │   └── (top-PC directions from pair-difference matrices)
│   ├── contrastive/
│   │   └── (same structure)
│   ├── irm/
│   │   ├── train.py, validate.py, sweep.py, plot_sweep_heatmap.py
│   │   ├── irm_probe_llama-3-3-70b-instruct.pt
│   │   ├── sweep_results_{ts}.json
│   │   └── plots/
│   ├── mahalanobis_lda/
│   │   ├── train.py, validate.py, shared_direction.py, plot.py, make_plots.sh
│   │   ├── {name}_probe_llama-3-3-70b-instruct.pt, shared_direction_llama-3-3-70b-instruct.pt
│   │   └── results_llama-3-3-70b-instruct.json
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

## Project

Deception probing in LLMs. Train linear probes on hidden-state activations to distinguish lie vs truth across multiple deception paradigms: instructed lying, spontaneous errors, sycophancy, game-theoretic deception. Evaluate probe transfer to scheming scenarios (Apollo datasets).

## Environment

Use `uv` from the project root:

    uv sync
    uv sync --extra gpu   # on the remote GPU box if the vLLM server/runtime is needed
    uv sync --python 3.12 --extra lens   # optional vLLM-Lens integration
    uv run python extract_activations.py --help

Code is edited locally. Experiments run on a remote machine with 1 B200 GPU (more can be added). vLLM serves models for dataset generation; transformers loads models for activation extraction and probe evaluation.

`vllm-lens` is available as an optional extra for faster activation capture/steering through vLLM. Keep the transformers extractor as the reference path unless a small parity check confirms that layer indexing, token position selection, and residual-stream values match the saved activation contract.

## Models

`config.MODEL_REGISTRY` maps short tags to HF IDs:

    llama-3-3-70b-instruct          -> meta-llama/Llama-3.3-70B-Instruct
    gemma-4-31b-it                  -> google/gemma-4-31B-it
    olmo-2-0325-32b                 -> allenai/OLMo-2-0325-32B
    qwen3-6-35b-a3b                 -> Qwen/Qwen3.6-35B-A3B
    glm-5-1                         -> zai-org/GLM-5.1
    trinity-large-thinking          -> arcee-ai/Trinity-Large-Thinking
    minimax-m2-7                    -> MiniMaxAI/MiniMax-M2.7
    kimi-linear-48b-a3b-instruct    -> moonshotai/Kimi-Linear-48B-A3B-Instruct

Default model tag is `llama-3-3-70b-instruct`. All artifacts include the model tag in filenames: datasets (`_llama-3-3-70b-instruct.json`), activations (`activations_llama-3-3-70b-instruct/`), probes (`shared_direction_llama-3-3-70b-instruct.pt`). Model tag propagates through .pt metadata; eval scripts auto-resolve model from probe files.

## Pipeline

    generate_datasets/  ->  extract_activations.py  ->  probes/*/train.py  ->  eval/

Probe methods: contrastive (logistic regression on pair diffs), mass_mean (mean-diff direction), irm (IRMv1 invariant), mahalanobis_lda (Fisher LDA).

Dataset names: instructed_system_prompt, instructed_user_prompt, spontaneous_1,
spontaneous_2, spontaneous_control, spontaneous_inconsistent,
sycophancy_answer, sycophancy_are_you_sure, sycophancy_feedback,
game_werewolf, game_mafia.

`autoprobe/` is the autonomous probe-search harness. Agents should edit only
`autoprobe/candidates.py` during a search run; evaluator/data/metrics are fixed.
Smoke command: `uv run python -m autoprobe.run --mode smoke --run-tag <tag>`.

See PIPELINE.md for detailed commands.

## Key files

    config.py               dataset/model registry, resolve_model(), dataset_filename()
    extract_activations.py  extract hidden states per sample per layer
    probes/*/train.py       train per-dataset probes
    probes/*/shared_direction.py  aggregate across conditions
    eval/replay_probe.py    ReplayScorer: hook-based dot-product scoring
    eval/probe_apollo.py    transfer eval on Apollo scheming datasets

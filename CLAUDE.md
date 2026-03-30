## Project

Deception probing in LLMs. Train linear probes on hidden-state activations to distinguish lie vs truth across multiple deception paradigms: instructed lying, spontaneous errors, sycophancy, game-theoretic deception. Evaluate probe transfer to scheming scenarios (Apollo datasets).

## Environment

Code is edited locally. Experiments run on a remote machine with 1 B200 GPU (more can be added). vLLM serves models for dataset generation; transformers loads models for activation extraction and probe evaluation.

## Models

`config.MODEL_REGISTRY` maps short tags to HF IDs:

    gemma3-27b          -> gghfez/gemma-3-27b-novision (62 layers, 5376 dim)
    olmo3-32b-instruct  -> allenai/Olmo-3.1-32B-Instruct (64 layers, 5120 dim)
    olmo3-32b-think     -> allenai/Olmo-3-32B-Think (64 layers, 5120 dim)
    olmo3-32b-base      -> allenai/Olmo-3-1125-32B (64 layers, 5120 dim)

All artifacts include the model tag in filenames: datasets (`_gemma3-27b.json`), activations (`activations_gemma3-27b/`), probes (`shared_direction_gemma3-27b.pt`). Model tag propagates through .pt metadata; eval scripts auto-resolve model from probe files.

## Pipeline

    generate_datasets/  ->  extract_activations.py  ->  probes/*/train.py  ->  eval/

Probe methods: contrastive (logistic regression on pair diffs), mass_mean (mean-diff direction), irm (IRMv1 invariant), mahalanobis_lda (Fisher LDA).

Deception conditions: instructed, spontaneous, sycophancy, sycophancy_feedback, game_lie, spontaneous_inconsistent.

See PIPELINE.md for detailed commands.

## Key files

    config.py               dataset/model registry, resolve_model(), dataset_filename()
    extract_activations.py  extract hidden states per sample per layer
    probes/*/train.py       train per-dataset probes
    probes/*/shared_direction.py  aggregate across conditions
    eval/replay_probe.py    ReplayScorer: hook-based dot-product scoring
    eval/probe_apollo.py    transfer eval on Apollo scheming datasets

"""Agent-editable candidate generator.

The autonomous loop is allowed to edit this file. Keep evaluator/data-loading
code fixed so improvements are measured against a stable objective.
"""

from __future__ import annotations

from .specs import DEFAULT_ENVS, ExperimentSpec


def propose(context: dict) -> list[ExperimentSpec]:
    """Return candidate specs for one autoprobe run.

    Agents should add, remove, or tune specs here. Do not import or mutate the
    fixed evaluator from this module.
    """

    mode = context.get("mode", "smoke")
    pair_cap = 200 if mode == "smoke" else None
    layer_range = (10, 50)

    specs = [
        ExperimentSpec(
            name="residual_contrastive_avg",
            description="Baseline residual LR directions averaged across default environments.",
            method="contrastive",
            shared_mode="average",
            train_datasets=DEFAULT_ENVS,
            layer_range=layer_range,
            pair_cap=pair_cap,
        ),
        ExperimentSpec(
            name="residual_mass_mean",
            description="Mean lie-truth residual difference averaged across default environments.",
            method="mass_mean",
            train_datasets=DEFAULT_ENVS,
            layer_range=layer_range,
            pair_cap=pair_cap,
        ),
        ExperimentSpec(
            name="residual_pooled_lr",
            description="Pooled logistic regression across default environments.",
            method="pooled_logistic",
            shared_mode="pooled",
            C=0.3,
            train_datasets=DEFAULT_ENVS,
            layer_range=layer_range,
            pair_cap=pair_cap,
        ),
        ExperimentSpec(
            name="residual_l1_sparse",
            description="Sparse L1 logistic directions averaged across default environments.",
            method="l1_logistic",
            l1_C=0.05,
            max_iter=1500,
            train_datasets=DEFAULT_ENVS,
            layer_range=layer_range,
            pair_cap=pair_cap,
        ),
        ExperimentSpec(
            name="residual_vrex_style",
            description="Pooled LR with variance-penalized layer selection across environments.",
            method="vrex",
            shared_mode="pooled",
            layer_objective="variance_penalized",
            variance_penalty=0.4,
            train_datasets=DEFAULT_ENVS,
            layer_range=layer_range,
            pair_cap=pair_cap,
        ),
        ExperimentSpec(
            name="residual_group_dro",
            description="Group-DRO weighted pooled LR emphasizing weaker environments.",
            method="group_dro",
            shared_mode="pooled",
            group_dro_rounds=4,
            group_dro_eta=1.5,
            train_datasets=DEFAULT_ENVS,
            layer_range=layer_range,
            pair_cap=pair_cap,
        ),
        ExperimentSpec(
            name="residual_top5_transfer_weighted",
            description="Score ensemble over top five transfer-selected residual layers.",
            method="contrastive",
            ensemble="transfer_weighted",
            ensemble_k=5,
            train_datasets=DEFAULT_ENVS,
            layer_range=layer_range,
            pair_cap=pair_cap,
        ),
        ExperimentSpec(
            name="residualized_length_dataset",
            description="Residual contrastive direction projected away from response-length and dataset confounds.",
            method="contrastive",
            residualize_confounds=("response_length", "dataset"),
            train_datasets=DEFAULT_ENVS,
            layer_range=layer_range,
            pair_cap=pair_cap,
        ),
    ]

    if mode != "smoke":
        specs.extend([
            ExperimentSpec(
                name="attention_output_contrastive",
                description="Per-head attention-output probes; skips until matching feature stores exist.",
                feature_type="attention_output",
                method="contrastive",
                ensemble="top_k",
                ensemble_k=8,
                train_datasets=DEFAULT_ENVS,
                layer_range=layer_range,
                pair_cap=pair_cap,
            ),
            ExperimentSpec(
                name="mlp_output_contrastive",
                description="Per-layer MLP-output probes; skips until matching feature stores exist.",
                feature_type="mlp_output",
                method="contrastive",
                train_datasets=DEFAULT_ENVS,
                layer_range=layer_range,
                pair_cap=pair_cap,
            ),
            ExperimentSpec(
                name="component_delta_block",
                description="Full block-delta probes; skips until matching feature stores exist.",
                feature_type="block_delta",
                method="contrastive",
                train_datasets=DEFAULT_ENVS,
                layer_range=layer_range,
                pair_cap=pair_cap,
            ),
        ])

    for spec in specs:
        spec.validate()
    return specs

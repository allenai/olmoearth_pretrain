"""v1.2 base with a projection-only target encoder, for the speedup A/B.

Identical to ``base_fused_e2e.py`` (fused AdamW, in-loop evals as separate
Beaker jobs) except ``projection_only_target=True``: since every modality has
token exit depth 0 and ``ema_decay=(1.0, 1.0)``, the target encoder never runs
its transformer blocks and is never EMA-updated, so the full frozen encoder
copy is replaced by just the frozen initial projection (patch embeddings +
optional projector). This removes the target's FSDP wrapping/all-gathers, the
unused ``project_and_aggregate`` compute, and the dead blocks from checkpoints.

Targets are mathematically identical to the exit-0 path of the full copy
(see tests/unit/nn/test_latent_mim.py), so the loss curve should track the
fused arm of W&B project 2026_07_08_fused_adamw_e2e (same init/data seeds)
up to non-determinism. Launch via ``launch_target_ddp_e2e.sh``.
"""

import logging

from base import (
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_train_module_config,
    build_visualize_config,
)
from base import build_model_config as _base_build_model_config
from base import build_trainer_config as _base_build_trainer_config

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_2/base_proj_target_e2e.py"
WANDB_PROJECT = "2026_07_08_fused_adamw_e2e"
LOOP_EVAL_CLUSTERS = ["ai2/jupiter", "ai2/ceres"]


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Base model config with the projection-only target encoder."""
    config = _base_build_model_config(common)
    config.projection_only_target = True
    return config


def build_trainer_config(common: CommonComponents):
    """Base trainer config with the in-loop evals run as separate Beaker jobs."""
    trainer_config = _base_build_trainer_config(common)
    evaluator = trainer_config.callbacks["downstream_evaluator"]
    evaluator.run_as_beaker_job = True
    evaluator.beaker_eval_module_path = MODULE_PATH
    evaluator.beaker_eval_clusters = list(LOOP_EVAL_CLUSTERS)
    trainer_config.callbacks["wandb"].project = WANDB_PROJECT
    return trainer_config


if __name__ == "__main__":
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )

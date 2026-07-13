"""v1.2 base with ALL speedups: fused AdamW + projection-only target + replicated DP.

This is the all-changes production candidate. Fused AdamW comes from base.py's
default (fused=True, validated in 2026_07_09_compile_fused_ab); on top of that
it stacks both speedups from the profiling work (see ``base_proj_target_e2e.py``
and ``base_ddp_e2e.py`` for the single-change arms and their rationale):

* ``projection_only_target=True`` -- the frozen exit-0 target is just the
  initial projection; no full encoder copy, no target FSDP all-gathers.
* ``dp_config.name=ddp`` -- replicated params + one coalesced fp32 gradient
  all-reduce per step instead of FSDP's per-layer collectives, with bf16
  autocast for compute.

If the single-change arms hold, this is the production config candidate.
Same init/data seeds as the fused arm of W&B project
2026_07_08_fused_adamw_e2e. Launch via ``launch_target_ddp_e2e.sh``.
"""

import logging

from base import (
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_visualize_config,
)
from base import build_model_config as _base_build_model_config
from base import build_train_module_config as _base_build_train_module_config
from base import build_trainer_config as _base_build_trainer_config
from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelConfig, DataParallelType

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/archived/2026_07_13_v1_2_speedup/base_proj_target_ddp_e2e.py"
WANDB_PROJECT = "2026_07_08_fused_adamw_e2e"
LOOP_EVAL_CLUSTERS = ["ai2/jupiter", "ai2/ceres"]


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Base model config with the projection-only target encoder."""
    config = _base_build_model_config(common)
    config.projection_only_target = True
    return config


def build_train_module_config(common: CommonComponents):
    """Base train module config with replicated DP + bf16 autocast."""
    config = _base_build_train_module_config(common)
    config.dp_config = DataParallelConfig(name=DataParallelType.ddp)
    config.autocast_precision = DType.bfloat16
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

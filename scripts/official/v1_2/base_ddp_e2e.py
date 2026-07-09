"""v1.2 base with replicated data parallelism instead of FSDP, for the speedup A/B.

Identical to ``base_fused_e2e.py`` (fused AdamW, in-loop evals as separate
Beaker jobs) except the data-parallel strategy: the ~160M-param model is small
enough to replicate, so FSDP's ~41 per-layer all-gathers per step (plus CPU
hook overhead) are replaced by a single coalesced gradient all-reduce in
``optim_step`` (``dp_config.name=ddp``; see the train module for why we don't
use the torch DDP reducer -- it can't handle the two contrastive forwards per
backward).

Numerics vs the FSDP arm: params stay fp32 and compute runs under
``autocast(bfloat16)`` instead of FSDP's bf16 param-cast (norms/softmax run in
fp32 under autocast -- same or better precision), and gradients are
mean-reduced in fp32 in both. Same init/data seeds as the fused arm of W&B
project 2026_07_08_fused_adamw_e2e; loss curves should match closely but not
bit-exactly. Launch via ``launch_target_ddp_e2e.sh``.
"""

import logging

from base import (
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_model_config,
    build_visualize_config,
)
from base import build_train_module_config as _base_build_train_module_config
from base import build_trainer_config as _base_build_trainer_config
from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelConfig, DataParallelType

from olmoearth_pretrain.internal.experiment import CommonComponents, main

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_2/base_ddp_e2e.py"
WANDB_PROJECT = "2026_07_08_fused_adamw_e2e"
LOOP_EVAL_CLUSTERS = ["ai2/jupiter", "ai2/ceres"]


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

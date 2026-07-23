"""Smaller-embedding speedups: replicated DP + bf16 autocast + beaker in-loop evals.

Mirrors ``scripts/official/v1_2/base_faster.py``: on top of the local ``base.py``
train module it switches to DDP + bf16 autocast, and runs the 4 in-loop evals as
separate Beaker jobs so they never block training. ``projection_only_target`` is
already set by ``build_model_config`` in ``base.py``.

The variant scripts (base_dim*.py, tiny_up768.py, ...) reuse
``build_train_module_config`` here and build their trainer via
``make_build_trainer_config(<their own module path>)`` so the beaker eval job
re-runs the correct script to rebuild the model.
"""

import logging
from collections.abc import Callable

from base import build_train_module_config as _base_build_train_module_config
from base import build_trainer_config as _base_build_trainer_config
from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelConfig, DataParallelType
from olmo_core.train.config import TrainerConfig

from olmoearth_pretrain.internal.experiment import CommonComponents

logger = logging.getLogger(__name__)

LOOP_EVAL_CLUSTERS = ["ai2/jupiter", "ai2/ceres"]


def build_train_module_config(common: CommonComponents):
    """Base train module config with replicated DP + bf16 autocast."""
    config = _base_build_train_module_config(common)
    config.dp_config = DataParallelConfig(name=DataParallelType.ddp)
    config.autocast_precision = DType.bfloat16
    return config


def make_build_trainer_config(
    module_path: str,
) -> Callable[[CommonComponents], TrainerConfig]:
    """Return a trainer-config builder that runs the in-loop evals as beaker jobs.

    Args:
        module_path: Repo-relative path of the launched variant script, so the
            beaker eval job re-runs it to rebuild the same model.
    """

    def build_trainer_config(common: CommonComponents) -> TrainerConfig:
        trainer_config = _base_build_trainer_config(common)
        evaluator = trainer_config.callbacks["downstream_evaluator"]
        evaluator.run_as_beaker_job = True
        evaluator.beaker_eval_module_path = module_path
        evaluator.beaker_eval_clusters = list(LOOP_EVAL_CLUSTERS)
        return trainer_config

    return build_trainer_config

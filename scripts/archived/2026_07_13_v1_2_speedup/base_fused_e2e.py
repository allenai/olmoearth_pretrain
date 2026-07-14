"""v1.2 base with in-loop evals routed to separate Beaker jobs, for the fused-AdamW E2E.

Wraps ``base.py`` (which now defaults to ``fused=True``) so the downstream
evaluators launch as separate Beaker jobs against saved checkpoints instead of
blocking training -- the training wall clock is then a clean end-to-end
measurement. Launch both arms with ``launch_fused_e2e.sh``:

* fused (the new default), and
* the pre-change baseline via ``--train_module.optim_config.fused=false``.

Everything else (model, data, masking, eval task set, checkpoint cadence) is
exactly ``base.py``. ``beaker_eval_module_path`` points back at THIS file so
the eval jobs rebuild the matching (base) architecture, mirroring how the
regbtl scripts do it.

Shorter-horizon validation behind this change: at 8000 steps (matched seeds
and data order vs production run q3x9lvww) fused-only tracked the baseline
loss to +0.0015 -- indistinguishable from a bit-identical control rerun --
at 0.79-0.81x the step time. See W&B project 2026_07_09_compile_fused_ab.
"""

import logging

from base import (
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_model_config,
    build_train_module_config,
    build_visualize_config,
)
from base import build_trainer_config as _base_build_trainer_config

from olmoearth_pretrain.internal.experiment import CommonComponents, main

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/archived/2026_07_13_v1_2_speedup/base_fused_e2e.py"
WANDB_PROJECT = "2026_07_08_fused_adamw_e2e"
# Clusters the in-loop eval Beaker jobs may run on (same as the regbtl runs).
LOOP_EVAL_CLUSTERS = ["ai2/jupiter", "ai2/ceres"]


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

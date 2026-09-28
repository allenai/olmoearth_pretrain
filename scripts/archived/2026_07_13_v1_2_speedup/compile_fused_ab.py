"""Longer A/B to validate compile_model + fused AdamW before adopting them.

The 500-step speed benchmark (``speed_benchmark.py``) measured a 1.37x
throughput gain from ``--train_module.compile_model=true`` +
``--train_module.optim_config.fused=true`` with in-family loss at step 500.
This script runs the same production config for longer (default 20k steps,
~3-4h) so the loss curves can be compared for subtle drift, and so any
torch.compile recompilation problems have time to surface.

Differences from the production script:

* ``max_duration`` is ``AB_STEPS`` (default 20000; env-var override);
* in-loop evals are removed (loss curves are the comparison signal --
  permanent checkpoints still land every 5000 steps, so eval Beaker jobs can
  be pointed at them afterwards if wanted); and
* W&B project is the A/B project below.

Checkpointing is kept at production cadence so preempted runs resume instead
of restarting. Both arms share the init seed and dataloader seed, so weights
and data order are identical; launch them with ``launch_compile_fused_ab.sh``
(one plain arm, one with the two overrides).

Model variant matches ``speed_benchmark.py``: ``SPEED_BENCH_MODEL`` in
``{auto, base, regbtl}``, defaulting to regbtl when ``regbtl_v1_2_common`` is
importable (the register-bottleneck branch).
"""

import logging
import os

from base import (
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_train_module_config,
    build_visualize_config,
)
from base import build_model_config as _base_build_model_config
from base import build_trainer_config as _base_build_trainer_config
from olmo_core.train.common import Duration

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

try:
    from regbtl_v1_2_common import build_regbtl_model_config

    _HAS_REGBTL = True
except ImportError:
    _HAS_REGBTL = False

logger = logging.getLogger(__name__)

AB_STEPS = int(os.environ.get("AB_STEPS", "20000"))
MODEL_VARIANT = os.environ.get("SPEED_BENCH_MODEL", "auto")
WANDB_PROJECT = "2026_07_09_compile_fused_ab"


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model under test: regbtl when available/requested, else v1.2 base."""
    variant = MODEL_VARIANT
    if variant == "auto":
        variant = "regbtl" if _HAS_REGBTL else "base"
    if variant == "regbtl":
        if not _HAS_REGBTL:
            raise RuntimeError(
                "SPEED_BENCH_MODEL=regbtl but regbtl_v1_2_common is not importable; "
                "launch from the register-bottleneck branch or use SPEED_BENCH_MODEL=base"
            )
        logger.info(
            "Compile+fused A/B model variant: regbtl (gdyn_d768_il_pdproj, nolsa)"
        )
        return build_regbtl_model_config(common, latent_self_attn=False)
    logger.info("Compile+fused A/B model variant: v1.2 base")
    return _base_build_model_config(common)


def build_trainer_config(common: CommonComponents):
    """Production trainer config, shortened, without in-loop evals."""
    trainer_config = _base_build_trainer_config(common)
    trainer_config.max_duration = Duration.steps(AB_STEPS)
    trainer_config.callbacks.pop("downstream_evaluator", None)
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

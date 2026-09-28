"""v1.2 base with all validated speedups: fused AdamW + projection-only target + replicated DP.

The production outcome of the July 2026 speedup program
(``scripts/archived/2026_07_13_v1_2_speedup/``, W&B ``2026_07_08_fused_adamw_e2e``):
1.39x end-to-end training speedup on 8xH100 (full 667k-step run 104.4h vs
~147h; median step 0.465s vs 0.636s), with matched-step loss identical to the
fourth decimal across the whole run and downstream evals within seed noise at
every rung. On top of ``base.py`` (whose AdamW is already ``fused=True``):

* ``projection_only_target=True`` -- the frozen exit-0 target encoder is just
  the initial projection; no dead full-encoder copy in checkpoints or grads.
* ``dp_config.name=ddp`` + bf16 autocast -- replicated params with one
  coalesced fp32 gradient all-reduce per step instead of FSDP's per-layer
  collectives.

torch.compile is deliberately EXCLUDED: on both the FSDP and this DDP stack it
degrades training (the loss gap opens at the ~2.5-3k-step phase transition and
grows monotonically, 10-20x outside the seed-noise envelope; see
``base_proj_target_ddp_compile_e2e.py`` in the archive). Do not enable
``compile_model`` here without a full-run quality revalidation.

In-loop evals run as separate Beaker jobs against saved checkpoints, so they
never block training. Watch VRAM: the combined config ran at ~90% allocated;
if a variant OOMs, drop ``rank_microbatch_size`` 64 -> 32.
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

MODULE_PATH = "scripts/official/v1_2/base_faster.py"
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

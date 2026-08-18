"""SMALL backbone, d384 teacher + detached perceiver [128, 64] student (pcv), sup768, w1, newsampling psuniform.

The small-backbone twin of
``regbtl_v1_2_gdyn_d768_proj128pcv_sup768_w1_newsampling_psuniform``. The v1.2 size
sweep's ViT-Small (384-d, depth 12, 6 heads) is very performant relative to base for
its cost, so the two w1 pcv arms are re-run on it: if the student's ceiling is set by
distillation quality rather than raw teacher capacity, the small backbone should keep
most of the d768 student's quality at a fraction of the training and serving cost.

Changes vs the d768 sibling, and nothing else:

* encoder/decoder size preset ``small_shallow_decoder`` (384-d, depth 12, 6 heads);
* teacher register width 768 -> 384. ``wideread`` ties the bottleneck's ATTENTION to
  the encoder width, so on a 384-d encoder a d768 register grid would store wider
  than anything it can read; d384 keeps storage == read width and the 6 x 64 head
  shape RoPE wants;
* LR 1e-4 -> 2e-4, the small preset's value from the size sweep's best-LR search
  (``small.py``). Fused AdamW, weight decay, warmup and clipping are unchanged.

The student is unchanged at [128, 64] -- the shipped embedding widths do not move with
the backbone, so these runs are directly comparable to the d768 arms on exactly the
same probes. Evals are identical to the d768 w1 runs: the d768-recipe embedding suite
(PASTIS ws16/ps1 first, then the projected _proj128 / _proj64 duplicates and
fifty_cities) every 40k steps, plus the shared catalog.

This arm: supervision heads on the register grid only (sup768 -- named for the recipe,
here the heads sit on the d384 registers); the student trains purely by distillation.

Recipe otherwise matches the d768 w1 run: wideread regbtl, regsup base_weight 1.0,
decorrelated sampler at UNIFORM patch sizes.
"""

import logging

from base import build_trainer_config as _base_build_trainer_config
from regbtl_v1_2_faster_common import build_faster_train_module_config
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_common_components,
    build_dataset_config,
    build_visualize_config,
)
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_dataloader_config as _base_build_dataloader_config,
)
from regbtl_v1_2_newsampling_common import (
    apply_microbatch,
    apply_new_sampling,
    apply_uniform_patch_sizes,
)
from regbtl_v1_2_proj_common import (
    SMALL_REGISTER_DIM,
    SMALL_SIZE_NAME,
    SUPERVISION_BASE_WEIGHT_W1,
    add_proj_loop_eval_beaker_job,
    apply_small_learning_rate,
    build_proj_model_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_2/regbtl_v1_2_small_gdyn_d384_proj128pcv_sup768_w1_newsampling_psuniform.py"


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Small encoder, d384 teacher + detached perceiver student, heads on the registers."""
    return build_proj_model_config(
        common,
        base_weight=SUPERVISION_BASE_WEIGHT_W1,
        projection_type="perceiver",
        supervision_source="registers",
        register_dim=SMALL_REGISTER_DIM,
        size_name=SMALL_SIZE_NAME,
    )


def build_dataloader_config(common: CommonComponents):
    """Newsampling dataloader at uniform patch sizes."""
    return apply_uniform_patch_sizes(
        apply_new_sampling(_base_build_dataloader_config(common))
    )


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """1fwd + fused AdamW at the newsampling microbatch size, at the small LR."""
    return apply_small_learning_rate(
        apply_microbatch(build_faster_train_module_config(common))
    )


def build_trainer_config(common: CommonComponents):
    """Base trainer + in-loop evals on BOTH the register and projected 128d heads."""
    return add_proj_loop_eval_beaker_job(
        _base_build_trainer_config(common),
        MODULE_PATH,
        embedding_eval_interval_steps=40000,
    )


def run() -> None:
    """Run the experiment."""
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )


if __name__ == "__main__":
    run()

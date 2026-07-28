"""d128 wideread + regsup (w0p1, newsampling, uniform ps) at time_priority_prob 0.75.

``regbtl_v1_2_gdyn_d128_wideread_regsup_w0p1_newsampling_psuniform`` (the centre of the
budget x bias 2x2) with ``time_priority_prob`` raised from the committed 0.5 to 0.75:
three quarters of batches sample the timestep count first (biased toward the full
sequence) and then a grid that fits it, instead of an even split with grid-first.
Budget stays at 3072 and every other knob is unchanged, so this is compute-neutral --
same tokens per step, same 662,700-step schedule as the centre run.

WHY: time-first and grid-first draws produce different marginals -- time-first
over-represents long sequences (which only small grids can afford), grid-first
over-represents large grids with short sequences. If the newsampling gain comes from
full-sequence temporal exposure, weighting the time-first branch up should move the
frozen ps=1 probes; if it is flat, the 0.5 split is not load-bearing. A/B partner:
``..._w0p1_newsampling_psuniform`` (time_priority_prob 0.5).
"""

import logging

from base import build_trainer_config as _base_build_trainer_config
from regbtl_v1_2_common import add_all_embedding_loop_evals
from regbtl_v1_2_faster_common import (
    build_faster_train_module_config,
    build_wideread_regbtl_model_config,
)
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_common_components,
    build_dataset_config,
    build_visualize_config,
)
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_dataloader_config as _base_build_dataloader_config,
)
from regbtl_v1_2_newsampling_common import (
    SUPERVISION_BASE_WEIGHT,
    apply_microbatch,
    apply_new_sampling,
    apply_uniform_patch_sizes,
)
from regbtl_v1_2_regsup_common import add_register_supervision

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

REGISTER_DIM = 128
TIME_PRIORITY_PROB = 0.75
MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_gdyn_d128_wideread_regsup_w0p1_newsampling_psuniform_tp0p75.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d128 wideread + register-grid supervision at w0p1 (base_weight 0.1)."""
    config = build_wideread_regbtl_model_config(
        common, latent_self_attn=True, register_dim=REGISTER_DIM
    )
    return add_register_supervision(
        config, include_latlon=False, base_weight=SUPERVISION_BASE_WEIGHT
    )


def build_dataloader_config(common: CommonComponents):
    """Uniform-patch-size newsampling dataloader with time_priority_prob 0.75."""
    config = apply_uniform_patch_sizes(
        apply_new_sampling(_base_build_dataloader_config(common))
    )
    config.time_priority_prob = TIME_PRIORITY_PROB
    return config


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """1fwd + fused AdamW train module with a halved rank microbatch size."""
    return apply_microbatch(build_faster_train_module_config(common))


def build_trainer_config(common: CommonComponents):
    """Base trainer + the FULL embedding eval set (PASTIS all ws + AEF), via Beaker.

    Unlike the other sweep arms (ws16 PASTIS only), this run's eval job also covers the
    smaller PASTIS window sizes and the AEF supplemental LP/kNN probes.
    """
    return add_all_embedding_loop_evals(_base_build_trainer_config(common), MODULE_PATH)


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

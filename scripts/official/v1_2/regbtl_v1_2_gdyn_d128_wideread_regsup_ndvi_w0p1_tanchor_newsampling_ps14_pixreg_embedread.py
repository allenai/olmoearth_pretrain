"""Pixel registers + an extra patch-embed register read (``embedread`` arm).

``regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_ps14_pixreg``
(run 1 of the pixreg group: pixel-resolution registers, ps 1..4, mb 32) plus ONE
extra cross-attention read of the PATCH-EMBED output (``register_embed_read``): the
bottleneck reads the trunk's block-0 input -- through its own input norm + K/V
projection, at the same wideread shape as the standard reads -- before the standard
``[read -> LSA] x 4`` schedule. The extra read is zero-initialized, so this model
equals run 1 exactly at initialization.

WHY: at ps=4 the patch embed is a full linear map on the 4x4xC pixel block
(~160 values into 768 dims), so sub-patch content is still exactly present in the
embed tokens -- but the reads only see the FINAL trunk layer, which has mixed that
phase information away. Reading the embed tokens directly lets each pixel
register's RoPE query phase select its own pixel's subspace from the undegraded
linear projection, testing whether pixel fidelity can be RECOVERED from a coarse
trunk rather than re-computed by a pixel branch (the convbranch/thinconv arms'
answer, at much higher pixel-level compute). A/B partner: ``..._ps14_pixreg``
(identical except the extra read).
"""

import logging

from base import build_trainer_config as _base_build_trainer_config
from regbtl_v1_2_common import add_loop_eval_beaker_job
from regbtl_v1_2_faster_common import build_wideread_regbtl_model_config
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_common_components,
    build_visualize_config,
)
from regbtl_v1_2_newsampling_common import (
    SUPERVISION_BASE_WEIGHT,
    apply_new_sampling,
)
from regbtl_v1_2_pixreg_common import (
    apply_embed_read,
    apply_pixel_registers,
    apply_pixreg_microbatch,
    apply_ps14,
)
from regbtl_v1_2_regsup_common import (
    add_register_supervision,
    build_extra_decode_dataloader_config,
    build_extra_decode_dataset_config,
    build_extra_decode_train_module_config,
)

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)

REGISTER_DIM = 128
REGISTER_TEMPORAL_ANCHOR = "year_start"
EXTRA_DECODE_MODALITIES = [Modality.NDVI.name]
MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_ps14_pixreg"
    "_embedread.py"
)
WANDB_PROJECT = "2026_08_19_pixel_branch"


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Run 1's pixel-register model plus the zero-init patch-embed read."""
    config = build_wideread_regbtl_model_config(
        common, latent_self_attn=True, register_dim=REGISTER_DIM
    )
    config.encoder_config.register_temporal_anchor = REGISTER_TEMPORAL_ANCHOR
    config = add_register_supervision(
        config,
        include_latlon=False,
        include_ndvi=True,
        base_weight=SUPERVISION_BASE_WEIGHT,
    )
    return apply_embed_read(apply_pixel_registers(config))


def build_dataset_config(common: CommonComponents):
    """Base dataset config, additionally deriving ndvi from the raw S2 bands."""
    return build_extra_decode_dataset_config(common, EXTRA_DECODE_MODALITIES)


def build_dataloader_config(common: CommonComponents):
    """ndvi-aware newsampling dataloader, patch sizes uniform over 1..4."""
    return apply_ps14(
        apply_new_sampling(
            build_extra_decode_dataloader_config(common, EXTRA_DECODE_MODALITIES)
        )
    )


def build_train_module_config(common: CommonComponents):
    """ndvi-aware faster train module at the pixreg microbatch (32)."""
    return apply_pixreg_microbatch(
        build_extra_decode_train_module_config(common, EXTRA_DECODE_MODALITIES)
    )


def build_trainer_config(common: CommonComponents):
    """Beaker-job evals + logging to the pixel-branch W&B project."""
    trainer_config = add_loop_eval_beaker_job(
        _base_build_trainer_config(common), MODULE_PATH
    )
    trainer_config.callbacks["wandb"].project = WANDB_PROJECT
    return trainer_config


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

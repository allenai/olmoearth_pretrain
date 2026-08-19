"""Pixel registers + standalone thin conv stack (run 3 of the pixreg group).

``..._ps14_pixreg`` plus a STANDALONE thin pixel conv net that never interacts with
the coarse encoder: the same dense pixel packing and per-(timestep, bandset) frame
layout as ``..._ps14_pixreg_convbranch``, but 4 unconditioned ConvNeXt-style blocks
at 128 dim (no FiLM, no fusion back to the coarse trunk, affine-free LNs), run once
on the ONLINE-zeroed input in parallel with the trunk. The output is pooled over
(timestep, bandset) per pixel -- ONLINE-only -- into the same zero-init
``register_init`` handoff as run 2, so at initialization this model also equals
``..._ps14_pixreg`` exactly. The registers then do the normal coarse-token reads +
latent self-attention.

WHY: run 2 entangles two mechanisms -- a high-resolution register init AND
bidirectional coupling with the coarse trunk (FiLM conditioning, fusion residuals,
one conv step per 4 blocks). This arm keeps only the first at a fraction of the
cost (4 cheap conv blocks total, zero trunk overhead). If it matches run 2, the
coupling is dead weight and the cheap variant wins; if run 2 clearly leads, the
pixel stream genuinely needs trunk context. A/B partners: ``..._ps14_pixreg``
(no conv net) and ``..._ps14_pixreg_convbranch`` (interleaved coupled branch).
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
    apply_pixel_branch,
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
    "_thinconv.py"
)
WANDB_PROJECT = "2026_08_19_pixel_branch"


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Pixel-register model (run 1) + the standalone thin conv register init."""
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
    return apply_pixel_branch(apply_pixel_registers(config), "thinconv")


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

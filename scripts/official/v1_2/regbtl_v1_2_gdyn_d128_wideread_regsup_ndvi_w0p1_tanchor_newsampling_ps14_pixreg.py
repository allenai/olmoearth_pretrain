"""Pixel-resolution registers, ps 1..4 (run 1 of the pixreg group).

``regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform``
with the register grid moved to PIXEL resolution: one 128-dim register per pixel
(``register_pixel_grid``), reading from the coarse trunk at whatever patch size the
batch sampled. Latent self-attention stays TRUE but narrows to width 128 (2 x 64
heads); the reads keep the wideread encoder-width shape. Patch sizes are sampled
uniformly over 1..4 (ps>4 pixel grids make the quadratic LSA unaffordable) and the
rank microbatch drops to 32 for the larger register grids. regsup+NDVI (w0p1),
``register_temporal_anchor="year_start"``, and the rest of the newsampling recipe
are unchanged. See ``regbtl_v1_2_pixreg_common`` for the shared knobs.

WHY: the register grid is the representation the decoder and the frozen probes
consume, and at ps>1 today it cannot resolve anything finer than a patch cell. A
pixel-resolution grid gives every pixel its own latent regardless of trunk patch
size -- the encoder's information already covers the pixels (each patch token SAW
its pixels), the registers just currently have nowhere to put per-pixel detail.
A/B partners: ``..._newsampling_ps14`` (same sampler, patch-resolution registers)
isolates the pixel grid; ``..._ps14_pixreg_convbranch`` / ``..._ps14_pixreg_thinconv``
add a pixel-level encoder branch on top of this run.
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
    "regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_ps14_pixreg.py"
)
WANDB_PROJECT = "2026_08_19_pixel_branch"


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d128 wideread + regsup/NDVI + anchored read, registers at pixel resolution."""
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
    return apply_pixel_registers(config)


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

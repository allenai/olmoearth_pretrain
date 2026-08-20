"""Pixel registers + per-pixel raw-band reconstruction (``pixrecon`` arm).

``regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_ps14_pixreg``
(run 1 of the pixreg group: pixel-resolution registers, ps 1..4, mb 32) plus
TIME-CONDITIONED reconstruction of the raw inputs from the register grid: the NDVI
head mechanism (small MLP on ``[register_cell ; phi(day_of_year)]``, evaluated at
every observed timestep, MISSING_VALUE-masked) pointed at the normalized S2 L2A
(12 bands) and S1 (2 bands) values themselves, with MSE at weight 0.05 each (0.1
total -- the NDVI head's own weight). The targets are the raw inputs already in
every batch, so only the head config changes; the architecture is identical to
run 1.

WHY: the pixel-register MIM targets sit at the sampled patch size and the map-
modality supervision is largely static, so nothing except NDVI directly demands
that a pixel register store its own pixel's temporal signature. Reconstructing the
raw bands per (pixel, timestep) is the strongest cheap detail-forcing signal
available: the head is tiny (the register must store the trajectory, the head just
decodes it given time), and the DOY basis limits the demand to the seasonal
component -- clouds and one-off events stay unpredictable and are ignored, exactly
as with NDVI. A/B partner: ``..._ps14_pixreg`` (identical except the two extra
heads).
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
    apply_pixel_reconstruction,
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
    "_pixrecon.py"
)
WANDB_PROJECT = "2026_08_19_pixel_branch"


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Run 1's pixel-register model plus S2 L2A + S1 reconstruction heads."""
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
    return apply_pixel_reconstruction(apply_pixel_registers(config))


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

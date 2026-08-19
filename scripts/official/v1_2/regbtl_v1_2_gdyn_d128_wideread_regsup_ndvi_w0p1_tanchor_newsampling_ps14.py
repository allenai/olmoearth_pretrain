"""ps 1..4 control for the pixreg group (patch-resolution registers).

``regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform``
with ONE change: patch sizes sampled uniformly over 1..4 instead of 1..8
(``regbtl_v1_2_pixreg_common.apply_ps14``). No pixel registers, no pixel branch,
microbatch stays at the newsampling 64.

WHY: the three pixreg arms (``..._ps14_pixreg`` / ``_convbranch`` / ``_thinconv``)
all need the ps<=4 cap (a ps=8 pixel-register grid would be 65k registers, whose
quadratic latent self-attention is unaffordable), so they differ from the existing
psuniform run in the ps distribution AS WELL AS the register grid. This run is the
fair A/B anchor: vs psuniform it isolates the ps 1..4 restriction itself; vs
``..._ps14_pixreg`` it isolates the pixel-resolution register grid.
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
    apply_microbatch,
    apply_new_sampling,
)
from regbtl_v1_2_pixreg_common import apply_ps14
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
    "regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_ps14.py"
)
WANDB_PROJECT = "2026_08_19_pixel_branch"


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d128 wideread + regsup incl. time-conditioned NDVI, anchored register read."""
    config = build_wideread_regbtl_model_config(
        common, latent_self_attn=True, register_dim=REGISTER_DIM
    )
    config.encoder_config.register_temporal_anchor = REGISTER_TEMPORAL_ANCHOR
    return add_register_supervision(
        config,
        include_latlon=False,
        include_ndvi=True,
        base_weight=SUPERVISION_BASE_WEIGHT,
    )


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
    """ndvi-aware faster train module with a halved rank microbatch size."""
    return apply_microbatch(
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

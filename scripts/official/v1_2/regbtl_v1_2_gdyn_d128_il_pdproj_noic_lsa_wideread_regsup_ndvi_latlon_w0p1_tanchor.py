"""d128 wideread + regsup + NDVI + latlon (w0p1), anchored register read, OLD sampling.

``regbtl_v1_2_gdyn_d128_il_pdproj_noic_lsa_wideread_regsup_ndvi_w0p1_tanchor`` with
the latlon location-regression arm added (unit-sphere xyz from the mean-pooled
register grid; a supervised TARGET, never a model input). See that script and
``regbtl_v1_2_regsup_common`` for the NDVI time-conditioned head details. NOTE:
under this OLD sampling recipe ndvi counts against the token budget; the
newsampling variant excludes decode-only modalities from the budget automatically.

In-loop evals run as separate Beaker jobs (run_as_beaker_job=True) and include the
fifty_cities random-split S2 + S1+S2 probes.
"""

import logging

from base import build_trainer_config as _base_build_trainer_config
from regbtl_v1_2_common import add_loop_eval_beaker_job
from regbtl_v1_2_faster_common import build_wideread_regbtl_model_config
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_common_components,
    build_visualize_config,
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
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

REGISTER_DIM = 128
# 10x the regsup_common SUPERVISION_WEIGHT nudge (0.01 -> 0.1).
SUPERVISION_BASE_WEIGHT = 0.1
REGISTER_TEMPORAL_ANCHOR = "year_start"
EXTRA_DECODE_MODALITIES = [Modality.NDVI.name, Modality.LATLON.name]
MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_gdyn_d128_il_pdproj_noic_lsa_wideread_regsup_ndvi_latlon_w0p1_tanchor.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d128 wideread + regsup incl. NDVI + latlon heads, anchored register read."""
    config = build_wideread_regbtl_model_config(
        common, latent_self_attn=True, register_dim=REGISTER_DIM
    )
    config.encoder_config.register_temporal_anchor = REGISTER_TEMPORAL_ANCHOR
    return add_register_supervision(
        config,
        include_latlon=True,
        include_ndvi=True,
        base_weight=SUPERVISION_BASE_WEIGHT,
    )


def build_dataset_config(common: CommonComponents):
    """Base dataset config + derived ndvi + latlon loaded from the h5 files."""
    return build_extra_decode_dataset_config(common, EXTRA_DECODE_MODALITIES)


def build_dataloader_config(common: CommonComponents):
    """1fwd dataloader whose masking knows ndvi + latlon are decode-only."""
    return build_extra_decode_dataloader_config(common, EXTRA_DECODE_MODALITIES)


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """Faster (1fwd + fused AdamW + ddp/bf16) train module, extras decode-only."""
    return build_extra_decode_train_module_config(common, EXTRA_DECODE_MODALITIES)


def build_trainer_config(common: CommonComponents):
    """Base trainer config + fifty_cities evals routed through a Beaker job."""
    return add_loop_eval_beaker_job(_base_build_trainer_config(common), MODULE_PATH)


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

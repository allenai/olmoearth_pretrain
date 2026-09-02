"""cand_ndvi ported onto the NEW MAP SET and LANDSAT REFLECTANCE.

The shipping candidate recipe -- ``scripts/official/v1_2/
regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform.py``
(cand_ndvi) -- rebuilt on the new-maps base. Model, sampler, weights and losses are
the candidate's; only the data and the map set move:

* maps: ``srtm`` -> ``glo30`` (elevation band only, the direct analog of the old
  single-band SRTM elevation target) and ``wri_canopy_height_map`` ->
  ``meta_canopy_height``. ``gse`` and ``worldpop`` are not in the new H5 at all.
* radiometry: the Landsat modality is top-of-atmosphere reflectance (B1-B9) /
  brightness temperature (B10-B11) with per-scene sun-elevation correction, read with
  the matching reflectance-scale norm stats.

Everything the candidate is judged on is kept: register_dim 128 (the shipped storage
width, reading at encoder width 768), regsup at base_weight 0.1, the time-conditioned
NDVI arm on the derived decode-only ``ndvi`` modality, the ``year_start`` temporal
anchor on the register read, decorrelated newsampling at uniform patch sizes, and
1fwd + fused AdamW. No cloud masking (that is a separate candidate sibling).

READ IT AGAINST: ``regbtl_v1_2_gdyn_d128_wideread_regsup_w0p1_newsampling_psuniform_
landsat_refl`` (the in-flight d128 w0p1 reflectance arm), which is this run minus the
NDVI arm and the anchor. Those two knobs move together here because the candidate has
both; the official v1_2 A/B already separated them on the old maps.

The run name is shortened to ``regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_
tanchor_psuniform_newmaps_refl`` (78 chars): the in-loop eval callback has only 94
characters of train-run name before Beaker's 128-char experiment-name limit forces a
truncation, and the full-convention name overruns it.
"""

import logging

from base import build_common_components, build_visualize_config
from base import build_trainer_config as _base_build_trainer_config
from perceiver_common import (
    SUPERVISION_BASE_WEIGHT,
    add_register_supervision,
    apply_landsat_reflectance,
    apply_microbatch,
    apply_new_sampling,
    apply_uniform_patch_sizes,
    build_extra_decode_dataloader_config,
    build_extra_decode_dataset_config,
    build_extra_decode_train_module_config,
    build_wideread_regbtl_model_config,
    route_loop_evals_to_beaker,
)

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig
from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

REGISTER_DIM = 128
REGISTER_TEMPORAL_ANCHOR = "year_start"
# ndvi is derived in the dataset from the raw S2 L2A B04/B08 bands and rides along in
# the batch as a decode-only modality, excluded from the token budget.
EXTRA_DECODE_MODALITIES = [Modality.NDVI.name]
MODULE_PATH = (
    "scripts/vnext/2026_07_24_new_maps/"
    "regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform"
    "_landsat_refl.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d128 wideread + new-maps regsup incl. NDVI, anchored register read."""
    config = build_wideread_regbtl_model_config(
        common, latent_self_attn=True, register_dim=REGISTER_DIM
    )
    config.encoder_config.register_temporal_anchor = REGISTER_TEMPORAL_ANCHOR
    return add_register_supervision(
        config,
        base_weight=SUPERVISION_BASE_WEIGHT,
        include_ndvi=True,
    )


def build_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """Reflectance H5 + reflectance norms, additionally deriving ndvi."""
    return apply_landsat_reflectance(
        build_extra_decode_dataset_config(common, EXTRA_DECODE_MODALITIES)
    )


def build_dataloader_config(common: CommonComponents):
    """ndvi-aware newsampling dataloader at uniform patch sizes."""
    return apply_uniform_patch_sizes(
        apply_new_sampling(
            build_extra_decode_dataloader_config(common, EXTRA_DECODE_MODALITIES)
        )
    )


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """ndvi-aware 1fwd + fused AdamW train module at the newsampling microbatch."""
    return apply_microbatch(
        build_extra_decode_train_module_config(common, EXTRA_DECODE_MODALITIES)
    )


def build_trainer_config(common: CommonComponents):
    """New-maps base trainer, with in-loop evals routed through Beaker jobs."""
    return route_loop_evals_to_beaker(_base_build_trainer_config(common), MODULE_PATH)


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

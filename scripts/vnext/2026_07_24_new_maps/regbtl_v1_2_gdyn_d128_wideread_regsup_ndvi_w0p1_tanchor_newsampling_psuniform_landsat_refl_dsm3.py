"""cand_ndvi on the new maps + Landsat reflectance, with the FULL GLO30 DSM target.

Exact twin of ``regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_
psuniform_landsat_refl`` -- same d128 register bottleneck, same NDVI arm, same
``year_start`` anchor, same sampler, same reflectance H5 and norms -- with ONE change
to the register-grid supervision head: GLO30 is supervised on all three of its bands
instead of elevation alone.

WHAT "ALL THREE" MEANS, since it is not one target:

* elevation + slope (bands 0, 1) as a 2-channel L1 regression. Slope (deg [0, 90)) is
  a well-behaved continuous target, so it needs nothing special -- the existing head
  simply threw the band away.
* aspect as a SEPARATE 2-channel L1 regression on the derived ``glo30_aspect``
  modality, which carries ``[sin(theta), cos(theta)]`` of the compass bearing. The
  stored aspect band cannot be regressed directly: it is circular, so 359 deg and
  1 deg are one degree apart but land at opposite ends of the range (measured:
  ``|norm(359) - norm(1)|`` = 0.827 of the [0, 1] range, with 4.5% of pixels within
  10 deg of the seam), and its -1 "flat" sentinel z-scores to 0.1078 versus 0.1102
  for due north -- indistinguishable, over ~5.5% of pixels. sin/cos is bounded and
  continuous across north, and the derived modality writes flat pixels out as
  MISSING_VALUE so the supervision valid mask drops them. This is the pretraining-side
  twin of the eval probes' GLO30_LABEL_ASPECT_SIN/_COS.

CAVEAT WORTH REMEMBERING WHEN READING THE RESULT: GLO-30 is a *surface* model, not a
terrain model -- elevation includes canopy and buildings (which is why it partly
overlaps the meta_canopy_height target). Slope and aspect derived from a DSM are
canopy-edge and rooftop derivatives in forested and urban areas, not landform. If this
arm helps, it is worth checking whether it helps on terrain-poor scenes too.

READ IT AGAINST: the elevation-only twin above (isolates the two extra DSM targets at
fixed everything-else), and ``..._w0p1_newsampling_psuniform_glo30_elev_slope`` (the
DN-h5 d768 arm that added slope without aspect).

Run name: ``regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_psuniform_newmaps_
refl_dsm3`` (83 chars, inside the in-loop eval callback's 94-char budget). ``dsm3`` =
all three DSM bands supervised.
"""

import logging

from base import build_common_components, build_visualize_config
from base import build_trainer_config as _base_build_trainer_config
from perceiver_common import (
    GLO30_ELEV_SLOPE_BAND_INDICES,
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
# Both extras are DERIVED in the dataset from raw un-normalized bands and ride along
# as decode-only modalities: ndvi from S2 L2A B04/B08, glo30_aspect from glo30's
# aspect band as [sin, cos].
EXTRA_DECODE_MODALITIES = [Modality.NDVI.name, Modality.GLO30_ASPECT.name]
MODULE_PATH = (
    "scripts/vnext/2026_07_24_new_maps/"
    "regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform"
    "_landsat_refl_dsm3.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d128 wideread + regsup incl. NDVI and the full DSM target, anchored read."""
    config = build_wideread_regbtl_model_config(
        common, latent_self_attn=True, register_dim=REGISTER_DIM
    )
    config.encoder_config.register_temporal_anchor = REGISTER_TEMPORAL_ANCHOR
    return add_register_supervision(
        config,
        base_weight=SUPERVISION_BASE_WEIGHT,
        include_ndvi=True,
        glo30_bands=GLO30_ELEV_SLOPE_BAND_INDICES,
        include_glo30_aspect=True,
    )


def build_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """Reflectance H5 + norms, deriving both ndvi and the aspect sin/cos target."""
    return apply_landsat_reflectance(
        build_extra_decode_dataset_config(common, EXTRA_DECODE_MODALITIES)
    )


def build_dataloader_config(common: CommonComponents):
    """Extra-decode-aware newsampling dataloader at uniform patch sizes."""
    return apply_uniform_patch_sizes(
        apply_new_sampling(
            build_extra_decode_dataloader_config(common, EXTRA_DECODE_MODALITIES)
        )
    )


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """Extra-decode-aware 1fwd + fused AdamW train module at the newsampling micro."""
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

"""d768 wideread + regsup w1 on LANDSAT REFLECTANCE, with the FULL GLO30 DSM target.

Exact twin of ``regbtl_v1_2_gdyn_d768_wideread_regsup_w1_newsampling_psuniform_
landsat_refl`` -- same d768 full-width Perceiver register bottleneck, same regsup
base_weight 1.0, same 1fwd + fused AdamW train module, same decorrelated newsampling
at uniform patch sizes, same reflectance h5 and reflectance-scale Landsat norm stats
-- with ONE change: GLO30 is supervised on all three of its bands instead of
elevation alone.

* elevation + slope (bands 0, 1) as a 2-channel L1 regression. Slope (deg [0, 90)) is
  a well-behaved continuous target that the elevation-only head simply discarded.
* aspect as a SEPARATE 2-channel L1 regression on the derived ``glo30_aspect``
  modality, which carries ``[sin(theta), cos(theta)]`` of the compass bearing. The
  stored aspect band cannot be regressed directly: it is circular (``|norm(359) -
  norm(1)|`` = 0.827 of the [0, 1] range for the same bearing, 4.5% of pixels within
  10 deg of the seam) and its -1 "flat" sentinel z-scores to 0.1078 against 0.1102 for
  due north over ~5.5% of pixels. The derived modality writes flat pixels out as
  MISSING_VALUE so the supervision valid mask drops them.

WHY THIS RUN EXISTS: it is the TEACHER-ONLY mirror of
``regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform_landsat_refl_dsm3``
(the distilled arm, d768 teacher + detached linear [128, 64] student). Without it the
distilled arm has no single-knob partner: against the elevation-only w1 run it differs
by both the DSM target and the student, and there is no way to tell which moved the
result. With it the ladder is clean --

    ..._d768_wideread_regsup_w1_..._landsat_refl        elev-only, no student
    ..._d768_wideread_regsup_w1_..._landsat_refl_dsm3   full DSM,  no student  <- HERE
    ..._d768_proj128lin_sup768_w1_..._landsat_refl_dsm3 full DSM,  + student

-- the first pair isolates the DSM target, the second isolates the student.

Same DSM-not-DTM caveat as the other _dsm3 arms: GLO-30 is a surface model, so slope
and aspect over forest and cities are canopy-edge and rooftop derivatives rather than
landform.

Run name: ``regbtl_v1_2_gdyn_d768_wideread_regsup_w1_psuniform_newmaps_refl_dsm3``
(70 chars, inside the in-loop eval callback's 94-char budget). ``dsm3`` = all three
DSM bands supervised.
"""

import logging

from base import build_common_components, build_visualize_config
from base import build_trainer_config as _base_build_trainer_config
from perceiver_common import (
    GLO30_ELEV_SLOPE_BAND_INDICES,
    SUPERVISION_BASE_WEIGHT_W1,
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

REGISTER_DIM = 768
# The aspect sin/cos target is derived in the dataset from the raw glo30 aspect band.
EXTRA_DECODE_MODALITIES = [Modality.GLO30_ASPECT.name]
MODULE_PATH = (
    "scripts/vnext/2026_07_24_new_maps/"
    "regbtl_v1_2_gdyn_d768_wideread_regsup_w1_newsampling_psuniform"
    "_landsat_refl_dsm3.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d768 wideread + new-maps regsup at w1, GLO30 supervised on all three bands."""
    config = build_wideread_regbtl_model_config(
        common, latent_self_attn=True, register_dim=REGISTER_DIM
    )
    return add_register_supervision(
        config,
        base_weight=SUPERVISION_BASE_WEIGHT_W1,
        glo30_bands=GLO30_ELEV_SLOPE_BAND_INDICES,
        include_glo30_aspect=True,
    )


def build_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """Reflectance H5 + norms, deriving the glo30 aspect sin/cos target."""
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

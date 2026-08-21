"""d128 wideread + regsup w0p1, reflectance, full DSM target, CLOUD SKIP at 0.5.

Exact twin of ``regbtl_v1_2_gdyn_d128_wideread_regsup_w0p1_newsampling_psuniform_
landsat_refl_dsm3`` -- same d128 Perceiver register bottleneck reading at encoder width,
same regsup base_weight 0.1, same 1fwd + fused AdamW train module, same decorrelated
newsampling at uniform patch sizes, same reflectance h5 and reflectance-scale Landsat
norm stats, same full GLO30 DSM target -- with ONE change: the cloud skip is on at
threshold 0.5.

The DSM target is unchanged from the twin: elevation + slope (glo30 bands 0, 1) as a
2-channel L1 regression, plus aspect as a separate 2-channel L1 regression on the
derived ``glo30_aspect`` modality (sin/cos of the bearing, flat pixels written out as
MISSING_VALUE).

CLOUD SKIP: decoder tokens whose OmniCloudMask cloud/shadow fraction exceeds 0.5 are
reassigned to MISSING and dropped from the patch-discrimination loss, so the model is
not asked to discriminate patches whose content is weather. Two knobs turn it on and
both are set here: the dataset is pointed at the precomputed sidecars, and the
threshold is recorded on BOTH masking configs (the dataloader's copy is the operative
one -- masking runs in the collate fn -- but the train module keeps its own, and
setting only one leaves the saved config disagreeing with what ran).

CACHE PREREQUISITE: the sidecars must already exist for THIS h5 set. An absent cache
is SILENT -- load_sample_clouds returns None per sample and the side-payload is just
skipped, so the run trains with no cloud skip and reports nothing wrong (the
_l8pixmask failure mode). apply_cloud_cache therefore hard-fails when the configured
directory is missing or holds no shards, whenever weka is mounted to check.
CLOUD_CACHE_DIR = None derives the sibling of the reflectance h5; set it explicitly to
reuse a cache computed for another h5 built from the same sample set -- the sidecar key
is the raw h5 sample id, so those are index-compatible.

READ IT AGAINST ``..._d128_wideread_regsup_w0p1_..._landsat_refl_dsm3`` (already
training): the cloud skip is the ONLY difference, so this pair isolates it at the
width we ship. Its d768 counterpart is
``..._d768_proj128lin_sup768_w1_..._landsat_refl_dsm3_cloudmask0p5``.

Same DSM-not-DTM caveat as the other _dsm3 arms: GLO-30 is a surface model, so slope
and aspect over forest and cities are canopy-edge and rooftop derivatives rather than
landform.

Run name:
``regbtl_v1_2_gdyn_d128_wideread_regsup_w0p1_psuniform_newmaps_refl_dsm3_cloudmask0p5``
(83 chars, inside the in-loop eval callback's 94-char budget). ``dsm3`` = all three DSM
bands supervised.
"""

import logging

from base import build_common_components, build_visualize_config
from base import build_trainer_config as _base_build_trainer_config
from perceiver_common import (
    GLO30_ELEV_SLOPE_BAND_INDICES,
    SUPERVISION_BASE_WEIGHT,
    add_register_supervision,
    apply_cloud_cache,
    apply_cloud_skip_threshold,
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
# The aspect sin/cos target is derived in the dataset from the raw glo30 aspect band.
EXTRA_DECODE_MODALITIES = [Modality.GLO30_ASPECT.name]
# Above this per-token cloud/shadow fraction a DECODER token is reassigned to MISSING.
CLOUD_SKIP_THRESHOLD = 0.5
# None => sibling of the reflectance h5. See the cache prerequisite above.
CLOUD_CACHE_DIR = None
MODULE_PATH = (
    "scripts/vnext/2026_07_24_new_maps/"
    "regbtl_v1_2_gdyn_d128_wideread_regsup_w0p1_newsampling_psuniform"
    "_landsat_refl_dsm3_cloudmask0p5.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d128 wideread + new-maps regsup at w0p1, GLO30 supervised on all three bands."""
    config = build_wideread_regbtl_model_config(
        common, latent_self_attn=True, register_dim=REGISTER_DIM
    )
    return add_register_supervision(
        config,
        base_weight=SUPERVISION_BASE_WEIGHT,
        glo30_bands=GLO30_ELEV_SLOPE_BAND_INDICES,
        include_glo30_aspect=True,
    )


def build_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """Reflectance H5 + norms + cloud sidecars, deriving the aspect sin/cos target."""
    return apply_cloud_cache(
        apply_landsat_reflectance(
            build_extra_decode_dataset_config(common, EXTRA_DECODE_MODALITIES)
        ),
        CLOUD_CACHE_DIR,
    )


def build_dataloader_config(common: CommonComponents):
    """Extra-decode-aware newsampling dataloader, uniform patch sizes, cloud skip."""
    return apply_cloud_skip_threshold(
        apply_uniform_patch_sizes(
            apply_new_sampling(
                build_extra_decode_dataloader_config(common, EXTRA_DECODE_MODALITIES)
            )
        ),
        CLOUD_SKIP_THRESHOLD,
    )


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """Extra-decode-aware 1fwd + fused AdamW train module, cloud skip on its copy."""
    return apply_cloud_skip_threshold(
        apply_microbatch(
            build_extra_decode_train_module_config(common, EXTRA_DECODE_MODALITIES)
        ),
        CLOUD_SKIP_THRESHOLD,
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

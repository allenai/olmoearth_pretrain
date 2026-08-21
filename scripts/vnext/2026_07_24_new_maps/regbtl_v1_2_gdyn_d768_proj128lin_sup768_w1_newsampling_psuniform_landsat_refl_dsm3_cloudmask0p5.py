"""d768 teacher + detached linear [128, 64] student: new maps, reflectance, full DSM, CLOUD SKIP 0.5.

``scripts/official/v1_2/regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_
psuniform.py`` rebuilt on the new-maps base, with the full GLO30 DSM target. The
student wiring is unchanged from the official run:

* student architecture: **per-cell Linear(768, 128) on the detached registers**, with
  the first 64 dims a self-sufficient Matryoshka prefix, so one artifact ships both
  widths. Its input is detached, so the encoder and the primary bottleneck train
  exactly as the plain regsup_w1 arm does.
* supervision heads: **d768 registers only** (``supervision_source="registers"``,
  i.e. sup768). The student is trained by the distillation terms alone.
* regsup at ``base_weight=1.0`` (w1) -- the weight the proj program settled on,
  since the w1 teacher's ceiling is what the student inherits.

Three things move relative to the official run:

* maps: ``srtm`` -> ``glo30`` and ``wri_canopy_height_map`` -> ``meta_canopy_height``.
* radiometry: Landsat as TOA reflectance / brightness temperature with the matching
  reflectance-scale norm stats.
* the GLO30 target: elevation + slope (bands 0, 1) as a 2-channel L1 regression, plus
  aspect as a separate 2-channel L1 regression on the derived ``glo30_aspect``
  modality (``[sin, cos]`` of the bearing, flat pixels written out as MISSING_VALUE).
  Raw aspect degrees cannot be regressed directly -- circular target, and the -1 flat
  sentinel z-scores to almost exactly due north. See the d128 ``_dsm3`` sibling for
  the measured numbers and the DSM-not-DTM caveat.

ONE change from ``..._d768_proj128lin_sup768_w1_..._landsat_refl_dsm3`` (already
training): the cloud skip is on at threshold 0.5. Decoder tokens whose OmniCloudMask
cloud/shadow fraction exceeds 0.5 are reassigned to MISSING and dropped from the
patch-discrimination loss, so the model is not asked to discriminate patches whose
content is weather. The threshold is set on BOTH masking configs -- the dataloader's
copy is the operative one (masking runs in the collate fn), but the train module keeps
its own, and setting only one leaves the saved config disagreeing with what ran.

CACHE PREREQUISITE: the sidecars must already exist for THIS h5 set. An absent cache is
SILENT -- load_sample_clouds returns None per sample and the payload is skipped, so the
run trains with no cloud skip and reports nothing wrong (the _l8pixmask failure mode).
apply_cloud_cache hard-fails on a missing or shard-less directory whenever weka is
mounted to check. CLOUD_CACHE_DIR = None derives the sibling of the reflectance h5; set
it explicitly to reuse a cache computed for another h5 over the same sample set, which
is index-compatible because the sidecar key is the raw h5 sample id.

No NDVI arm and no temporal anchor here: this run's point is the student, and its
comparison partners on the proj axis do not have them.

WHY THIS ONE MATTERS: it ships 128 dims. The in-flight d768 new-maps reflectance arms
are diagnostic only; this is the shippable-width member of that family, and its
in-loop evals score the checkpoint at 768 / 128 / 64 every 40k steps.

Run name:
``regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_psuniform_newmaps_refl_dsm3_cloudmask0p5``
(83 chars, inside the in-loop eval callback's 94-char budget).
"""

import logging

from base import build_common_components, build_visualize_config
from base import build_trainer_config as _base_build_trainer_config
from perceiver_common import (
    GLO30_ELEV_SLOPE_BAND_INDICES,
    SUPERVISION_BASE_WEIGHT_W1,
    apply_cloud_cache,
    apply_cloud_skip_threshold,
    apply_landsat_reflectance,
    apply_microbatch,
    apply_new_sampling,
    apply_uniform_patch_sizes,
    build_extra_decode_dataloader_config,
    build_extra_decode_dataset_config,
    build_extra_decode_train_module_config,
    build_proj_model_config,
    route_proj_loop_evals_to_beaker,
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
# 40k, not the 20k default: this run's 8-task eval jobs take longer than 20k training
# steps, so consecutive jobs would overlap on one resumed W&B run and the overlapping
# writer's rows get silently dropped.
EMBEDDING_EVAL_INTERVAL_STEPS = 40000
# Above this per-token cloud/shadow fraction a DECODER token is reassigned to MISSING.
CLOUD_SKIP_THRESHOLD = 0.5
# None => sibling of the reflectance h5. See the cache prerequisite above.
CLOUD_CACHE_DIR = None
MODULE_PATH = (
    "scripts/vnext/2026_07_24_new_maps/"
    "regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform"
    "_landsat_refl_dsm3_cloudmask0p5.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d768 teacher + detached linear student; supervision on the d768 registers."""
    return build_proj_model_config(
        common,
        projection_type="linear",
        supervision_source="registers",
        register_dim=REGISTER_DIM,
        base_weight=SUPERVISION_BASE_WEIGHT_W1,
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
    """New-maps base trainer + in-loop evals on BOTH the d768 and projected heads."""
    return route_proj_loop_evals_to_beaker(
        _base_build_trainer_config(common),
        MODULE_PATH,
        embedding_eval_interval_steps=EMBEDDING_EVAL_INTERVAL_STEPS,
    )


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

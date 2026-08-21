"""d768 teacher + lin supstu0p1 student + NDVI + cloud skip 0.5 + student uniformity.

``regbtl_v1_2_gdyn_d768_proj128lin_supstu0p1_ndvi_w1_newsampling_psuniform``
(see its docstring) with the cloud-aware patch-discrimination skip at the 0.5
threshold: decoder tokens over 50% cloud/shadow are dropped from the patch-disc
loss, covering S2, Landsat and the S2-derived ndvi decode target. NOTE launched
2026-08-20 while the cand_ndvi threshold sweep reads null-to-slightly-negative
at mid-training (trial metrics below cand_ndvi at all thresholds, cache
completeness verified) -- included at Gabi's request; if the threshold sweep
confirms the null at convergence, read this arm against the +ndvi supstu arm as
a near-replicate.

One of four supstu arms launched 2026-08-20 (plain / +ndvi /
+ndvi+cloudmask0p5 / +ndvi+cloudmask0p5+stuunif0p1).

In-loop evals: the year-aligned early-read set on both heads (student tasks
first), including aeftrial metrics -- ``set_proj_earlyread_loop_evals``.
"""

import logging

from base import build_trainer_config as _base_build_trainer_config
from regbtl_v1_2_cloudmask_common import apply_cloud_cache, apply_cloud_skip_threshold
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_common_components,
    build_visualize_config,
)
from regbtl_v1_2_newsampling_common import (
    apply_microbatch,
    apply_new_sampling,
    apply_uniform_patch_sizes,
)
from regbtl_v1_2_proj_common import (
    SUPERVISION_BASE_WEIGHT_W1,
    build_proj_model_config,
    set_proj_earlyread_loop_evals,
)
from regbtl_v1_2_regsup_common import (
    build_extra_decode_dataloader_config,
    build_extra_decode_dataset_config,
    build_extra_decode_train_module_config,
)

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

EXTRA_DECODE_MODALITIES = [Modality.NDVI.name]
PROJECTION_SUPERVISION_SCALE = 0.1
# Fraction of a token's pixels that must be cloud/shadow before it is dropped as
# a decode target. Set on both masking configs by apply_cloud_skip_threshold.
CLOUD_SKIP_THRESHOLD = 0.5
# Matches the stuunif0p1 arm: student-only spread, an order of magnitude below
# the supervision weight.
PROJECTION_UNIFORMITY_WEIGHT = 0.1
MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_gdyn_d768_proj128lin_supstu0p1_ndvi_w1_newsampling_psuniform_cloudmask0p5_stuunif0p1.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Supervision incl. NDVI on both widths, student heads at 0.1x."""
    return build_proj_model_config(
        common,
        base_weight=SUPERVISION_BASE_WEIGHT_W1,
        projection_type="linear",
        supervision_source="both",
        projection_supervision_weight_scale=PROJECTION_SUPERVISION_SCALE,
        include_ndvi=True,
    )


def build_dataset_config(common: CommonComponents):
    """The ndvi arm's dataset, additionally pointed at the cloud-mask sidecars."""
    return apply_cloud_cache(
        build_extra_decode_dataset_config(common, EXTRA_DECODE_MODALITIES)
    )


def build_dataloader_config(common: CommonComponents):
    """ndvi-aware newsampling dataloader at uniform patch sizes, cloud-skipping."""
    return apply_cloud_skip_threshold(
        apply_uniform_patch_sizes(
            apply_new_sampling(
                build_extra_decode_dataloader_config(common, EXTRA_DECODE_MODALITIES)
            )
        ),
        CLOUD_SKIP_THRESHOLD,
    )


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """ndvi-aware cloud-skipping train module + student-only uniformity."""
    config = apply_cloud_skip_threshold(
        apply_microbatch(
            build_extra_decode_train_module_config(common, EXTRA_DECODE_MODALITIES)
        ),
        CLOUD_SKIP_THRESHOLD,
    )
    config.projection_uniformity_weight = PROJECTION_UNIFORMITY_WEIGHT
    return config


def build_trainer_config(common: CommonComponents):
    """Base trainer + the year-aligned early-read evals on both heads."""
    return set_proj_earlyread_loop_evals(
        _base_build_trainer_config(common), MODULE_PATH
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

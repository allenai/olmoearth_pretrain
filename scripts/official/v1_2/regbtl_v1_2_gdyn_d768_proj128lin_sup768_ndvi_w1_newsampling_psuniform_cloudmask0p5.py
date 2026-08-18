"""d768 teacher + lin student, sup768 w1 + NDVI + cloud skip at the 0.5 threshold.

``regbtl_v1_2_gdyn_d768_proj128lin_sup768_ndvi_w1_newsampling_psuniform`` (the NDVI
arm of the 2026-08-18 distillation 2x2) with exactly one thing added: DECODER tokens
whose OmniCloudMask cloud/shadow fraction exceeds ``CLOUD_SKIP_THRESHOLD`` are
reassigned to MISSING and dropped from the patch-discrimination loss. The skip
covers S2, Landsat AND the S2-derived ndvi decode target (``CLOUD_SKIP_MODALITIES``
in ``train/masking.py``).

WHY, and why launched before the threshold sweep reads out: this pulls the
cloudmask axis onto the distillation lineage -- the pretext task otherwise asks the
model to predict weather, and the NDVI head sharpens that failure (a vegetation
index through cloud is noise), so NDVI x cloudmask is the mechanism-predicted
pairing rather than an axis to decompose. 0.5 is the middle threshold -- the arm the
cand_ndvi cloudmask sweep's own launch script calls "the one that decides whether
the effect exists at all". If that sweep later picks 0.25/0.75, this run becomes
the wrong-threshold datapoint rather than the candidate; launched anyway on free
GPUs (2026-08-18). A/B partner: ``..._proj128lin_sup768_ndvi_w1_newsampling_
psuniform`` (identical, no cloud skip), launched the same day with the same eval
set, so the comparison reads directly off matched in-loop curves.

Cloud maps come from the precomputed sidecar beside the training h5 set
(``data/cloud_mask_cache``). If a sample is uncached the batch silently trains
without cloud masking, so verify the cache is COMPLETE (1,138,828 .npz) before
reading anything into a null result -- see launch_regbtl_v1_2_cloudmask.sh.

In-loop evals: the early-read year-aligned set on BOTH heads (teacher + proj128
student), including the aeftrial_* metrics -- ``set_proj_earlyread_loop_evals``.
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
# Fraction of a token's pixels that must be cloud/shadow before it is dropped as a
# decode target. Set on both masking configs by apply_cloud_skip_threshold.
CLOUD_SKIP_THRESHOLD = 0.5
MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_gdyn_d768_proj128lin_sup768_ndvi_w1_newsampling_psuniform_cloudmask0p5.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d768 teacher + detached linear student, register supervision incl. NDVI."""
    return build_proj_model_config(
        common,
        base_weight=SUPERVISION_BASE_WEIGHT_W1,
        projection_type="linear",
        supervision_source="registers",
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
    """ndvi-aware 1fwd + fused AdamW train module, cloud-skipping."""
    return apply_cloud_skip_threshold(
        apply_microbatch(
            build_extra_decode_train_module_config(common, EXTRA_DECODE_MODALITIES)
        ),
        CLOUD_SKIP_THRESHOLD,
    )


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

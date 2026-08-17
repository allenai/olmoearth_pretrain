"""cand_ndvi + cloud-aware patch discrimination, drop tokens over 75% cloud.

``regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform``
(cand_ndvi, the release-readiness candidate) with exactly one thing added: DECODER
tokens whose OmniCloudMask cloud/shadow fraction exceeds ``CLOUD_SKIP_THRESHOLD`` are
reassigned to MISSING and dropped from the patch-discrimination loss. Model,
sampler, anchor, NDVI arm, token budget, LR schedule, epochs and loss set are untouched,
so this is a single-variable A/B against cand_ndvi itself. Note that cand_ndvi builds a
``LatentMIMTrainModuleConfig``, which carries no instance-contrastive (InfoNCE) term at
all -- Yawen's original cloud run set ``contrastive_config=None`` on a Contrastive train
module, confounding two changes at once, but here there is simply nothing to disable.

WHY: the pretext task currently asks the model to predict the latent content of cloudy
pixels, which is unpredictable by construction -- the target is weather, not ground. The
NDVI supervision head makes this sharper: a vegetation index read through cloud is noise,
and forcing per-cell temporal trajectories to fit it should actively corrupt the register
grid. The skip covers S2, Landsat AND the S2-derived ndvi decode target (see
``CLOUD_SKIP_MODALITIES`` in ``train/masking.py``); ndvi borrows the S2 cloud map because
it is computed from B04/B08 on S2's exact post-crop grid.

Swept arm: 0.25 / 0.5 / 0.75. This is the CONSERVATIVE end -- only tokens that are
three-quarters cloud are dropped, so it keeps the most decode targets and is the closest
of the three to cand_ndvi itself. If cloud masking helps at all it should show here
weakest; if it helps MOST here, the effect is target starvation rather than cloud noise.
Note the sampler draws ps=1..8 uniformly and at ps=1 the per-token fraction is 0 or 1,
so the threshold only discriminates at ps>=2.

Cloud maps come from the precomputed sidecar beside the training h5 set
(``data/cloud_mask_cache``). If a sample is uncached the batch silently trains without
cloud masking, so verify the cache is COMPLETE before reading anything into a null
result.
"""

import logging

from base import build_trainer_config as _base_build_trainer_config
from regbtl_v1_2_cloudmask_common import apply_cloud_cache, apply_cloud_skip_threshold
from regbtl_v1_2_earlyread_common import set_earlyread_loop_evals
from regbtl_v1_2_faster_common import build_wideread_regbtl_model_config
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_common_components,
    build_visualize_config,
)
from regbtl_v1_2_newsampling_common import (
    SUPERVISION_BASE_WEIGHT,
    apply_microbatch,
    apply_new_sampling,
    apply_uniform_patch_sizes,
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
# Fraction of a token's pixels that must be cloud/shadow before it is dropped as a
# decode target. Set on both masking configs by apply_cloud_skip_threshold.
CLOUD_SKIP_THRESHOLD = 0.75
MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform_cloudmask0p75.py"
)


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
    """cand_ndvi's dataset, additionally pointed at the cloud-mask sidecars."""
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


def build_train_module_config(common: CommonComponents):
    """ndvi-aware faster train module with a halved rank microbatch size."""
    return apply_cloud_skip_threshold(
        apply_microbatch(
            build_extra_decode_train_module_config(common, EXTRA_DECODE_MODALITIES)
        ),
        CLOUD_SKIP_THRESHOLD,
    )


def build_trainer_config(common: CommonComponents):
    """Early-read eval set (the 6 S1+S2+Landsat probes), routed through a Beaker job.

    REPLACES the shared catalog rather than merging into it, matching the eval set the
    current runs are judged on. Set here rather than via a
    ``downstream_evaluator.tasks_to_run`` CLI override because that flag only FILTERS the
    existing task dict -- the five year-aligned names are not in the catalog
    ``add_loop_eval_beaker_job`` builds, so overriding there would silently run just the
    one pastis bridge task.
    """
    return set_earlyread_loop_evals(_base_build_trainer_config(common), MODULE_PATH)


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

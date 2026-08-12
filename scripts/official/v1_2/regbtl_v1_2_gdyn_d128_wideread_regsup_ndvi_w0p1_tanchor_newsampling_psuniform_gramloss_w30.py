"""d128 wideread + regsup + NDVI (psuniform, tanchor), Gram-matching pretext loss at WEIGHT 30.

``regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform`` with
the pretext loss REPLACED: instead of the patch-discrimination InfoNCE
(``modality_patch_discrimination_masked_negatives_vec``: each decoded token classifies
its own frozen-random-projection target against the sample's other targets, tau=0.1,
near-duplicate negatives masked), the predictions must reproduce the targets'
RELATIONAL GEOMETRY -- per sample and modality, the cosine-Gram matrix of the
DECODER-masked predicted tokens is matched to the targets' cosine-Gram matrix by MSE
over off-diagonal pairs (``modality_gram_matching_vec``; the distillation-Gram / RKD
formulation of the proj128 student work, applied to the pretext task).

Everything else -- sampler, anchor, NDVI arm, regsup weight, frozen
projection-only targets -- is unchanged, so the A/B partner is the psuniform ndvi
tanchor run itself. No temperature and no same-target masking: duplicate targets are
high-similarity Gram entries to reproduce, not degenerate InfoNCE negatives.

WHY w30: the weight-1.0 run confirmed the scale risk -- by step 75k the gram term
had shrunk to ~0.0055 vs ~0.039 weighted supervision (7x smaller; at init 0.12 vs
0.26), i.e. the model trained as a supervision-only model, and its evals ran ~0.09-
0.11 mIoU behind baseline at 20k-60k. weight=30 restores the pretext to ~3.6x
supervision at init values (~4x at the 75k values), making it the dominant signal
again; the ~4-nat InfoNCE it replaced was ~16x. If this still trails with the
ratio collapsing late, the next rung is ~100x. A/B partners: the baseline run and
the w1 gramloss run (stopped at ~80k).
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
from olmoearth_pretrain.train.loss import LossConfig

logger = logging.getLogger(__name__)

REGISTER_DIM = 128
REGISTER_TEMPORAL_ANCHOR = "year_start"
EXTRA_DECODE_MODALITIES = [Modality.NDVI.name]
MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform"
    "_gramloss_w30.py"
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
    """Base dataset config, additionally deriving ndvi from the raw S2 bands."""
    return build_extra_decode_dataset_config(common, EXTRA_DECODE_MODALITIES)


def build_dataloader_config(common: CommonComponents):
    """ndvi-aware newsampling dataloader at uniform patch sizes."""
    return apply_uniform_patch_sizes(
        apply_new_sampling(
            build_extra_decode_dataloader_config(common, EXTRA_DECODE_MODALITIES)
        )
    )


def build_train_module_config(common: CommonComponents):
    """ndvi-aware faster train module with the Gram-matching pretext loss."""
    config = apply_microbatch(
        build_extra_decode_train_module_config(common, EXTRA_DECODE_MODALITIES)
    )
    # Drop-in pretext-loss replacement: relational (cosine-Gram MSE) matching of
    # the frozen random-projection targets instead of patch-discrimination InfoNCE.
    config.loss_config = LossConfig(
        loss_config={"type": "modality_gram_matching_vec", "weight": 30.0}
    )
    return config


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

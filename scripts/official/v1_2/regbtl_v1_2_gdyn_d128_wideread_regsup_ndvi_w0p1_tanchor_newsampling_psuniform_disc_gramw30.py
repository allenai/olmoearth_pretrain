"""d128 wideread + regsup + NDVI (psuniform, tanchor): InfoNCE + Gram(w30) TOGETHER.

The BOTH-LOSSES arm: the baseline patch-discrimination InfoNCE
(``modality_patch_discrimination_masked_negatives_vec``, tau 0.1, same-target
masking -- byte-identical config to the baseline's) PLUS the within-sample
cosine-Gram MSE at weight 30 (``modality_gram_matching_vec``; the scale the w30
rerun chose so the gram term is ~4x supervision, comparable to InfoNCE's ~16x).
Combined via the ``weighted_sum`` composite loss.

Completes the triangle with the baseline (InfoNCE only) and ``_gramloss_w30``
(gram only): if gram-only underperforms but this arm beats the baseline, the
relational term ADDS signal InfoNCE lacks; if this arm matches the baseline, the
gram term is redundant with discrimination; if it hurts, the two objectives
conflict. Everything else is unchanged from the baseline arm.
"""

import logging

from base import ONLY_DECODE_MODALITIES
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
    "_disc_gramw30.py"
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
    """ndvi-aware faster train module with InfoNCE + Gram(w30) pretext losses."""
    config = apply_microbatch(
        build_extra_decode_train_module_config(common, EXTRA_DECODE_MODALITIES)
    )
    # Drop-in pretext-loss replacement: relational (cosine-Gram MSE) matching of
    # the frozen random-projection targets instead of patch-discrimination InfoNCE.
    # Both pretext losses: the baseline InfoNCE (identical config) + gram at 30.
    config.loss_config = LossConfig(
        loss_config={
            "type": "weighted_sum",
            "losses": [
                {
                    "type": "modality_patch_discrimination_masked_negatives_vec",
                    "tau": 0.1,
                    "same_target_threshold": 0.999,
                    "mask_negatives_for_modalities": ONLY_DECODE_MODALITIES,
                },
                {"type": "modality_gram_matching_vec", "weight": 30.0},
            ],
        }
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

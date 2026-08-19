"""cand_ndvi evaluated on an EMA of the encoder instead of the live weights.

``regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform``
with exactly one thing added: a frozen EMA copy of the online encoder
(``LatentMIMConfig.keep_encoder_ema``) that the downstream evals read instead of
the online weights. Model, sampler, anchor, NDVI arm, token budget, LR schedule,
epochs and loss set are untouched, so training is a repeat of cand_ndvi and the
only question is whether the EMA weights probe better than the live ones.

This is NOT the target encoder. These runs keep the target at its frozen random
init (``projection_only_target=True``, ``ema_decay=(1.0, 1.0)``) to supply
pretext targets; turning target EMA on would change the pretext task. The eval
EMA is a separate ``ema_encoder`` module, updated after each optimizer step and
never touched by the loss.

Decay is a CONSTANT 0.9999 (MAE-style eval EMA), not the (0.996, 1.0) ramp the
target-EMA schedule uses: ramping to 1.0 would freeze the average late in the
run, so the final evals would score a stale mid-run encoder rather than a smooth
recent one. Override via ``--train_module.encoder_ema_decay`` to sweep.

The eval-side comparison is A/B against cand_ndvi's own in-loop eval curves
(same eval set via ``add_loop_eval_beaker_job``); this run logs to its own W&B
project so the arms stay separate.
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

logger = logging.getLogger(__name__)

REGISTER_DIM = 128
REGISTER_TEMPORAL_ANCHOR = "year_start"
EXTRA_DECODE_MODALITIES = [Modality.NDVI.name]
# Constant decay for the eval-only encoder EMA; see the module docstring for why
# this is flat rather than the target-EMA ramp.
ENCODER_EMA_DECAY = (0.9999, 0.9999)
WANDB_PROJECT = "2026_08_19_ema"
MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform_ema.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """cand_ndvi's model plus the frozen eval-only encoder EMA copy."""
    config = build_wideread_regbtl_model_config(
        common, latent_self_attn=True, register_dim=REGISTER_DIM
    )
    config.encoder_config.register_temporal_anchor = REGISTER_TEMPORAL_ANCHOR
    config.keep_encoder_ema = True
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
    """cand_ndvi's train module plus the eval-only encoder EMA update."""
    config = apply_microbatch(
        build_extra_decode_train_module_config(common, EXTRA_DECODE_MODALITIES)
    )
    config.encoder_ema_decay = ENCODER_EMA_DECAY
    return config


def build_trainer_config(common: CommonComponents):
    """cand_ndvi's eval set via a Beaker job, logging to the EMA W&B project."""
    trainer_config = add_loop_eval_beaker_job(
        _base_build_trainer_config(common), MODULE_PATH
    )
    trainer_config.callbacks["wandb"].project = WANDB_PROJECT
    return trainer_config


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

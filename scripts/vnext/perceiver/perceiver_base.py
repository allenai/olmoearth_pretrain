"""Perceiver latent-bottleneck encoder on the v1.2 baseline (base size).

Identical to the official v1.2 base recipe (single bandsets, nonlinear
patch-embed projection, updated masking/loss, mixed 3D RoPE) except that the
encoder is a ``PerceiverEncoder`` (anchored latents + cross-attention reads +
dense read-out) and the decoder is the attention-free per-location
``PerceiverPredictor`` head. See ``olmoearth_pretrain/nn/perceiver.py``.

Everything else — data, masking, losses, optimizer, schedule, token budget —
is inherited verbatim from the local ``base.py`` (a copy of
``scripts/official/v1_2/base.py``) for clean attribution of the encoder
change.
"""

import logging

from base import (
    BAND_DROPOUT_MODALITIES,
    MAX_PATCH_SIZE,
    MIN_PATCH_SIZE,
    PATCH_EMBED_HIDDEN_SIZES,
    RANDOM_BAND_DROPOUT_MAX_RATE,
    ROPE_MIXED_BASE,
    ROPE_TEMPORAL_COORDINATE_SCALE,
    SPATIAL_POS_ENCODING,
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_train_module_config,
    build_visualize_config,
)
from base import (
    build_trainer_config as build_trainer_config_base,
)
from olmo_core.train.config import TrainerConfig

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.internal.utils import MODEL_SIZE_ARGS
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.nn.perceiver import (
    PerceiverEncoderConfig,
    PerceiverPredictorConfig,
)

logger = logging.getLogger(__name__)

WANDB_PROJECT = "2026_07_13_perceiver"

# Perceiver knobs (v0 defaults from the design discussion).
LATENT_STRIDE_HW = 2
LATENT_STRIDE_T = 2
NUM_READS = 2
READOUT_DEPTH = 2
HEAD_DEPTH = 2


def build_size_model_config(
    common: CommonComponents,
    size_name: str,
    patch_embed_hidden_sizes: list[int],
) -> LatentMIMConfig:
    """Build a perceiver model config for the given size preset."""
    model_size = MODEL_SIZE_ARGS[size_name]

    encoder_config = PerceiverEncoderConfig(
        embedding_size=model_size["encoder_embedding_size"],
        num_heads=model_size["encoder_num_heads"],
        depth=model_size["encoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        supported_modality_names=common.training_modalities,
        max_patch_size=MAX_PATCH_SIZE,
        min_patch_size=MIN_PATCH_SIZE,
        drop_path=0.1,
        max_sequence_length=12,
        tokenization_config=common.tokenization_config,
        band_dropout_rate=RANDOM_BAND_DROPOUT_MAX_RATE,
        random_band_dropout=True,
        band_dropout_modalities=BAND_DROPOUT_MODALITIES,
        patch_embed_hidden_sizes=patch_embed_hidden_sizes,
        position_encoding=SPATIAL_POS_ENCODING,
        rope_mixed_base=ROPE_MIXED_BASE,
        rope_temporal_coordinate_scale=ROPE_TEMPORAL_COORDINATE_SCALE,
        latent_stride_hw=LATENT_STRIDE_HW,
        latent_stride_t=LATENT_STRIDE_T,
        num_reads=NUM_READS,
        readout_depth=READOUT_DEPTH,
    )
    decoder_config = PerceiverPredictorConfig(
        encoder_embedding_size=model_size["encoder_embedding_size"],
        decoder_embedding_size=model_size["decoder_embedding_size"],
        depth=0,
        head_depth=HEAD_DEPTH,
        mlp_ratio=model_size["mlp_ratio"],
        num_heads=model_size["decoder_num_heads"],
        supported_modality_names=common.training_modalities,
        max_sequence_length=12,
        tokenization_config=common.tokenization_config,
        position_encoding=SPATIAL_POS_ENCODING,
        rope_mixed_base=ROPE_MIXED_BASE,
        rope_temporal_coordinate_scale=ROPE_TEMPORAL_COORDINATE_SCALE,
    )
    return LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
    )


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the perceiver base model config."""
    return build_size_model_config(
        common, "base_shallow_decoder", PATCH_EMBED_HIDDEN_SIZES
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """v1.2 trainer config with the perceiver wandb project."""
    trainer_config = build_trainer_config_base(common)
    trainer_config.callbacks["wandb"].project = WANDB_PROJECT
    return trainer_config


if __name__ == "__main__":
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )

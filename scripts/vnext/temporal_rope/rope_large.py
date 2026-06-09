"""ViT-large v1.1 hidden-projection baseline with 2D RoPE.

Same training / data / loss / trainer setup as scripts/official/v1_1/base.py,
but swaps the encoder + decoder to ``large_shallow_decoder`` (1024 dim, 24
encoder depth, 16 heads) and replaces additive absolute spatial encodings
with attention-level axial 2D RoPE (matching scripts/vnext/temporal_rope/rope.py).
"""

import logging

from base import (
    BAND_DROPOUT_MODALITIES,
    MAX_PATCH_SIZE,
    PATCH_EMBED_HIDDEN_SIZES,
    RANDOM_BAND_DROPOUT_MAX_RATE,
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_train_module_config,
    build_trainer_config,
    build_visualize_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.internal.utils import MODEL_SIZE_ARGS
from olmoearth_pretrain.nn.flexihelios import EncoderConfig, PredictorConfig
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)

SPATIAL_POS_ENCODING = "rope"
ROPE_BASE = 10000.0
ROPE_COORDINATE_SCALE = 1.0
MODEL_SIZE_KEY = "large_shallow_decoder"


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the v1.1 large model with 2D RoPE."""
    model_size = MODEL_SIZE_ARGS[MODEL_SIZE_KEY]
    encoder_config = EncoderConfig(
        embedding_size=model_size["encoder_embedding_size"],
        num_heads=model_size["encoder_num_heads"],
        depth=model_size["encoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        supported_modality_names=common.training_modalities,
        max_patch_size=MAX_PATCH_SIZE,
        drop_path=0.1,
        max_sequence_length=12,
        tokenization_config=common.tokenization_config,
        band_dropout_rate=RANDOM_BAND_DROPOUT_MAX_RATE,
        random_band_dropout=True,
        band_dropout_modalities=BAND_DROPOUT_MODALITIES,
        patch_embed_hidden_sizes=PATCH_EMBED_HIDDEN_SIZES,
        spatial_pos_encoding=SPATIAL_POS_ENCODING,
        rope_base=ROPE_BASE,
        rope_coordinate_scale=ROPE_COORDINATE_SCALE,
    )
    decoder_config = PredictorConfig(
        encoder_embedding_size=model_size["encoder_embedding_size"],
        decoder_embedding_size=model_size["decoder_embedding_size"],
        depth=model_size["decoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        num_heads=model_size["decoder_num_heads"],
        supported_modality_names=common.training_modalities,
        max_sequence_length=12,
        tokenization_config=common.tokenization_config,
        spatial_pos_encoding=SPATIAL_POS_ENCODING,
        rope_base=ROPE_BASE,
        rope_coordinate_scale=ROPE_COORDINATE_SCALE,
    )
    return LatentMIMConfig(encoder_config=encoder_config, decoder_config=decoder_config)


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

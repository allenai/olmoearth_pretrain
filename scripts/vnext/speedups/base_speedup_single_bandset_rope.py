"""Speedup script: single bandset + random band dropout + 2D RoPE.

Baseline match:
- Single bandset S2 (all 12 bands) / Landsat (all 11 bands)
- Random band dropout (rate ~ Uniform(0, 0.3))
- Random with decode masking
- Vectorized modality patch discrimination loss

Change from base_speedup_single_bandset_band_dropout_random_decode:
- Use attention-level axial 2D RoPE instead of additive absolute spatial encodings.
"""

import logging

from base_speedup_single_bandset_band_dropout_random_decode import (
    BAND_DROPOUT_MODALITIES,
    MAX_PATCH_SIZE,
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


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the single-bandset model config with 2D RoPE."""
    model_size = MODEL_SIZE_ARGS["base_shallow_decoder"]

    rope_kwargs = {
        "spatial_pos_encoding": SPATIAL_POS_ENCODING,
        "rope_base": ROPE_BASE,
        "rope_coordinate_scale": ROPE_COORDINATE_SCALE,
    }
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
        **rope_kwargs,
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
        **rope_kwargs,
    )
    return LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
    )


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

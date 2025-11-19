"""Trying to prototype fitting everything into olmo core."""

import logging

from script import (
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_train_module_config,
    build_trainer_config,
    build_visualize_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.internal.utils import MODEL_SIZE_ARGS
from olmoearth_pretrain.nn.flexihelios import (
    EncoderConfig,
    PredictorConfig,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)

MAX_PATCH_SIZE = 8
MIN_PATCH_SIZE = 1

# Fixed predictor config based on ViT base
BASE_PREDICTOR_CONFIG = {
    "decoder_depth": 12,
    "encoder_embedding_size": 768,  # For decoder's encoder_embedding_size
    "decoder_embedding_size": 768,
    "decoder_num_heads": 12,
    "mlp_ratio": 4.0,
}

# Shape sweep options for encoder configurations
# Predictor is fixed to BASE_PREDICTOR_CONFIG for all sweeps
SHAPE_SWEEP_OPTIONS = {
    # 0.25-0.4 band
    "C1_mid_depth_narrow_MLPlean": {
        "encoder_depth": 12,
        "encoder_embedding_size": 384,
        "encoder_num_heads": 6,
        "mlp_ratio": 2.0,
    },
    "C2_deep_narrow": {
        "encoder_depth": 18,
        "encoder_embedding_size": 256,
        "encoder_num_heads": 4,
        "mlp_ratio": 3.0,
    },
    "C3_shallow_wide": {
        "encoder_depth": 6,
        "encoder_embedding_size": 640,
        "encoder_num_heads": 10,
        "mlp_ratio": 3.0,
    },
    "C4_mid_depth_highMLP": {
        "encoder_depth": 10,
        "encoder_embedding_size": 320,
        "encoder_num_heads": 5,
        "mlp_ratio": 4.0,
    },
    # 0.5-0.6 band
    "A1_mid_depth_compressed": {
        "encoder_depth": 10,
        "encoder_embedding_size": 512,
        "encoder_num_heads": 8,
        "mlp_ratio": 2.0,
    },
    "A2_deep_narrow": {
        "encoder_depth": 16,
        "encoder_embedding_size": 384,
        "encoder_num_heads": 6,
        "mlp_ratio": 3.0,
    },
    "A3_shallow_wide": {
        "encoder_depth": 6,
        "encoder_embedding_size": 768,
        "encoder_num_heads": 12,
        "mlp_ratio": 3.0,
    },
    "A4_mid_depth_MLPheavy_narrow": {
        "encoder_depth": 12,
        "encoder_embedding_size": 384,
        "encoder_num_heads": 6,
        "mlp_ratio": 4.0,
    },
    # 0.7-0.8 band
    "B1_scaled_baseline": {
        "encoder_depth": 12,
        "encoder_embedding_size": 640,
        "encoder_num_heads": 10,
        "mlp_ratio": 3.0,
    },
    "B2_deep_narrow": {
        "encoder_depth": 18,
        "encoder_embedding_size": 512,
        "encoder_num_heads": 8,
        "mlp_ratio": 3.0,
    },
    "B3_shallow_wide": {
        "encoder_depth": 8,
        "encoder_embedding_size": 960,
        "encoder_num_heads": 15,
        "mlp_ratio": 4.0,
    },
    "B4_same_depth_MLPlean": {
        "encoder_depth": 12,
        "encoder_embedding_size": 768,
        "encoder_num_heads": 12,
        "mlp_ratio": 2.0,
    },
}


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model config for an experiment."""
    model_size = MODEL_SIZE_ARGS["base_shallow_decoder"]

    encoder_config = EncoderConfig(
        embedding_size=model_size["encoder_embedding_size"],
        num_heads=model_size["encoder_num_heads"],
        depth=model_size["encoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        supported_modality_names=common.training_modalities,
        max_patch_size=MAX_PATCH_SIZE,
        drop_path=0.1,
        max_sequence_length=12,
        use_flash_attn=True,
    )
    decoder_config = PredictorConfig(
        encoder_embedding_size=model_size["encoder_embedding_size"],
        decoder_embedding_size=model_size["decoder_embedding_size"],
        depth=model_size["decoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        num_heads=model_size["decoder_num_heads"],
        supported_modality_names=common.training_modalities,
        max_sequence_length=12,
        use_flash_attn=True,
    )
    model_config = LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
    )
    return model_config


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

"""hidden1_supervision with base_super_shallow_decoder (decoder_depth=2)."""

import logging
from dataclasses import replace

from hidden1 import (
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
from hidden1_supervision import SUPERVISION_MODALITY_CONFIGS

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.internal.utils import MODEL_SIZE_ARGS
from olmoearth_pretrain.nn.flexihelios import EncoderConfig, PredictorConfig
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.nn.supervision_head import SupervisionHeadConfig

logger = logging.getLogger(__name__)


def build_model_config(
    common: CommonComponents, weight_multiplier: float = 1.0
) -> LatentMIMConfig:
    """Build the model config for an experiment.

    weight_multiplier scales every supervision modality's loss weight uniformly.
    """
    model_size = MODEL_SIZE_ARGS["base_super_shallow_decoder"]

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
    )
    if weight_multiplier == 1.0:
        modality_configs = SUPERVISION_MODALITY_CONFIGS
    else:
        modality_configs = {
            name: replace(cfg, weight=cfg.weight * weight_multiplier)
            for name, cfg in SUPERVISION_MODALITY_CONFIGS.items()
        }
    supervision_head_config = SupervisionHeadConfig(
        modality_configs=modality_configs,
    )
    return LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        supervision_head_config=supervision_head_config,
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

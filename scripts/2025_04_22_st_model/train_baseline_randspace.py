"""Trying to prototype fitting everything into olmo core."""

from olmo_core.config import DType
from olmo_core.distributed.parallel.data_parallel import (
    DataParallelConfig,
    DataParallelType,
)
from shared import (
    MAX_PATCH_SIZE,
    MIN_PATCH_SIZE,
    build_dataloader_config,
    build_dataset_config,
    build_train_module_config,
    build_trainer_config,
    build_visualize_config,
)

from helios.internal.common import build_common_components
from helios.internal.experiment import CommonComponents, main
from helios.nn.flexihelios import EncoderConfig, PredictorConfig
from helios.nn.latent_mim import LatentMIMConfig
from helios.train.masking import MaskingConfig
from helios.train.train_module.latent_mim import LatentMIMTrainModuleConfig


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model config for an experiment."""
    ENCODER_EMBEDDING_SIZE = 768
    DECODER_EMBEDDING_SIZE = 768
    ENCODER_DEPTH = 12
    DECODER_DEPTH = 12
    ENCODER_NUM_HEADS = 12
    DECODER_NUM_HEADS = 12
    MLP_RATIO = 4.0
    encoder_config = EncoderConfig(
        supported_modality_names=common.training_modalities,
        embedding_size=ENCODER_EMBEDDING_SIZE,
        min_patch_size=MIN_PATCH_SIZE,
        max_patch_size=MAX_PATCH_SIZE,
        num_heads=ENCODER_NUM_HEADS,
        depth=ENCODER_DEPTH,
        mlp_ratio=MLP_RATIO,
        drop_path=0.1,
        max_sequence_length=12,
        use_channel_embs=True,
    )
    decoder_config = PredictorConfig(
        encoder_embedding_size=ENCODER_EMBEDDING_SIZE,
        decoder_embedding_size=DECODER_EMBEDDING_SIZE,
        depth=DECODER_DEPTH,
        mlp_ratio=MLP_RATIO,
        num_heads=DECODER_NUM_HEADS,
        max_sequence_length=12,
        supported_modality_names=common.training_modalities,
        learnable_channel_embeddings=True,
    )
    model_config = LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
    )
    return model_config


def my_build_train_module_config(
    common: CommonComponents,
) -> LatentMIMTrainModuleConfig:
    """Get updated train module with randspace masking."""
    train_module_config = build_train_module_config(common)
    train_module_config.masking_config = MaskingConfig(
        strategy_config={
            "type": "random_space",
            "encode_ratio": 0.3,
            "decode_ratio": 0.65,
        }
    )
    train_module_config.dp_config = DataParallelConfig(
        name=DataParallelType.fsdp,
        param_dtype=DType.bfloat16,
        reduce_dtype=DType.float32,
    )
    return train_module_config


if __name__ == "__main__":
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=my_build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )

"""Trying to prototype fitting everything into olmo core."""

from shared import (
    NUM_DATA_LOADER_WORKERS,
    build_dataset_config,
    build_train_module_config,
    build_trainer_config,
    build_visualize_config,
)

from helios.data.dataloader import HeliosDataLoaderConfig
from helios.internal.common import build_common_components
from helios.internal.experiment import CommonComponents, main
from helios.nn.latent_mim import LatentMIMConfig
from helios.nn.st_model import STEncoderConfig, STPredictorConfig

MIN_PATCH_SIZE = 1
MAX_PATCH_SIZE = 8


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model config for an experiment."""
    ENCODER_EMBEDDING_SIZE = 768
    DECODER_EMBEDDING_SIZE = 768
    ENCODER_DEPTH = 24
    DECODER_DEPTH = 12
    ENCODER_NUM_HEADS = 12
    DECODER_NUM_HEADS = 12
    MLP_RATIO = 4.0
    encoder_config = STEncoderConfig(
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
    decoder_config = STPredictorConfig(
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


def build_dataloader_config(common: CommonComponents) -> HeliosDataLoaderConfig:
    """Build the dataloader config for an experiment."""
    # things should be set during building
    # TODO: Include collate function here

    GLOBAL_BATCH_SIZE = 512
    TOKEN_BUDGET = 3000
    SAMPLE_HW_P_LIST = list(range(5, 13))

    dataloader_config = HeliosDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=3622,
        work_dir=common.save_folder,
        num_workers=NUM_DATA_LOADER_WORKERS,
        sampled_hw_p_list=SAMPLE_HW_P_LIST,
        min_patch_size=MIN_PATCH_SIZE,
        max_patch_size=MAX_PATCH_SIZE,
        token_budget=TOKEN_BUDGET,
    )
    return dataloader_config


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

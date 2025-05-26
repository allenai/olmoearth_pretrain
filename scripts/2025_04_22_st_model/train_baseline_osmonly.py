"""Trying to prototype fitting everything into olmo core."""

from olmo_core.config import DType
from shared import (
    MAX_PATCH_SIZE,
    MIN_PATCH_SIZE,
    build_dataloader_config,
    build_train_module_config,
    build_trainer_config,
    build_visualize_config,
)

from helios.data.concat import HeliosConcatDatasetConfig
from helios.data.dataset import HeliosDatasetConfig
from helios.internal.common import build_common_components
from helios.internal.experiment import CommonComponents, main
from helios.nn.flexihelios import EncoderConfig, PredictorConfig
from helios.nn.latent_mim import LatentMIMConfig


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


def build_dataset_config(common: CommonComponents) -> HeliosDatasetConfig:
    """Build the dataset config for an experiment."""
    # NOTE: Change this directory based on the supported modalities
    dataset_configs = [
        HeliosDatasetConfig(
            h5py_dir="/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_rerun/sentinel1_sentinel2_l2a_worldcover/283204",
            training_modalities=common.training_modalities,
            use_samples_with_missing_supported_modalities=False,
            dtype=DType.float32,
            cache_dir="/helios_cache/osm_sampling",
        ),
    ]
    return HeliosConcatDatasetConfig(dataset_configs=dataset_configs)


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

"""Script for Debugging Galileo.

These Settings are meant to help you get quick results on a single GPU in minimal time
"""

from galileo_shared import (
    build_common_components_limited_modalities,
    build_dataloader_config,
    build_train_module_config,
    build_trainer_config,
    build_visualize_config,
)
from olmo_core.config import Config, DType

from helios.data.concat import HeliosConcatDatasetConfig
from helios.data.dataset import HeliosDatasetConfig
from helios.internal.experiment import CommonComponents, main
from helios.internal.utils import MODEL_SIZE_ARGS
from helios.nn.flexihelios import EncoderConfig, PredictorConfig
from helios.nn.galileo import GalileoConfig

MIN_PATCH_SIZE = 1
MAX_PATCH_SIZE = 8


def build_model_config(common: CommonComponents) -> GalileoConfig:
    """Build the model config for an experiment."""
    base_model_args = MODEL_SIZE_ARGS["base"]
    ENCODER_EMBEDDING_SIZE = int(base_model_args["encoder_embedding_size"])
    DECODER_EMBEDDING_SIZE = int(base_model_args["decoder_embedding_size"])
    ENCODER_DEPTH = int(base_model_args["encoder_depth"])
    DECODER_DEPTH = 4
    ENCODER_NUM_HEADS = int(base_model_args["encoder_num_heads"])
    DECODER_NUM_HEADS = int(base_model_args["decoder_num_heads"])
    MLP_RATIO = float(base_model_args["mlp_ratio"])

    encoder_config = EncoderConfig(
        supported_modality_names=common.training_modalities,
        embedding_size=ENCODER_EMBEDDING_SIZE,
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
    model_config = GalileoConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
    )
    return model_config


def build_dataset_config(common: CommonComponents) -> Config:
    """Build the dataset config for an experiment."""
    dataset_configs = [
        HeliosDatasetConfig(
            h5py_dir="/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_rerun/sentinel1_sentinel2_l2a_worldcover/283204",
            training_modalities=common.training_modalities,
            use_samples_with_missing_supported_modalities=False,
            dtype=DType.float32,
        ),
    ]
    return HeliosConcatDatasetConfig(dataset_configs=dataset_configs)


if __name__ == "__main__":
    main(
        common_components_builder=build_common_components_limited_modalities,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )

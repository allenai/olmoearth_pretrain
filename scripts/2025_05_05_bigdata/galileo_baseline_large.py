"""Script for Debugging Galileo.

These Settings are meant to help you get quick results on a single GPU in minimal time
"""

from galileo_shared import (
    build_common_components_limited_modalities,
    build_dataloader_config,
    build_dataset_config,
    build_train_module_config,
    build_trainer_config,
    build_visualize_config,
)

from helios.internal.experiment import CommonComponents, main
from helios.internal.utils import MODEL_SIZE_ARGS
from helios.nn.flexihelios import EncoderConfig, PredictorConfig
from helios.nn.galileo import GalileoConfig
from helios.train.train_module.galileo import GalileoTrainModuleConfig

MIN_PATCH_SIZE = 1
MAX_PATCH_SIZE = 8


def build_model_config(common: CommonComponents) -> GalileoConfig:
    """Build the model config for an experiment."""
    model_args = MODEL_SIZE_ARGS["large"]
    ENCODER_EMBEDDING_SIZE = int(model_args["encoder_embedding_size"])
    DECODER_EMBEDDING_SIZE = int(model_args["decoder_embedding_size"])
    ENCODER_DEPTH = int(model_args["encoder_depth"])
    DECODER_DEPTH = int(model_args["decoder_depth"])
    ENCODER_NUM_HEADS = int(model_args["encoder_num_heads"])
    DECODER_NUM_HEADS = int(model_args["decoder_num_heads"])
    MLP_RATIO = float(model_args["mlp_ratio"])

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


def my_build_train_module_config(
    common: CommonComponents,
) -> GalileoTrainModuleConfig:
    """Build the train module config for an experiment."""
    train_module_config = build_train_module_config(common)
    train_module_config.rank_microbatch_size = 8
    return train_module_config


if __name__ == "__main__":
    main(
        common_components_builder=build_common_components_limited_modalities,
        model_config_builder=build_model_config,
        train_module_config_builder=my_build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )

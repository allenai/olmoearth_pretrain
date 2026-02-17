"""Single GPU tiny model script using autocast instead of FSDP."""

import logging

from script import (
    build_common_components,
    build_dataloader_config as build_dataloader_config_base,
    build_dataset_config,
    build_train_module_config as build_train_module_config_base,
    build_trainer_config as build_trainer_config_base,
)

from olmo_core.train.callbacks import ProfilerCallback

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


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model config for an experiment."""
    model_size = MODEL_SIZE_ARGS["tiny_shallow_decoder"]

    encoder_config = EncoderConfig(
        embedding_size=model_size["encoder_embedding_size"],
        num_heads=model_size["encoder_num_heads"],
        depth=model_size["encoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        supported_modality_names=common.training_modalities,
        max_patch_size=MAX_PATCH_SIZE,
        drop_path=0.1,
        max_sequence_length=12,
    )
    decoder_config = PredictorConfig(
        encoder_embedding_size=model_size["encoder_embedding_size"],
        decoder_embedding_size=model_size["decoder_embedding_size"],
        depth=model_size["decoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        num_heads=model_size["decoder_num_heads"],
        supported_modality_names=common.training_modalities,
        max_sequence_length=12,
    )
    model_config = LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
    )
    return model_config


def build_train_module_config(common: CommonComponents):
    """Build train module config with autocast bf16 instead of FSDP."""
    from olmo_core.config import DType

    config = build_train_module_config_base(common)
    # Remove FSDP, use autocast instead
    config.dp_config = None
    config.autocast_precision = DType.bfloat16
    # Smaller microbatch for single GPU
    config.rank_microbatch_size = 128
    return config


def build_dataloader_config(common: CommonComponents):
    """Build dataloader config with smaller batch size for single GPU."""
    config = build_dataloader_config_base(common)
    config.global_batch_size = 128
    return config


def build_trainer_config(common: CommonComponents):
    """Build trainer config with profiler callback."""
    config = build_trainer_config_base(common)
    config = config.with_callback(
        "profiler",
        ProfilerCallback(
            skip_first=5,
            wait=1,
            warmup=3,
            active=3,
            repeat=1,
            with_stack=True,
            profile_memory=True,
        ),
    )
    return config


if __name__ == "__main__":
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
    )

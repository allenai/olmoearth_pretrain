"""Like train.py but with windowed attention model."""

import logging

from train import (
    build_dataloader_config,
    build_dataset_config,
    build_train_module_config,
    build_trainer_config,
    build_visualize_config,
)

from helios.data.constants import Modality
from helios.internal.common import build_common_components
from helios.internal.experiment import (
    CommonComponents,
    SubCmd,
    main,
)
from helios.internal.utils import MODEL_SIZE_ARGS
from helios.nn.flexihelios import (
    PredictorConfig,
)
from helios.nn.latent_mim import LatentMIMConfig
from helios.nn.st_model import STEncoderConfig

logger = logging.getLogger(__name__)

MAX_PATCH_SIZE = 8
MIN_PATCH_SIZE = 1


def my_build_common_components(
    script: str,
    cmd: SubCmd,
    run_name: str,
    cluster: str,
    overrides: list[str],
) -> CommonComponents:
    """Build the common components for an experiment."""
    config = build_common_components(script, cmd, run_name, cluster, overrides)
    config.training_modalities = [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
        Modality.WORLDCOVER.name,
        Modality.SRTM.name,
        Modality.LANDSAT.name,
        Modality.OPENSTREETMAP_RASTER.name,
        Modality.NAIP_10.name,
    ]
    return config


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model config for an experiment."""
    model_size = MODEL_SIZE_ARGS["base_shallow_decoder"]

    encoder_config = STEncoderConfig(
        embedding_size=model_size["encoder_embedding_size"],
        num_heads=model_size["encoder_num_heads"],
        depth=model_size["encoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        supported_modality_names=common.training_modalities,
        min_patch_size=MIN_PATCH_SIZE,
        max_patch_size=MAX_PATCH_SIZE,
        drop_path=0.1,
        max_sequence_length=12,
        windowed_attention_size=3,
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


if __name__ == "__main__":
    main(
        common_components_builder=my_build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )

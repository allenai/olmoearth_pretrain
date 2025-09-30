"""Trying to prototype fitting everything into olmo core."""

import logging

from script import (
    build_dataloader_config,
    build_dataset_config,
    build_model_config,
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
from helios.train.masking import MaskingConfig
from helios.train.train_module.contrastive_latentmim import (
    ContrastiveLatentMIMTrainModuleConfig,
)

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
        Modality.LANDSAT.name,
        Modality.WORLDCOVER.name,
        Modality.SRTM.name,
        Modality.OPENSTREETMAP_RASTER.name,
        Modality.WRI_CANOPY_HEIGHT_MAP.name,
        Modality.CDL.name,
        Modality.WORLDCEREAL.name,
    ]
    config.launch.num_gpus = 8
    return config


def my_build_train_module_config(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """Build the train module config for an experiment."""
    train_module_config = build_train_module_config(common)
    train_module_config.masking_config = MaskingConfig(
        strategy_config={
            "type": "modality_cross_random",
            "encode_ratio": 0.5,
            "decode_ratio": 0.5,
            "allow_encoding_decoding_same_bandset": True,
            "only_decode_modalities": [
                "worldcover",
                "srtm",
                "openstreetmap_raster",
                "wri_canopy_height_map",
                "cdl",
                "worldcereal",
            ],
        }
    )
    return train_module_config


if __name__ == "__main__":
    main(
        common_components_builder=my_build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=my_build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )

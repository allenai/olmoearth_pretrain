"""Base model with collapsed S2/Landsat band sets and direct supervision.

Same architecture as base_collapsed_bandsets.py but adds direct supervision heads
on all decode-only modalities:

- WorldCover: classification (11 classes)
- SRTM: regression (elevation)
- OpenStreetMap raster: binary classification (30 bands)
- WRI canopy height: regression
- CDL: regression (normalized class IDs)
- WorldCereal: binary classification (8 bands)

Supervision replaces the contrastive loss for decode-only modalities.
Encoder tokens are pooled across T/BandSets, upsampled to pixel resolution,
then per-modality linear heads produce predictions against raw pixel targets.
"""

import logging

from base_collapsed_bandsets import (
    build_common_components,
)
from new_masking_script import (
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
from olmoearth_pretrain.nn.supervision_head import (
    SupervisionHeadConfig,
    SupervisionModalityConfig,
    SupervisionTaskType,
)

logger = logging.getLogger(__name__)

MAX_PATCH_SIZE = 8

WORLDCOVER_CLASS_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]

SUPERVISION_MODALITY_CONFIGS = {
    "worldcover": SupervisionModalityConfig(
        task_type=SupervisionTaskType.CLASSIFICATION,
        num_output_channels=11,
        weight=1.0,
        class_values=WORLDCOVER_CLASS_VALUES,
    ),
    "srtm": SupervisionModalityConfig(
        task_type=SupervisionTaskType.REGRESSION,
        num_output_channels=1,
        weight=1.0,
    ),
    "openstreetmap_raster": SupervisionModalityConfig(
        task_type=SupervisionTaskType.BINARY_CLASSIFICATION,
        num_output_channels=30,
        weight=1.0,
    ),
    "wri_canopy_height_map": SupervisionModalityConfig(
        task_type=SupervisionTaskType.REGRESSION,
        num_output_channels=1,
        weight=1.0,
    ),
    "cdl": SupervisionModalityConfig(
        task_type=SupervisionTaskType.REGRESSION,
        num_output_channels=1,
        weight=1.0,
    ),
    "worldcereal": SupervisionModalityConfig(
        task_type=SupervisionTaskType.BINARY_CLASSIFICATION,
        num_output_channels=8,
        weight=1.0,
    ),
}


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model config with collapsed band sets and supervision head."""
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
        tokenization_config=common.tokenization_config,
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

    supervision_head_config = SupervisionHeadConfig(
        modality_configs=SUPERVISION_MODALITY_CONFIGS,
    )

    model_config = LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        supervision_head_config=supervision_head_config,
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

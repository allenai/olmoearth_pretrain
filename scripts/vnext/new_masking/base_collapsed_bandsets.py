"""Base model with collapsed S2 and Landsat band sets.

This experiment collapses all Sentinel-2 L2A and Landsat bands into single tokens
per spatial location, instead of the default resolution-based band groupings.

Change from baseline:
- Sentinel-2 L2A: 1 token containing all 12 bands instead of 3 resolution-based tokens
- Landsat: 1 token containing all 11 bands instead of 2 resolution-based tokens
- All other modalities: unchanged (use default tokenization)

This forces the model to learn joint representations across all spectral bands
within each modality.
"""

import logging

from new_masking_script import (
    build_common_components as build_common_components_base,
)
from new_masking_script import (
    build_dataloader_config,
    build_dataset_config,
    build_train_module_config,
    build_trainer_config,
    build_visualize_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, SubCmd, main
from olmoearth_pretrain.internal.utils import MODEL_SIZE_ARGS
from olmoearth_pretrain.nn.flexihelios import (
    EncoderConfig,
    PredictorConfig,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.nn.tokenization import (
    ModalityTokenization,
    TokenizationConfig,
)

logger = logging.getLogger(__name__)

MAX_PATCH_SIZE = 8
MIN_PATCH_SIZE = 1


def build_common_components(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """Build common components with collapsed S2/Landsat tokenization."""
    # All Sentinel-2 L2A bands collapsed into a single token
    # Default has 3 tokens: 10m (B02,B03,B04,B08), 20m (B05,B06,B07,B8A,B11,B12), 60m (B01,B09)
    sentinel2_collapsed = ModalityTokenization(
        band_groups=[
            [
                "B02",
                "B03",
                "B04",
                "B08",
                "B05",
                "B06",
                "B07",
                "B8A",
                "B11",
                "B12",
                "B01",
                "B09",
            ],
        ]
    )

    # All Landsat bands collapsed into a single token
    # Default has 2 tokens: 15m (B8), 30m (B1,B2,B3,B4,B5,B6,B7,B9,B10,B11)
    landsat_collapsed = ModalityTokenization(
        band_groups=[
            ["B8", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"],
        ]
    )

    tokenization_config = TokenizationConfig(
        overrides={
            "sentinel2_l2a": sentinel2_collapsed,
            "landsat": landsat_collapsed,
        }
    )

    common = build_common_components_base(script, cmd, run_name, cluster, overrides)
    common.tokenization_config = tokenization_config
    return common


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model config with collapsed S2 and Landsat band sets."""
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
    model_config = LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
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

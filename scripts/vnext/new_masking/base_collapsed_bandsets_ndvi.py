"""Base model with collapsed S2/Landsat band sets + NDVI as a decode-only modality.

Same as base_collapsed_bandsets.py but adds NDVI. NDVI is computed from
Sentinel-2 L2A bands B04 (Red) and B08 (NIR) in the dataset before
normalization and is only predicted by the decoder (never encoded).

Change from base_collapsed_bandsets:
- Adds NDVI to training modalities
- NDVI is decode-only (never seen by the encoder, only predicted by the decoder)
"""

import logging

from new_masking_script import (
    build_common_components as build_common_components_base,
)
from new_masking_script import (
    build_dataloader_config as build_dataloader_config_base,
)
from new_masking_script import (
    build_dataset_config,
    build_trainer_config,
    build_visualize_config,
)
from new_masking_script import (
    build_train_module_config as build_train_module_config_base,
)
from new_masking_script import (
    get_masking_config as get_masking_config_base,
)

from olmoearth_pretrain.data.constants import Modality
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


def _get_tokenization_config() -> TokenizationConfig:
    """Build collapsed tokenization config for S2, Landsat, and NDVI."""
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
    landsat_collapsed = ModalityTokenization(
        band_groups=[
            ["B8", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"],
        ]
    )
    ndvi_tokenization = ModalityTokenization(band_groups=[["ndvi"]])
    return TokenizationConfig(
        overrides={
            "sentinel2_l2a": sentinel2_collapsed,
            "landsat": landsat_collapsed,
            "ndvi": ndvi_tokenization,
        }
    )


def get_masking_config(common: CommonComponents):
    """Get masking config with NDVI as a decode-only modality."""
    base = get_masking_config_base(common)
    base.strategy_config["only_decode_modalities"].append(Modality.NDVI.name)
    return base


def build_common_components(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """Build common components with collapsed tokenization and NDVI."""
    common = build_common_components_base(script, cmd, run_name, cluster, overrides)
    common.training_modalities.append(Modality.NDVI.name)
    common.tokenization_config = _get_tokenization_config()
    return common


def build_train_module_config(common: CommonComponents):
    """Build train module config, overriding masking to include NDVI as decode-only."""
    config = build_train_module_config_base(common)
    config.masking_config = get_masking_config(common)
    return config


def build_dataloader_config(common: CommonComponents):
    """Build dataloader config, overriding masking to include NDVI as decode-only."""
    config = build_dataloader_config_base(common)
    config.masking_config = get_masking_config(common)
    return config


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build model config with collapsed bandsets and NDVI."""
    model_size = MODEL_SIZE_ARGS["base_shallow_decoder"]
    tokenization_config = _get_tokenization_config()

    encoder_config = EncoderConfig(
        embedding_size=model_size["encoder_embedding_size"],
        num_heads=model_size["encoder_num_heads"],
        depth=model_size["encoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        supported_modality_names=common.training_modalities,
        max_patch_size=MAX_PATCH_SIZE,
        drop_path=0.1,
        max_sequence_length=12,
        tokenization_config=tokenization_config,
    )
    decoder_config = PredictorConfig(
        encoder_embedding_size=model_size["encoder_embedding_size"],
        decoder_embedding_size=model_size["decoder_embedding_size"],
        depth=model_size["decoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        num_heads=model_size["decoder_num_heads"],
        supported_modality_names=common.training_modalities,
        max_sequence_length=12,
        tokenization_config=tokenization_config,
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

"""Spectral grouping tokenization for Sentinel-2.

Hypothesis: Grouping bands by spectral similarity creates more semantically
meaningful tokens that better align with physical properties of the Earth's surface.

Change from baseline:
- Sentinel-2 L2A: 5 tokens grouped by spectral similarity instead of 3 resolution-based tokens
  - RGB (visible): B02, B03, B04
  - NIR (near-infrared): B08, B8A
  - Red Edge: B05, B06, B07
  - SWIR (shortwave infrared): B11, B12
  - Atmospheric: B01, B09
- All other modalities: unchanged (use default tokenization)

Expected outcome: Tokens that better capture spectral phenomena like vegetation indices,
water detection, and atmospheric effects.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path for script imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from script import (
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_train_module_config,
    build_trainer_config,
    build_visualize_config,
)

from olmoearth_pretrain.data.constants import (
    ModalityTokenization,
    TokenizationBandSet,
    TokenizationConfig,
)
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

# Sentinel-2 L2A bands grouped by spectral similarity
SENTINEL2_SPECTRAL_GROUPING = ModalityTokenization(
    band_groups=[
        # Visible RGB
        TokenizationBandSet(bands=["B02", "B03", "B04"]),
        # Near-infrared
        TokenizationBandSet(bands=["B08", "B8A"]),
        # Red edge
        TokenizationBandSet(bands=["B05", "B06", "B07"]),
        # Shortwave infrared
        TokenizationBandSet(bands=["B11", "B12"]),
        # Atmospheric bands
        TokenizationBandSet(bands=["B01", "B09"]),
    ]
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model config with spectral grouping tokenization for Sentinel-2."""
    model_size = MODEL_SIZE_ARGS["base_shallow_decoder"]

    tokenization_config = TokenizationConfig(
        overrides={
            "sentinel2_l2a": SENTINEL2_SPECTRAL_GROUPING,
        }
    )

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

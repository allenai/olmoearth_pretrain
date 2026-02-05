"""All bands in a single token for Sentinel-2.

Hypothesis: Combining all bands into a single token forces the model to learn
joint representations across all spectral bands, potentially capturing holistic
spectral signatures better than split tokenization.

Change from baseline:
- Sentinel-2 L2A: 1 token containing all 12 bands instead of 3 bandset tokens
- All other modalities: unchanged (use default tokenization)

Expected outcome: More compact token representation that may better capture
full spectral signatures, at the cost of losing band-level granularity.
"""

import logging
import sys
from pathlib import Path

# Add official directory to path for script imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "official"))

from script import (
    build_common_components as build_common_components_base,
)
from script import (
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

# All Sentinel-2 L2A bands in a single token
SENTINEL2_ALL_BANDS_SINGLE_TOKEN = ModalityTokenization(
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

# Tokenization config used by model, masking, and dataloader
TOKENIZATION_CONFIG = TokenizationConfig(
    overrides={
        "sentinel2_l2a": SENTINEL2_ALL_BANDS_SINGLE_TOKEN,
    }
)


def build_common_components(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """Build common components with single-token S2 tokenization."""
    common = build_common_components_base(script, cmd, run_name, cluster, overrides)
    common.tokenization_config = TOKENIZATION_CONFIG
    return common


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model config with all bands in a single token for Sentinel-2."""
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

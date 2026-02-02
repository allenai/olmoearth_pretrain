"""Single-band tokenization for Sentinel-2.

Hypothesis: Each band as its own token allows the model to learn more fine-grained
band-specific representations, potentially improving downstream task performance
where specific spectral bands are important.

Change from baseline:
- Sentinel-2 L2A: 12 tokens (one per band) instead of 3 bandset tokens
- All other modalities: unchanged (use default tokenization)

Expected outcome: Better band-level representations at the cost of more tokens
per spatial location.
"""

import logging
import sys
from pathlib import Path

# Add official directory to path for script imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "official"))

from script import (
    build_common_components,
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
from olmoearth_pretrain.nn.tokenization import (
    ModalityTokenization,
    TokenizationConfig,
)

logger = logging.getLogger(__name__)

MAX_PATCH_SIZE = 8
MIN_PATCH_SIZE = 1

# Sentinel-2 L2A bands in order: each becomes its own token
SENTINEL2_SINGLE_BAND_TOKENIZATION = ModalityTokenization(
    band_groups=[
        ["B02"],
        ["B03"],
        ["B04"],
        ["B08"],
        ["B05"],
        ["B06"],
        ["B07"],
        ["B8A"],
        ["B11"],
        ["B12"],
        ["B01"],
        ["B09"],
    ]
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model config with single-band tokenization for Sentinel-2."""
    model_size = MODEL_SIZE_ARGS["base_shallow_decoder"]

    tokenization_config = TokenizationConfig(
        overrides={
            "sentinel2_l2a": SENTINEL2_SINGLE_BAND_TOKENIZATION,
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

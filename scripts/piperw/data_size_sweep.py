"""Script for running pretraining with different data sizes.

This script supports both nano and tiny models with different data sizes:
- 500 samples
- 1000 samples  
- 5000 samples
- all data (full dataset)
"""

import logging
import sys
from pathlib import Path

# Add the official directory to the path so we can import from script
sys.path.insert(0, str(Path(__file__).parent.parent / "official"))

from script import (
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_train_module_config,
    build_trainer_config,
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

# Total dataset size from the h5py_dir name
TOTAL_DATASET_SIZE = 1138828


def calculate_dataset_percentage(num_samples: int | str) -> float:
    """Calculate dataset_percentage from number of samples.
    
    Args:
        num_samples: Number of samples (int) or "all" for full dataset
        
    Returns:
        Dataset percentage (0.0 to 1.0)
    """
    if num_samples == "all":
        return 1.0
    elif isinstance(num_samples, int):
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        if num_samples > TOTAL_DATASET_SIZE:
            logger.warning(
                f"Requested {num_samples} samples but dataset only has {TOTAL_DATASET_SIZE}. "
                f"Using full dataset (1.0)"
            )
            return 1.0
        return num_samples / TOTAL_DATASET_SIZE
    else:
        raise ValueError(f"num_samples must be int or 'all', got {type(num_samples)}")


def build_model_config(
    common: CommonComponents, model_size_name: str
) -> LatentMIMConfig:
    """Build the model config for an experiment.
    
    Args:
        common: Common components
        model_size_name: Model size name ("nano" or "tiny_shallow_decoder")
    """
    if model_size_name not in MODEL_SIZE_ARGS:
        raise ValueError(
            f"Unknown model size: {model_size_name}. "
            f"Available: {list(MODEL_SIZE_ARGS.keys())}"
        )
    
    model_size = MODEL_SIZE_ARGS[model_size_name]

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


def build_model_config_nano(common: CommonComponents) -> LatentMIMConfig:
    """Build nano model config."""
    return build_model_config(common, "nano")


def build_model_config_tiny(common: CommonComponents) -> LatentMIMConfig:
    """Build tiny model config."""
    return build_model_config(common, "tiny_shallow_decoder")


if __name__ == "__main__":
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config_nano,  # Default to nano, can be overridden
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
    )


"""Large model training script with scaled hyperparameters.

This script uses properly scaled hyperparameters for the large model:
- Learning rate: Scaled down using sqrt scaling (0.000035)
- Warmup steps: Scaled up with depth (48000 steps)
- Drop path: Scaled up with depth (0.6)
- Weight decay: Slightly increased (0.003)
"""

import logging
import math

try:
    from .script import (
        build_common_components,
        build_dataloader_config,
        build_dataset_config as build_dataset_config_default,
        build_train_module_config as build_train_module_config_default,
        build_trainer_config,
    )
    from .scaling_utils import (
        get_scaled_optim_config,
        get_scaled_scheduler,
        get_scaled_drop_path,
        get_model_scaling_factors,
    )
except ImportError:
    # Fallback for when running directly (not as a module)
    from script import (
        build_common_components,
        build_dataloader_config,
        build_dataset_config as build_dataset_config_default,
        build_train_module_config as build_train_module_config_default,
        build_trainer_config,
    )
    from scaling_utils import (
        get_scaled_optim_config,
        get_scaled_scheduler,
        get_scaled_drop_path,
        get_model_scaling_factors,
    )

from olmo_core.optim import AdamWConfig
from olmo_core.optim.scheduler import CosWithWarmup
from olmo_core.config import DType
from olmo_core.distributed.parallel.data_parallel import (
    DataParallelConfig,
    DataParallelType,
)

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig
from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.internal.utils import MODEL_SIZE_ARGS
from olmoearth_pretrain.nn.flexihelios import (
    EncoderConfig,
    PredictorConfig,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import MaskingConfig
from olmoearth_pretrain.train.train_module.contrastive_latentmim import (
    ContrastiveLatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

MAX_PATCH_SIZE = 8
MIN_PATCH_SIZE = 1

# Model size key for large model
MODEL_SIZE_KEY = "large_shallow_decoder"


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model config for large model with scaled drop_path."""
    model_size = MODEL_SIZE_ARGS[MODEL_SIZE_KEY]

    # Get scaled drop_path
    drop_path = get_scaled_drop_path(
        MODEL_SIZE_KEY,
        base_drop_path=0.1,
        drop_path_scaling="depth",
    )

    # Log scaling factors
    factors = get_model_scaling_factors(MODEL_SIZE_KEY)
    logger.info(
        f"Large model scaling factors: "
        f"embedding_ratio={factors['embedding_ratio']:.2f}, "
        f"depth_ratio={factors['depth_ratio']:.2f}, "
        f"param_ratio={factors['param_ratio']:.2f}"
    )
    logger.info(f"Using scaled drop_path: {drop_path:.3f} (base: 0.1)")

    encoder_config = EncoderConfig(
        embedding_size=model_size["encoder_embedding_size"],
        num_heads=model_size["encoder_num_heads"],
        depth=model_size["encoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        supported_modality_names=common.training_modalities,
        max_patch_size=MAX_PATCH_SIZE,
        drop_path=drop_path,
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


def build_train_module_config(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """Build the train module config with scaled hyperparameters for large model."""
    # Get scaled optimizer config
    optim_config = get_scaled_optim_config(
        MODEL_SIZE_KEY,
        base_lr=0.0001,
        base_weight_decay=0.02,
        lr_scaling="sqrt",  # Recommended: sqrt scaling
        wd_scaling="slight_increase",
    )

    # Get scaled scheduler
    scheduler = get_scaled_scheduler(
        MODEL_SIZE_KEY,
        base_warmup=8000,
        warmup_scaling="depth",  # Recommended: scale with depth
    )

    # Log the scaled values
    logger.info(
        f"Scaled hyperparameters for large model: "
        f"LR={optim_config.lr:.6f} (base: 0.0001), "
        f"WD={optim_config.weight_decay:.4f} (base: 0.02), "
        f"warmup={scheduler.warmup_steps} (base: 8000)"
    )

    return ContrastiveLatentMIMTrainModuleConfig(
        optim_config=optim_config,
        rank_microbatch_size=32,
        masking_config=MaskingConfig(
            strategy_config={
                "type": "modality_cross_random",
                "encode_ratio": 0.5,
                "decode_ratio": 0.5,
                "allow_encoding_decoding_same_bandset": True,
                "only_decode_modalities": [
                    Modality.WORLDCOVER.name,
                    Modality.SRTM.name,
                    Modality.OPENSTREETMAP_RASTER.name,
                    Modality.WRI_CANOPY_HEIGHT_MAP.name,
                    Modality.CDL.name,
                    Modality.WORLDCEREAL.name,
                ],
            }
        ),
        loss_config=LossConfig(
            loss_config={
                "type": "modality_patch_discrimination_new",
                "tau": 0.1,
            }
        ),
        contrastive_config=LossConfig(
            loss_config={
                "type": "InfoNCE",
                "weight": 0.1,
            }
        ),
        token_exit_cfg={modality: 0 for modality in common.training_modalities},
        max_grad_norm=1.0,
        scheduler=scheduler,
        ema_decay=(1.0, 1.0),
        dp_config=DataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
    )


def build_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """Build the dataset config for an experiment."""
    config = build_dataset_config_default(common)
    # Only limit training samples if explicitly provided via command line
    if common.max_training_samples is not None:
        config.max_training_samples = common.max_training_samples
        logger.info(f"Limiting dataset to {common.max_training_samples} training samples")
    else:
        logger.info("Using all available training samples (no limit)")
    config.seed = 3622  # For reproducible random sampling
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


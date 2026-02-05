"""Histogram prediction experiment with collapsed S2 and Landsat band sets.

This experiment extends base_collapsed_bandsets.py by adding histogram prediction
heads to the encoder. The model learns to predict aggregated value distributions
(histograms) for decode-only modalities directly from the pooled encoder
representations.

Histogram prediction modalities:
- worldcover: 11 bins (ESA WorldCover classes 10-100 mapped to 0-10)
- worldcereal: 16 bins (8 bands x 2 classes each, binary classification)
- openstreetmap_raster: 60 bins (30 bands x 2 classes each, binary presence/absence)
"""

import logging
import sys
from pathlib import Path

# Add parent directories to path for script imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "official"))

from base_collapsed_bandsets import (
    MAX_PATCH_SIZE,
    build_common_components,
)
from new_masking_script import (
    build_dataloader_config,
    build_dataset_config,
    build_trainer_config,
    build_visualize_config,
    get_masking_config,
)
from olmo_core.config import DType
from olmo_core.distributed.parallel.data_parallel import (
    DataParallelConfig,
    DataParallelType,
)
from olmo_core.optim import AdamWConfig
from olmo_core.optim.scheduler import CosWithWarmup

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.internal.utils import MODEL_SIZE_ARGS
from olmoearth_pretrain.nn.flexihelios import (
    EncoderConfig,
    PredictorConfig,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.train_module.contrastive_latentmim import (
    ContrastiveLatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

# WorldCover class values (normalized: original_value / 100)
# Original classes: 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100
WORLDCOVER_CLASS_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]

# Histogram configuration for each modality
# Format: {modality_name: num_bins}
# The encoder will have a linear head for each modality that predicts num_bins values
HISTOGRAM_CONFIG = {
    "worldcover": 11,  # ESA WorldCover has 11 classes
    "worldcereal": 8,  # 8 bands, binary per band -> predict mean per band
    "openstreetmap_raster": 30,  # 30 bands, binary per band -> predict mean per band
}

# Histogram modalities config for train module
# This tells the train module how to compute histogram targets from the batch
HISTOGRAM_MODALITIES = {
    "worldcover": {
        "num_bins": 11,
        "categorical": True,
        # class_values maps normalized values to bin indices
        "class_values": WORLDCOVER_CLASS_VALUES,
    },
    "worldcereal": {
        # 8 bands, each is a one-hot encoded category
        # Count positive values (>= threshold) per band
        "num_bins": 8,
        "one_hot": True,
        "one_hot_threshold": 0.5,  # Values are normalized, threshold at 0.5
    },
    "openstreetmap_raster": {
        # 30 bands, each is a one-hot encoded category (presence/absence)
        # Count positive values (>= threshold) per band
        "num_bins": 30,
        "one_hot": True,
        "one_hot_threshold": 0.5,
    },
}


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model config with histogram prediction heads.

    Extends base_collapsed_bandsets by adding histogram_config to the encoder.
    """
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
        # Add histogram prediction heads
        histogram_config=HISTOGRAM_CONFIG,
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


def build_train_module_config(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """Build the train module config with histogram reconstruction loss.

    Extends base train module config by adding:
    - histogram_loss_config: Configures the loss function for histogram prediction
    - histogram_modalities: Specifies how to compute histogram targets from batch
    """
    return ContrastiveLatentMIMTrainModuleConfig(
        optim_config=AdamWConfig(lr=0.0001, weight_decay=0.02, fused=False),
        rank_microbatch_size=32,
        masking_config=get_masking_config(common),
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
        # Histogram reconstruction loss configuration
        histogram_loss_config=LossConfig(
            loss_config={
                "type": "histogram_reconstruction",
                "loss_type": "cross_entropy",  # Options: cross_entropy, mse, kl_divergence
                "weight": 0.1,
            }
        ),
        histogram_modalities=HISTOGRAM_MODALITIES,
        token_exit_cfg={modality: 0 for modality in common.training_modalities},
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=8000),
        ema_decay=(1.0, 1.0),
        dp_config=DataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
    )


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

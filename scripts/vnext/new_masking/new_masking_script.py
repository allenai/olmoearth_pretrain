"""Trying to prototype fitting everything into olmo core."""

import logging
import sys
from pathlib import Path

# Add official directory to path for script imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "official"))

from olmo_core.config import DType
from olmo_core.distributed.parallel.data_parallel import (
    DataParallelConfig,
    DataParallelType,
)
from olmo_core.optim import AdamWConfig
from olmo_core.optim.scheduler import CosWithWarmup
from script import (
    build_common_components,
    build_dataset_config,
    build_trainer_config,
    build_visualize_config,
)

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig
from olmoearth_pretrain.internal.experiment import (
    CommonComponents,
)
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import MaskingConfig
from olmoearth_pretrain.train.train_module.contrastive_latentmim import (
    ContrastiveLatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

MAX_PATCH_SIZE = 8
MIN_PATCH_SIZE = 1

__all__ = [
    "build_common_components",
    "build_dataset_config",
    "build_visualize_config",
    "build_trainer_config",
]


def get_masking_config(common: CommonComponents) -> MaskingConfig:
    """Get the masking configuration for the experiment.

    Args:
        common: Common experiment components containing optional tokenization_config.
    """
    return MaskingConfig(
        strategy_config={
            # "type": "modality_cross_random",
            "type": "random_with_decode",
            "encode_ratio": 0.5,
            "decode_ratio": 0.5,
            # "allow_encoding_decoding_same_bandset": True,
            "only_decode_modalities": [
                Modality.WORLDCOVER.name,
                Modality.SRTM.name,
                Modality.OPENSTREETMAP_RASTER.name,
                Modality.WRI_CANOPY_HEIGHT_MAP.name,
                Modality.CDL.name,
                Modality.WORLDCEREAL.name,
            ],
        },
        tokenization_config=common.tokenization_config,
    )


def build_train_module_config(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """Build the train module config for an experiment.

    Args:
        common: Common experiment components.
    """
    # The train module still needs the masking_config for reference (e.g., for metric
    # naming), but the actual masking happens in the dataloader workers.
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


def build_dataloader_config(
    common: CommonComponents,
) -> OlmoEarthDataLoaderConfig:
    """Build the dataloader config for an experiment.

    Masking is performed in the dataloader workers (CPU) instead of in the train module
    (GPU). This improves throughput by offloading CPU-bound masking operations to
    dataloader workers.

    Args:
        common: Common experiment components.
    """
    return OlmoEarthDataLoaderConfig(
        num_workers=12,
        global_batch_size=512,
        token_budget=2250,
        prefetch_factor=2,
        sampled_hw_p_list=list(range(1, 13)),  # try only temporal tokens
        min_patch_size=MIN_PATCH_SIZE,
        max_patch_size=MAX_PATCH_SIZE,
        work_dir=common.save_folder,
        seed=3622,
        num_masked_views=2,  # ContrastiveLatentMIM needs 2 views
        masking_config=get_masking_config(common),
        # masking_config_b is not set, so both views use the same strategy
    )

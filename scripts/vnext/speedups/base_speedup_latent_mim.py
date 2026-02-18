"""Speed-optimized base model using LatentMIM (single forward pass, no contrastive loss).

Same model architecture and speed optimizations as base_speedup.py, but uses
LatentMIMTrainModule instead of ContrastiveLatentMIMTrainModule:
  - Single masked view per step (vs two views for the contrastive version)
  - No InfoNCE loss term
  - ~2x cheaper per step vs the contrastive variant

Launch example:
    python scripts/vnext/speedups/base_speedup_latent_mim.py launch base_speedup_latentmim ai2/jupiter \
        --launch.num_gpus=8 \
        --launch.clusters=[ai2/ceres,ai2/jupiter,ai2/titan] \
        --trainer.callbacks.wandb.project=YYYY_MM_DD_speed_optimizations
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "official"))

from script import (  # noqa: E402
    build_common_components,
    build_dataset_config,
    build_trainer_config,
    build_visualize_config,
    get_masking_config,
)

# Reuse the model config (flash attn + linear patch embed) from the contrastive script
from base_speedup import build_model_config  # noqa: E402

from olmo_core.config import DType  # noqa: E402
from olmo_core.distributed.parallel.data_parallel import (  # noqa: E402
    DataParallelConfig,
    DataParallelType,
)
from olmo_core.optim import AdamWConfig  # noqa: E402
from olmo_core.optim.scheduler import CosWithWarmup  # noqa: E402

from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig  # noqa: E402
from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.train.loss import LossConfig  # noqa: E402
from olmoearth_pretrain.train.train_module.latent_mim import (  # noqa: E402
    LatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

MAX_PATCH_SIZE = 8
MIN_PATCH_SIZE = 1


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """Single-view LatentMIM train module with compile enabled."""
    return LatentMIMTrainModuleConfig(
        optim_config=AdamWConfig(lr=0.0001, weight_decay=0.02, fused=True),
        rank_microbatch_size=32,
        masking_config=get_masking_config(common),
        loss_config=LossConfig(
            loss_config={
                "type": "modality_patch_discrimination_new_vec",
                "tau": 0.1,
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
        compile_model=True,
    )


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """Single-view dataloader — LatentMIM only needs one masked view per step."""
    from script import build_dataloader_config as _base

    config = _base(common)
    config.num_masked_views = 1
    return config


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

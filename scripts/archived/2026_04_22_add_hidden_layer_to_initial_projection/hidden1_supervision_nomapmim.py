"""hidden1_supervision with patch-discrimination disabled on the map modalities.

Keeps the latent MIM / token contrastive signal on sentinel2/landsat/sentinel1
only. The ONLY_DECODE_MODALITIES (worldcover, srtm, openstreetmap_raster,
wri_canopy_height_map, cdl, worldcereal) are still supervised via dedicated
supervision heads but no longer contribute to the patch discrimination loss.
"""

import logging

from hidden1 import (
    ONLY_DECODE_MODALITIES,
    _masking_config,
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_trainer_config,
    build_visualize_config,
)
from hidden1_supervision import build_model_config
from olmo_core.config import DType
from olmo_core.distributed.parallel.data_parallel import (
    DataParallelConfig,
    DataParallelType,
)
from olmo_core.optim import AdamWConfig
from olmo_core.optim.scheduler import CosWithWarmup

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.train_module.contrastive_latentmim import (
    ContrastiveLatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

NO_MIM_MODALITY_WEIGHTS = {modality: 0.0 for modality in ONLY_DECODE_MODALITIES}


def build_train_module_config(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """Patch-discrimination loss zeroed out on the map modalities."""
    return ContrastiveLatentMIMTrainModuleConfig(
        optim_config=AdamWConfig(lr=0.0001, weight_decay=0.02, fused=False),
        rank_microbatch_size=64,
        masking_config=_masking_config(common.tokenization_config),
        loss_config=LossConfig(
            loss_config={
                "type": "modality_patch_discrimination_masked_negatives_vec",
                "tau": 0.1,
                "same_target_threshold": 0.999,
                "mask_negatives_for_modalities": ONLY_DECODE_MODALITIES,
                "modality_weights": NO_MIM_MODALITY_WEIGHTS,
            }
        ),
        contrastive_config=LossConfig(
            loss_config={
                "type": "InfoNCE",
                "weight": 0.05,
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


def run() -> None:
    """Run the experiment."""
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )


if __name__ == "__main__":
    run()

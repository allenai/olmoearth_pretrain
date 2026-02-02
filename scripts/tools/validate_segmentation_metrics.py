#!/usr/bin/env python3
"""Validate segmentation metrics by running LP and Finetune on PASTIS with the tiny model.

This script runs both tasks in a single job:
1. Linear Probe on pastis_sentinel2 (pastis_lp)
2. Finetune on pastis_sentinel2 (pastis_ft)

Both should return dict metrics with: miou, overall_acc, macro_acc, micro_f1, macro_f1

Usage:
    # Dry run (just print config):
    python scripts/tools/validate_segmentation_metrics.py dry_run_evaluate validate_seg_metrics local

    # Run locally:
    torchrun scripts/tools/validate_segmentation_metrics.py evaluate validate_seg_metrics local

    # Launch on Beaker:
    python scripts/tools/validate_segmentation_metrics.py launch_evaluate validate_seg_metrics ai2/saturn-cirrascale
"""

import logging

from olmo_core.config import DType
from olmo_core.distributed.parallel.data_parallel import (
    DataParallelConfig,
    DataParallelType,
)
from olmo_core.optim import AdamWConfig
from olmo_core.optim.scheduler import CosWithWarmup
from olmo_core.train.callbacks import (
    BeakerCallback,
    ConfigSaverCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
)
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.common import Duration, LoadStrategy
from olmo_core.train.config import TrainerConfig

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig
from olmoearth_pretrain.internal.constants import WANDB_ENTITY
from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.internal.utils import MODEL_SIZE_ARGS
from olmoearth_pretrain.nn.flexi_vit import PoolingType
from olmoearth_pretrain.nn.flexihelios import EncoderConfig, PredictorConfig
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.callbacks import (
    DownstreamEvaluatorCallbackConfig,
    OlmoEarthWandBCallback,
)
from olmoearth_pretrain.train.callbacks.evaluator_callback import (
    DownstreamTaskConfig,
    EvalMode,
)
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import MaskingConfig
from olmoearth_pretrain.train.train_module.contrastive_latentmim import (
    ContrastiveLatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

# Default checkpoint path for tiny model
DEFAULT_CHECKPOINT = (
    "/weka/dfive-default/helios/checkpoints/joer/tiny_lr0.0002_wd0.02/step360000"
)

MAX_PATCH_SIZE = 8
MIN_PATCH_SIZE = 1


def build_common_components(
    script: str,
    cmd: str,
    run_name: str,
    cluster: str,
    overrides: list[str],
) -> CommonComponents:
    """Build common components."""
    from olmoearth_pretrain.internal.common import build_common_components as _build

    return _build(script, cmd, run_name, cluster, overrides)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model config for tiny model."""
    model_size = MODEL_SIZE_ARGS["tiny_shallow_decoder"]

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
    return LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
    )


def build_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """Build the dataset config."""
    return OlmoEarthDatasetConfig(
        h5py_dir="/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_zstd_3_128_x_4/cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_worldpop_wri_canopy_height_map/1138828",
        training_modalities=common.training_modalities,
    )


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """Build the dataloader config."""
    return OlmoEarthDataLoaderConfig(
        num_workers=16,
        global_batch_size=512,
        token_budget=2250,
        prefetch_factor=4,
        sampled_hw_p_list=list(range(1, 13)),
        min_patch_size=MIN_PATCH_SIZE,
        max_patch_size=MAX_PATCH_SIZE,
        work_dir=common.save_folder,
        seed=3622,
    )


def build_train_module_config(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """Build the train module config (needed to load checkpoint)."""
    return ContrastiveLatentMIMTrainModuleConfig(
        optim_config=AdamWConfig(lr=0.0002, weight_decay=0.02, fused=False),
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
        scheduler=CosWithWarmup(warmup_steps=8000),
        ema_decay=(1.0, 1.0),
        dp_config=DataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
    )


# Both LP and Finetune tasks for validation
TASKS = {
    "pastis_lp": DownstreamTaskConfig(
        dataset="pastis",
        embedding_batch_size=32,
        probe_batch_size=8,
        num_workers=2,
        pooling_type=PoolingType.MAX,
        norm_stats_from_pretrained=True,
        probe_lr=0.1,
        eval_interval=Duration.epochs(10),
        input_modalities=[Modality.SENTINEL2_L2A.name],
        epochs=20,  # Reduced for faster validation
        eval_mode=EvalMode.LINEAR_PROBE,
    ),
    "pastis_ft": DownstreamTaskConfig(
        dataset="pastis",
        ft_batch_size=16,
        num_workers=2,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        ft_lr=0.0001,
        input_modalities=[Modality.SENTINEL2_L2A.name],
        epochs=10,  # Reduced for faster validation
        eval_mode=EvalMode.FINETUNE,
    ),
    "m-eurosat": DownstreamTaskConfig(
        dataset="m-eurosat",
        embedding_batch_size=128,
        num_workers=0,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        eval_interval=Duration.steps(4000),
    ),
}


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """Build the trainer config for validation."""
    MAX_DURATION = Duration.epochs(300)
    LOAD_STRATEGY = LoadStrategy.if_available
    checkpointer_config = CheckpointerConfig(work_dir=common.save_folder)

    wandb_callback = OlmoEarthWandBCallback(
        name=common.run_name,
        project="2025_01_30_validate_seg_metrics",
        entity=WANDB_ENTITY,
        enabled=True,
        upload_dataset_distribution_pre_train=False,
        upload_modality_data_band_distribution_pre_train=False,
    )
    garbage_collector_callback = GarbageCollectorCallback(gc_interval=1)

    logger.info("Running both LINEAR_PROBE and FINETUNE on PASTIS")

    trainer_config = (
        TrainerConfig(
            work_dir=common.save_folder,
            load_path=DEFAULT_CHECKPOINT,
            load_strategy=LOAD_STRATEGY,
            save_folder=common.save_folder,
            cancel_check_interval=1,
            metrics_collect_interval=10,
            max_duration=MAX_DURATION,
            checkpointer=checkpointer_config,
        )
        .with_callback("wandb", wandb_callback)
        .with_callback("gpu_memory_monitor", GPUMemoryMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=TASKS,
                eval_on_startup=True,
                cancel_after_first_eval=True,
                run_on_test=True,
            ),
        )
        .with_callback("garbage_collector", garbage_collector_callback)
        .with_callback("beaker", BeakerCallback())
    )
    return trainer_config


if __name__ == "__main__":
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        train_module_config_builder=build_train_module_config,
    )

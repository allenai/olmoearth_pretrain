"""Compare DDP vs FSDP training for debugging divergence.

This script sets up identical training runs with either DDP or FSDP
to diagnose why results differ between the two strategies.

Usage:
    # DDP run
    python scripts/ddp_fsdp_comparison/compare_ddp_fsdp.py launch ddp_tiny_test ai2/saturn-cirrascale --dp_type=ddp

    # FSDP run
    python scripts/ddp_fsdp_comparison/compare_ddp_fsdp.py launch fsdp_tiny_test ai2/saturn-cirrascale --dp_type=fsdp
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
    CheckpointerCallback,
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
from olmoearth_pretrain.internal.common import (
    build_common_components as build_common_components_default,
)
from olmoearth_pretrain.internal.experiment import (
    CommonComponents,
    OlmoEarthVisualizeConfig,
    SubCmd,
    main,
)
from olmoearth_pretrain.internal.utils import MODEL_SIZE_ARGS
from olmoearth_pretrain.nn.flexivit import EncoderConfig, PoolingType, PredictorConfig
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.callbacks import (
    DownstreamEvaluatorCallbackConfig,
    OlmoEarthSpeedMonitorCallback,
    OlmoEarthWandBCallback,
)
from olmoearth_pretrain.train.callbacks.evaluator_callback import DownstreamTaskConfig
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import MaskingConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

MAX_PATCH_SIZE = 8
MIN_PATCH_SIZE = 1

# Global to store dp_type from command line
_DP_TYPE: DataParallelType = DataParallelType.fsdp


def extract_dp_type_from_overrides(
    overrides: list[str],
) -> tuple[DataParallelType, list[str]]:
    """Extract dp_type from overrides and return remaining overrides."""
    dp_type = DataParallelType.fsdp  # default
    remaining = []
    for override in overrides:
        if override.startswith("--dp_type="):
            dp_type_str = override.split("=")[1].lower()
            if dp_type_str == "ddp":
                dp_type = DataParallelType.ddp
            elif dp_type_str == "fsdp":
                dp_type = DataParallelType.fsdp
            else:
                raise ValueError(
                    f"Unknown dp_type: {dp_type_str}. Use 'ddp' or 'fsdp'."
                )
        else:
            remaining.append(override)
    return dp_type, remaining


def build_common_components(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """Build the common components for an experiment."""
    global _DP_TYPE
    _DP_TYPE, overrides[:] = extract_dp_type_from_overrides(overrides)

    # Add dp_type to run name for clarity
    dp_suffix = "ddp" if _DP_TYPE == DataParallelType.ddp else "fsdp"
    run_name_with_suffix = f"{run_name}_{dp_suffix}"

    config = build_common_components_default(
        script, cmd, run_name_with_suffix, cluster, overrides
    )
    # Use simple modality set for faster iteration
    config.training_modalities = [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
    ]
    return config


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build a tiny model config for quick comparison."""
    model_size = MODEL_SIZE_ARGS["tiny"]  # Use tiny for fast iteration

    encoder_config = EncoderConfig(
        embedding_size=model_size["encoder_embedding_size"],
        num_heads=model_size["encoder_num_heads"],
        depth=model_size["encoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        supported_modality_names=common.training_modalities,
        max_patch_size=MAX_PATCH_SIZE,
        drop_path=0.0,  # Disable drop path for determinism
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


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """Build the train module config for comparison experiment."""
    global _DP_TYPE

    # Build dp_config based on selected type
    if _DP_TYPE == DataParallelType.ddp:
        dp_config = DataParallelConfig(
            name=DataParallelType.ddp,
        )
        logger.info("Using DDP for data parallelism")
    else:
        dp_config = DataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        )
        logger.info("Using FSDP for data parallelism")

    return LatentMIMTrainModuleConfig(
        optim_config=AdamWConfig(lr=0.0001, weight_decay=0.02, fused=False),
        rank_microbatch_size=16,  # Smaller batch for 4 GPU
        masking_config=MaskingConfig(
            strategy_config={
                "type": "random",  # Simple random masking for comparison
                "encode_ratio": 0.5,
                "decode_ratio": 0.5,
            }
        ),
        loss_config=LossConfig(
            loss_config={
                "type": "patch_discrimination_new",
                "tau": 0.1,
            }
        ),
        token_exit_cfg={modality: 0 for modality in common.training_modalities},
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=100),
        ema_decay=(1.0, 1.0),  # Disable EMA to isolate DDP vs FSDP difference
        dp_config=dp_config,
        autocast_precision=DType.bfloat16,  # Use same precision for both
    )


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """Build the dataloader config for an experiment."""
    return OlmoEarthDataLoaderConfig(
        num_workers=4,
        global_batch_size=64,  # 16 per GPU * 4 GPUs
        token_budget=1500,
        prefetch_factor=2,
        sampled_hw_p_list=list(range(4, 10)),  # Smaller range for speed
        min_patch_size=MIN_PATCH_SIZE,
        max_patch_size=MAX_PATCH_SIZE,
        work_dir=common.save_folder,
        seed=42,  # Fixed seed for reproducibility
    )


def build_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """Build the dataset config for an experiment."""
    return OlmoEarthDatasetConfig(
        h5py_dir="/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_zstd_3_128_x_4/cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_worldpop_wri_canopy_height_map/1138828",
        training_modalities=common.training_modalities,
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """Build the trainer config for an experiment."""
    global _DP_TYPE
    dp_suffix = "ddp" if _DP_TYPE == DataParallelType.ddp else "fsdp"

    MAX_DURATION = Duration.steps(500)  # Short run for comparison
    METRICS_COLLECT_INTERVAL = 1  # Log every step for detailed comparison
    CANCEL_CHECK_INTERVAL = 25
    LOAD_STRATEGY = LoadStrategy.never  # Always start fresh for fair comparison
    WANDB_PROJECT = "ddp_vs_fsdp_comparison"

    checkpointer_config = CheckpointerConfig(work_dir=common.save_folder)
    wandb_callback = OlmoEarthWandBCallback(
        name=common.run_name,
        project=WANDB_PROJECT,
        entity="eai-ai2",
        enabled=True,
        tags=[dp_suffix, "comparison"],
    )
    garbage_collector_callback = GarbageCollectorCallback(gc_interval=10)

    # Simple downstream eval for comparison
    EVAL_TASKS = {
        "m-eurosat": DownstreamTaskConfig(
            dataset="m-eurosat",
            embedding_batch_size=128,
            num_workers=0,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            eval_interval=Duration.steps(250),  # Eval at 250 and 500
        ),
    }

    trainer_config = (
        TrainerConfig(
            work_dir=common.save_folder,
            load_strategy=LOAD_STRATEGY,
            save_folder=common.save_folder,
            cancel_check_interval=CANCEL_CHECK_INTERVAL,
            metrics_collect_interval=METRICS_COLLECT_INTERVAL,
            max_duration=MAX_DURATION,
            checkpointer=checkpointer_config,
        )
        .with_callback("wandb", wandb_callback)
        .with_callback("speed_monitor", OlmoEarthSpeedMonitorCallback())
        .with_callback("gpu_memory_monitor", GPUMemoryMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(tasks=EVAL_TASKS),
        )
        .with_callback("garbage_collector", garbage_collector_callback)
        .with_callback("beaker", BeakerCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=500,  # Save at end
                ephemeral_save_interval=250,
            ),
        )
    )
    return trainer_config


def build_visualize_config(common: CommonComponents) -> OlmoEarthVisualizeConfig:
    """Build the visualize config for an experiment."""
    return OlmoEarthVisualizeConfig(
        num_samples=None,
        output_dir=str(f"{common.save_folder}/visualizations"),
        std_multiplier=2.0,
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

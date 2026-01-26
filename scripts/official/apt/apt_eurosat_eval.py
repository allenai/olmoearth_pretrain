"""APT (Adaptive Patch Transformers) evaluation on EuroSAT.

This script loads a pretrained model (from base.py or similar) and evaluates
on EuroSAT with APT adaptive patchification.

Reference: "Accelerating Vision Transformers with Adaptive Patch Sizes" (arXiv:2510.18091)

Usage:
    # Local evaluation with a pretrained checkpoint
    python scripts/official/apt/apt_eurosat_eval.py evaluate \
        --run-name apt_eurosat_eval \
        --cluster local \
        --trainer.load_path=/path/to/checkpoint

    # Or launch on Beaker
    python scripts/official/apt/apt_eurosat_eval.py launch \
        --run-name apt_eurosat_eval \
        --cluster ai2/saturn-cirrascale \
        --trainer.load_path=/weka/dfive-default/helios/checkpoints/.../step50000
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
from olmoearth_pretrain.nn.apt.config import APTConfig
from olmoearth_pretrain.nn.flexi_vit import (
    EncoderConfig,
    PoolingType,
    PredictorConfig,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.callbacks import (
    DownstreamEvaluatorCallbackConfig,
    OlmoEarthSpeedMonitorCallback,
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

# Model configuration - match base.py
MAX_PATCH_SIZE = 8
MIN_PATCH_SIZE = 1


def build_apt_config() -> APTConfig:
    """Build APT configuration for S2 optical imagery."""
    return APTConfig.default_s2_config()


def build_common_components(
    script: str,
    cmd: SubCmd,
    run_name: str,
    cluster: str,
    overrides: list[str],
) -> CommonComponents:
    """Build the common components for an experiment."""
    config = build_common_components_default(script, cmd, run_name, cluster, overrides)

    # Match the modalities from base.py for checkpoint compatibility
    config.training_modalities = [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
        Modality.LANDSAT.name,
        Modality.WORLDCOVER.name,
        Modality.SRTM.name,
        Modality.OPENSTREETMAP_RASTER.name,
        Modality.WRI_CANOPY_HEIGHT_MAP.name,
        Modality.CDL.name,
        Modality.WORLDCEREAL.name,
    ]

    return config


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model config - must match the pretrained checkpoint."""
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


def build_train_module_config(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """Build the train module config.

    For evaluation, we don't actually train, but we need a valid config
    to load the checkpoint.
    """
    return ContrastiveLatentMIMTrainModuleConfig(
        optim_config=AdamWConfig(lr=1e-6, weight_decay=0.02, fused=False),
        rank_microbatch_size=32,
        masking_config=MaskingConfig(
            strategy_config={
                "type": "random",
                "encode_ratio": 0.5,
                "decode_ratio": 0.5,
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
        scheduler=CosWithWarmup(warmup_steps=100),
        ema_decay=(1.0, 1.0),
        dp_config=DataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
    )


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """Build the dataloader config."""
    return OlmoEarthDataLoaderConfig(
        num_workers=4,  # Fewer workers for local runs
        global_batch_size=32,  # Smaller batch for local
        token_budget=2250,
        prefetch_factor=2,
        sampled_hw_p_list=list(range(1, 13)),
        min_patch_size=MIN_PATCH_SIZE,
        max_patch_size=MAX_PATCH_SIZE,
        work_dir=common.save_folder,
        seed=3622,
    )


def build_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """Build the dataset config.

    For evaluation, we use a minimal dataset since we're just running
    downstream eval on EuroSAT.
    """
    return OlmoEarthDatasetConfig(
        h5py_dir="/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_zstd_3_128_x_4/cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_worldpop_wri_canopy_height_map/1138828",
        training_modalities=common.training_modalities,
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """Build the trainer config for evaluation."""
    # Short run - just enough to trigger eval callbacks
    MAX_DURATION = Duration.steps(10)
    METRICS_COLLECT_INTERVAL = 5
    CANCEL_CHECK_INTERVAL = 10

    # Always try to load from checkpoint
    LOAD_STRATEGY = LoadStrategy.always

    WANDB_USERNAME = "eai-ai2"  # nosec
    WANDB_PROJECT = "01_2026_apt_investigation"

    checkpointer_config = CheckpointerConfig(work_dir=common.save_folder)
    wandb_callback = OlmoEarthWandBCallback(
        name=common.run_name,
        project=WANDB_PROJECT,
        entity=WANDB_USERNAME,
        enabled=True,
    )
    garbage_collector_callback = GarbageCollectorCallback(gc_interval=1)

    # EuroSAT finetuning evaluation
    EVAL_TASKS = {
        "m-eurosat-finetune": DownstreamTaskConfig(
            dataset="m-eurosat",
            embedding_batch_size=128,
            num_workers=4,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            eval_mode=EvalMode.FINETUNE,
            ft_lr=1e-4,  # Finetune learning rate
            ft_batch_size=32,
            epochs=50,
            eval_interval=Duration.steps(1),
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
            DownstreamEvaluatorCallbackConfig(
                tasks=EVAL_TASKS,
                eval_on_startup=True,  # Run eval immediately on load
                cancel_after_first_eval=True,  # Exit after eval completes
            ),
        )
        .with_callback("garbage_collector", garbage_collector_callback)
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=10000,  # Don't save during eval
                ephemeral_save_interval=10000,
            ),
        )
    )
    return trainer_config


def build_visualize_config(common: CommonComponents) -> OlmoEarthVisualizeConfig:
    """Build the visualize config."""
    return OlmoEarthVisualizeConfig(
        num_samples=None,
        output_dir=f"{common.save_folder}/visualizations",
        std_multiplier=2.0,
    )


if __name__ == "__main__":
    # Log APT configuration
    apt_config = build_apt_config()
    logger.info(f"APT Configuration: {apt_config}")

    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )

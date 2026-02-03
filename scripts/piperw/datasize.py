"""Script for training models with configurable model size, dataset size, and batch size.

This script allows you to specify:
- model_type: one of 'nano', 'tiny', 'base', 'large'
- num_train_samples: number of training samples to use
- global_batch_size: global batch size for training

Usage:
    python scripts/piperw/datasize.py launch RUN_NAME CLUSTER --model_type=nano --num_train_samples=500 --global_batch_size=256
    python scripts/piperw/datasize.py train RUN_NAME CLUSTER --model_type=base --num_train_samples=1000 --global_batch_size=1024
"""

import logging
import sys

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
    SubCmd,
)
from olmoearth_pretrain.internal.utils import MODEL_SIZE_ARGS
from olmoearth_pretrain.nn.flexi_vit import PoolingType
from olmoearth_pretrain.nn.flexihelios import (
    EncoderConfig,
    PredictorConfig,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.callbacks import (
    DownstreamEvaluatorCallbackConfig,
    OlmoEarthSpeedMonitorCallback,
    OlmoEarthWandBCallback,
)
from olmoearth_pretrain.train.callbacks.evaluator_callback import DownstreamTaskConfig
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import MaskingConfig
from olmoearth_pretrain.train.train_module.contrastive_latentmim import (
    ContrastiveLatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

MAX_PATCH_SIZE = 8
MIN_PATCH_SIZE = 1

# Module-level variables to store parsed arguments
_model_type: str | None = None
_num_train_samples: int | None = None
_global_batch_size: int | None = None


def parse_args() -> tuple[str | None, int | None, int | None]:
    """Parse model_type, num_train_samples, and global_batch_size from sys.argv."""
    model_type = None
    num_train_samples = None
    global_batch_size = None
    
    for i, arg in enumerate(sys.argv):
        if arg == "--model_type" and i + 1 < len(sys.argv):
            model_type = sys.argv[i + 1]
        elif arg.startswith("--model_type="):
            model_type = arg.split("=", 1)[1]
        elif arg == "--num_train_samples" and i + 1 < len(sys.argv):
            num_train_samples = int(sys.argv[i + 1])
        elif arg.startswith("--num_train_samples="):
            num_train_samples = int(arg.split("=", 1)[1])
        elif arg == "--global_batch_size" and i + 1 < len(sys.argv):
            global_batch_size = int(sys.argv[i + 1])
        elif arg.startswith("--global_batch_size="):
            global_batch_size = int(arg.split("=", 1)[1])
    
    # Validate model_type if provided
    if model_type is not None and model_type not in ["nano", "tiny", "base", "large"]:
        raise ValueError(
            f"Invalid model_type: {model_type}. Must be one of: nano, tiny, base, large"
        )
    
    # Validate global_batch_size if provided
    if global_batch_size is not None and global_batch_size <= 0:
        raise ValueError(
            f"Invalid global_batch_size: {global_batch_size}. Must be positive"
        )
    
    return model_type, num_train_samples, global_batch_size


def get_model_size_key(model_type: str) -> str:
    """Map model type to MODEL_SIZE_ARGS key."""
    mapping = {
        "nano": "nano",
        "tiny": "tiny_shallow_decoder",
        "base": "base_shallow_decoder",
        "large": "large_shallow_decoder",
    }
    if model_type not in mapping:
        raise ValueError(
            f"Invalid model_type: {model_type}. Must be one of {list(mapping.keys())}"
        )
    return mapping[model_type]


def build_common_components(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """Build the common components for an experiment."""
    config = build_common_components_default(script, cmd, run_name, cluster, overrides)
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
    """Build the model config for an experiment."""
    global _model_type
    
    # Get model_type from module variable or default to "nano"
    model_type = _model_type or "nano"
    model_size_key = get_model_size_key(model_type)
    model_size = MODEL_SIZE_ARGS[model_size_key]
    
    logger.info(f"Using model type: {model_type} (key: {model_size_key})")

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


def build_train_module_config(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """Build the train module config for an experiment."""
    return ContrastiveLatentMIMTrainModuleConfig(
        optim_config=AdamWConfig(lr=0.0001, weight_decay=0.02, fused=False),
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


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """Build the dataloader config for an experiment."""
    global _global_batch_size
    
    # Use provided global_batch_size or default to 512
    global_batch_size = _global_batch_size if _global_batch_size is not None else 512
    
    if _global_batch_size is not None:
        logger.info(f"Using global_batch_size: {global_batch_size}")
    
    return OlmoEarthDataLoaderConfig(
        num_workers=16,
        global_batch_size=global_batch_size,
        token_budget=2250,
        prefetch_factor=4,
        sampled_hw_p_list=list(range(1, 13)),  # try only temporal tokens
        min_patch_size=MIN_PATCH_SIZE,
        max_patch_size=MAX_PATCH_SIZE,
        work_dir=common.save_folder,
        seed=3622,
    )


def build_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """Build the dataset config for an experiment."""
    global _num_train_samples
    
    config = OlmoEarthDatasetConfig(
        h5py_dir="/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_zstd_3_128_x_4/cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_worldpop_wri_canopy_height_map/1138828",
        training_modalities=common.training_modalities,
    )
    
    # Set max_training_samples if provided
    if _num_train_samples is not None:
        config.max_training_samples = _num_train_samples
        config.seed = 3622  # For reproducible random sampling
        logger.info(f"Limiting dataset to {_num_train_samples} training samples")
    
    return config


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """Build the trainer config for an experiment."""
    MAX_DURATION = Duration.epochs(300)
    METRICS_COLLECT_INTERVAL = 10
    CANCEL_CHECK_INTERVAL = 25
    LOAD_STRATEGY = LoadStrategy.if_available
    WANDB_USERNAME = "eai-ai2"  # nosec
    WANDB_PROJECT = "2025_10_02_phase2"
    PERMANENT_SAVE_INTERVAL = 5000
    EPHERMERAL_SAVE_INTERVAL = 250
    checkpointer_config = CheckpointerConfig(work_dir=common.save_folder)
    wandb_callback = OlmoEarthWandBCallback(
        name=common.run_name,
        project=WANDB_PROJECT,
        entity=WANDB_USERNAME,
        enabled=True,  # set to False to avoid wandb errors
    )
    # Safe to collect every step for now
    garbage_collector_callback = GarbageCollectorCallback(gc_interval=1)
    EVAL_TASKS = {
        "m-eurosat": DownstreamTaskConfig(
            dataset="m-eurosat",
            embedding_batch_size=128,
            num_workers=0,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            eval_interval=Duration.steps(4000),
        ),
        "m_so2sat": DownstreamTaskConfig(
            dataset="m-so2sat",
            embedding_batch_size=128,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            eval_interval=Duration.steps(20000),
        ),
        "mados": DownstreamTaskConfig(
            dataset="mados",
            embedding_batch_size=128,
            probe_batch_size=128,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=False,
            probe_lr=0.01,
            epochs=50,
            eval_interval=Duration.steps(4000),
        ),
        "pastis": DownstreamTaskConfig(
            dataset="pastis",
            embedding_batch_size=32,
            probe_batch_size=8,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            probe_lr=0.1,
            eval_interval=Duration.steps(20000),
            input_modalities=[Modality.SENTINEL2_L2A.name],
            epochs=50,
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
            ),
        )
        .with_callback("garbage_collector", garbage_collector_callback)
        .with_callback(
            "beaker", BeakerCallback()
        )  # this should not be here, but for now it is
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=PERMANENT_SAVE_INTERVAL,
                ephemeral_save_interval=EPHERMERAL_SAVE_INTERVAL,
            ),
        )
    )
    return trainer_config


if __name__ == "__main__":
    from olmoearth_pretrain.internal.experiment import main
    
    # Parse arguments before calling main
    _model_type, _num_train_samples, _global_batch_size = parse_args()
    
    if _model_type is None:
        logger.warning("--model_type not specified, defaulting to 'nano'")
    if _num_train_samples is None:
        logger.warning("--num_train_samples not specified, using full dataset")
    if _global_batch_size is None:
        logger.info("--global_batch_size not specified, using default: 512")
    
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
    )


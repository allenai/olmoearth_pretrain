"""Script for Debugging Galileo.

These Settings are meant to help you get quick results on a single GPU in minimal time
"""

import logging
from typing import Any

from olmo_core.config import Config, DType
from olmo_core.distributed.parallel.data_parallel import (
    DataParallelConfig,
    DataParallelType,
)
from olmo_core.optim import AdamWConfig
from olmo_core.optim.scheduler import CosWithWarmup
from olmo_core.train.callbacks import (
    ConfigSaverCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
)
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.common import Duration, LoadStrategy
from olmo_core.train.config import TrainerConfig
from upath import UPath

from helios.data.concat import HeliosConcatDatasetConfig
from helios.data.constants import Modality
from helios.data.dataloader import HeliosDataLoaderConfig
from helios.data.dataset import HeliosDatasetConfig
from helios.internal.common import build_common_components
from helios.internal.experiment import CommonComponents, HeliosVisualizeConfig
from helios.internal.utils import MODEL_SIZE_ARGS
from helios.nn.flexihelios import PoolingType
from helios.nn.galileo import GalileoConfig
from helios.nn.st_model import STEncoderConfig, STPredictorConfig
from helios.train.callbacks import (
    DownstreamEvaluatorCallbackConfig,
    HeliosSpeedMonitorCallback,
    HeliosWandBCallback,
)
from helios.train.callbacks.evaluator_callback import DownstreamTaskConfig
from helios.train.loss import LossConfig
from helios.train.masking import MaskingConfig
from helios.train.train_module.galileo import GalileoTrainModuleConfig

logger = logging.getLogger(__name__)

MAX_PATCH_SIZE = 8  # NOTE: actual patch_size <= max_patch_size
MIN_PATCH_SIZE = 1
NUM_DATA_LOADER_WORKERS = 8


def build_model_config(common: CommonComponents) -> GalileoConfig:
    """Build the model config for an experiment."""
    base_model_args = MODEL_SIZE_ARGS["base"]
    ENCODER_EMBEDDING_SIZE = int(base_model_args["encoder_embedding_size"])
    DECODER_EMBEDDING_SIZE = int(base_model_args["decoder_embedding_size"])
    ENCODER_DEPTH = int(base_model_args["encoder_depth"])
    DECODER_DEPTH = 8
    ENCODER_NUM_HEADS = int(base_model_args["encoder_num_heads"])
    DECODER_NUM_HEADS = int(base_model_args["decoder_num_heads"])
    MLP_RATIO = float(base_model_args["mlp_ratio"])

    encoder_config = STEncoderConfig(
        supported_modality_names=common.training_modalities,
        embedding_size=ENCODER_EMBEDDING_SIZE,
        max_patch_size=MAX_PATCH_SIZE,
        num_heads=ENCODER_NUM_HEADS,
        depth=ENCODER_DEPTH,
        mlp_ratio=MLP_RATIO,
        drop_path=0.1,
        max_sequence_length=12,
        use_channel_embs=True,
    )
    decoder_config = STPredictorConfig(
        encoder_embedding_size=ENCODER_EMBEDDING_SIZE,
        decoder_embedding_size=DECODER_EMBEDDING_SIZE,
        depth=DECODER_DEPTH,
        mlp_ratio=MLP_RATIO,
        num_heads=DECODER_NUM_HEADS,
        max_sequence_length=12,
        supported_modality_names=common.training_modalities,
        learnable_channel_embeddings=True,
    )
    model_config = GalileoConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
    )
    return model_config


def build_train_module_config(
    common: CommonComponents,
) -> GalileoTrainModuleConfig:
    """Build the train module config for an experiment."""
    LR = 0.0001
    RANK_MICROBATCH_SIZE = 32
    ENCODE_RATIO = 0.1
    DECODE_RATIO = 0.75
    WD = 0.02
    optim_config = AdamWConfig(lr=LR, weight_decay=WD)
    masking_config_a = MaskingConfig(
        strategy_config={
            "type": "space_time",
            "encode_ratio": ENCODE_RATIO,
            "decode_ratio": DECODE_RATIO,
        }
    )
    masking_config_b = MaskingConfig(
        strategy_config={
            "type": "random_increasing",
            "initial_encode_ratio": 0.6,
            "initial_decode_ratio": 0.35,
            "final_encode_ratio": 0.1,
            "final_decode_ratio": 0.75,
            "steps": 1000,
        }
    )
    loss_config_a = LossConfig(
        loss_config={
            "type": "patch_discrimination_new",
        }
    )
    loss_config_b = LossConfig(
        loss_config={
            "type": "patch_discrimination_new",
        }
    )
    contrastive_config = LossConfig(
        loss_config={
            "type": "InfoNCE",
            "weight": 0.05,
        }
    )
    # Maybe we want to increase token exit for base model?
    base_model_args = MODEL_SIZE_ARGS["base"]
    token_exit_cfg_a = {
        Modality.SENTINEL2_L2A.name: int(base_model_args["encoder_depth"]),
        Modality.LATLON.name: int(base_model_args["encoder_depth"]),
        Modality.SENTINEL1.name: int(base_model_args["encoder_depth"]),
        Modality.WORLDCOVER.name: 0,
        Modality.SRTM.name: int(base_model_args["encoder_depth"]),
        Modality.OPENSTREETMAP_RASTER.name: 0,
        Modality.LANDSAT.name: int(base_model_args["encoder_depth"]),
    }
    if any(modality not in token_exit_cfg_a for modality in common.training_modalities):
        raise ValueError(
            f"All modalities must be in token_exit_cfg_a: {common.training_modalities}"
        )
    token_exit_cfg_b = {modality: 0 for modality in common.training_modalities}
    WARMUP_EPOCHS = 5
    dp_config = DataParallelConfig(
        name=DataParallelType.fsdp,
        param_dtype=DType.bfloat16,
        reduce_dtype=DType.float32,
    )

    # TODO: would need a scheduler config and registry to be able to change this with overrides
    scheduler = CosWithWarmup()
    train_module_config = GalileoTrainModuleConfig(
        # TODO: change name to optim config
        optim_config=optim_config,
        warmup_duration=Duration.epochs(WARMUP_EPOCHS),
        masking_config_a=masking_config_a,
        masking_config_b=masking_config_b,
        loss_config_a=loss_config_a,
        loss_config_b=loss_config_b,
        contrastive_config=contrastive_config,
        rank_microbatch_size=RANK_MICROBATCH_SIZE,
        token_exit_cfg_a=token_exit_cfg_a,
        token_exit_cfg_b=token_exit_cfg_b,
        autocast_precision=DType.bfloat16,
        max_grad_norm=1.0,
        dp_config=dp_config,
        scheduler=scheduler,
    )
    return train_module_config


def build_dataloader_config(common: CommonComponents) -> HeliosDataLoaderConfig:
    """Build the dataloader config for an experiment."""
    # things should be set during building
    # TODO: Include collate function here

    NUM_WORKERS = 8
    GLOBAL_BATCH_SIZE = 512
    TOKEN_BUDGET = 1500
    SAMPLE_HW_P_LIST = list(range(5, 13))

    dataloader_config = HeliosDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,
        min_patch_size=MIN_PATCH_SIZE,
        max_patch_size=MAX_PATCH_SIZE,
        seed=3622,
        work_dir=common.save_folder,
        num_workers=NUM_WORKERS,
        sampled_hw_p_list=SAMPLE_HW_P_LIST,
        token_budget=TOKEN_BUDGET,
    )
    # Should the dataloader build the config or take an object?
    return dataloader_config


def build_dataset_config(common: CommonComponents) -> Config:
    """Build the dataset config for an experiment."""
    dataset_configs = [
        HeliosDatasetConfig(
            h5py_dir="/weka/dfive-default/helios/dataset/presto/h5py_data/sentinel1_sentinel2_l2a_worldcover/116711",
            training_modalities=common.training_modalities,
            use_samples_with_missing_supported_modalities=False,
            dtype=DType.float32,
        ),
        HeliosDatasetConfig(
            h5py_dir="/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_rerun/sentinel1_sentinel2_l2a_worldcover/283204",
            training_modalities=common.training_modalities,
            use_samples_with_missing_supported_modalities=False,
            dtype=DType.float32,
        ),
        HeliosDatasetConfig(
            h5py_dir="/weka/dfive-default/helios/dataset/presto_neighbor/h5py_data/sentinel1_sentinel2_l2a_worldcover/871285",
            training_modalities=common.training_modalities,
            use_samples_with_missing_supported_modalities=False,
            dtype=DType.float32,
        ),
    ]
    return HeliosConcatDatasetConfig(dataset_configs=dataset_configs)


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """Build the trainer config for an experiment."""
    MAX_DURATION = Duration.epochs(150)
    METRICS_COLLECT_INTERVAL = 10
    CANCEL_CHECK_INTERVAL = 25
    LOAD_STRATEGY = LoadStrategy.if_available
    WANDB_USERNAME = "eai-ai2"  # nosec
    WANDB_PROJECT = "2025_05_05_bigdata"
    checkpointer_config = CheckpointerConfig(work_dir=common.save_folder)
    wandb_callback = HeliosWandBCallback(
        name=common.run_name,
        project=WANDB_PROJECT,
        entity=WANDB_USERNAME,
        enabled=True,  # set to False to avoid wandb errors
    )
    # Safe to collect everys tep for now
    garbage_collector_callback = GarbageCollectorCallback(gc_interval=1)
    logger.warning("WANDB Distribution Uploads are disabled for Debugging")
    EVAL_TASKS = {
        "m-eurosat": DownstreamTaskConfig(
            dataset="m-eurosat",
            batch_size=128,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            eval_interval=Duration.epochs(2),
        ),
        "mados": DownstreamTaskConfig(
            dataset="mados",
            batch_size=128,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=False,
            probe_lr=0.01,
            eval_interval=Duration.epochs(6),
        ),
        "sen1floods11": DownstreamTaskConfig(
            dataset="sen1floods11",
            batch_size=128,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            probe_lr=0.01,
            eval_interval=Duration.epochs(6),
        ),
        "pastis": DownstreamTaskConfig(
            dataset="pastis",
            batch_size=8,
            num_workers=2,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            probe_lr=0.01,
            eval_interval=Duration.epochs(6),
        ),
        "pastis-r": DownstreamTaskConfig(
            dataset="pastis-r",
            batch_size=8,
            num_workers=2,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            probe_lr=0.01,
            eval_interval=Duration.epochs(6),
        ),
    }
    # Let us not use garbage collector fallback
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
        .with_callback("speed_monitor", HeliosSpeedMonitorCallback())
        .with_callback("gpu_memory_monitor", GPUMemoryMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=EVAL_TASKS,
            ),
        )
        .with_callback("garbage_collector", garbage_collector_callback)
    )
    return trainer_config


def build_visualize_config(common: CommonComponents) -> HeliosVisualizeConfig:
    """Build the visualize config for an experiment."""
    return HeliosVisualizeConfig(
        num_samples=50,
        output_dir=str(UPath(common.save_folder) / "visualizations"),
        std_multiplier=2.0,
    )


def build_common_components_limited_modalities(*args: Any) -> CommonComponents:
    """Build the common components for an experiment."""
    config = build_common_components(*args)
    config.training_modalities = [
        Modality.SENTINEL1.name,
        Modality.SENTINEL2_L2A.name,
        Modality.WORLDCOVER.name,
    ]
    return config

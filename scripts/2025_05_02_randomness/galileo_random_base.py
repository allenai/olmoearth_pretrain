"""This script is aimed to be the base script for trying to get maximum throughput on a galileo style model."""

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
    BeakerCallback,
    CheckpointerCallback,
    ConfigSaverCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
)
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.common import Duration, LoadStrategy
from olmo_core.train.config import TrainerConfig

from helios.data.concat import HeliosConcatDatasetConfig
from helios.data.constants import Modality
from helios.data.dataloader import HeliosDataLoaderConfig
from helios.data.dataset import HeliosDatasetConfig
from helios.internal.common import build_common_components, build_visualize_config
from helios.internal.experiment import CommonComponents, main
from helios.internal.utils import MODEL_SIZE_ARGS
from helios.nn.flexihelios import EncoderConfig, PoolingType, PredictorConfig
from helios.nn.galileo import GalileoConfig
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
NUM_WORKERS = 8

base_model_args = MODEL_SIZE_ARGS["base_super_shallow_decoder"]


def build_model_config(common: CommonComponents) -> GalileoConfig:
    """Build the model config for an experiment."""
    ENCODER_EMBEDDING_SIZE = int(base_model_args["encoder_embedding_size"])
    ENCODER_EMBEDDING_SIZE = int(base_model_args["encoder_embedding_size"])
    DECODER_EMBEDDING_SIZE = int(base_model_args["decoder_embedding_size"])
    ENCODER_DEPTH = int(base_model_args["encoder_depth"])
    DECODER_DEPTH = int(base_model_args["decoder_depth"])
    ENCODER_NUM_HEADS = int(base_model_args["encoder_num_heads"])
    DECODER_NUM_HEADS = int(base_model_args["decoder_num_heads"])
    MLP_RATIO = float(base_model_args["mlp_ratio"])

    encoder_config = EncoderConfig(
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
    decoder_config = PredictorConfig(
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
            "type": "random",
            "encode_ratio": ENCODE_RATIO,
            "decode_ratio": DECODE_RATIO,
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
    token_exit_cfg_a = {
        Modality.SENTINEL2_L2A.name: int(base_model_args["encoder_depth"]),
        Modality.LATLON.name: int(base_model_args["encoder_depth"]),
        Modality.SENTINEL1.name: int(base_model_args["encoder_depth"]),
        Modality.WORLDCOVER.name: 0,
        # galileo may vary this
        Modality.SRTM.name: int(base_model_args["encoder_depth"]),
        Modality.OPENSTREETMAP_RASTER.name: 0,
        Modality.LANDSAT.name: int(base_model_args["encoder_depth"]),
    }
    if any(modality not in token_exit_cfg_a for modality in common.training_modalities):
        raise ValueError(
            f"All modalities must be in token_exit_cfg_a: {common.training_modalities}"
        )
    token_exit_cfg_b = {modality: 0 for modality in common.training_modalities}
    WARMUP_EPOCHS = 20
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
        rank_microbatch_size=RANK_MICROBATCH_SIZE,
        token_exit_cfg_a=token_exit_cfg_a,
        token_exit_cfg_b=token_exit_cfg_b,
        autocast_precision=DType.bfloat16,  # how does this interact with the fsdp?
        compile_model=True,
        max_grad_norm=1.0,
        dp_config=dp_config,
        scheduler=scheduler,
    )
    return train_module_config


def build_dataloader_config(common: CommonComponents) -> HeliosDataLoaderConfig:
    """Build the dataloader config for an experiment."""
    GLOBAL_BATCH_SIZE = 512
    PREFETCH_FACTOR = 4
    TOKEN_BUDGET = 1500
    SAMPLE_HW_P_LIST = list(range(5, 13))
    # GBS * PREFETCH_FACTOR * NUM_WORKERS is the total number of instances that can be put into prefetch queue

    dataloader_config = HeliosDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,
        min_patch_size=MIN_PATCH_SIZE,
        max_patch_size=MAX_PATCH_SIZE,
        seed=3622,
        work_dir=common.save_folder,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        sampled_hw_p_list=SAMPLE_HW_P_LIST,
        token_budget=TOKEN_BUDGET,
    )
    return dataloader_config


def build_dataset_config(common: CommonComponents) -> Config:
    """Build the dataset config for an experiment."""
    dataset_configs = [
        HeliosDatasetConfig(
            h5py_dir="/weka/dfive-default/helios/dataset/presto/h5py_data_gzip_3_shuffle/landsat_naip_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/118861",
            training_modalities=common.training_modalities,
            use_samples_with_missing_supported_modalities=True,  # Check if we want to set this to True
            dtype=DType.float32,
            cache_dir="/helios_cache/presto",
            samples_per_sec=4 / NUM_WORKERS,  # 2/ GBS
        ),
        HeliosDatasetConfig(
            h5py_dir="/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_gzip_3_shuffle/landsat_naip_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/324192",
            training_modalities=common.training_modalities,
            use_samples_with_missing_supported_modalities=True,
            dtype=DType.float32,
            cache_dir="/helios_cache/osm_sampling",
            samples_per_sec=4 / NUM_WORKERS,  # 2/ GBS
        ),
    ]
    return HeliosConcatDatasetConfig(dataset_configs=dataset_configs)


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """Build the trainer config for an experiment."""
    MAX_DURATION = Duration.epochs(400)
    METRICS_COLLECT_INTERVAL = 10  # SHould be turned off for final run
    CANCEL_CHECK_INTERVAL = 25  # should be turned off for final run
    LOAD_STRATEGY = LoadStrategy.if_available
    WANDB_USERNAME = "eai-ai2"  # nosec
    WANDB_PROJECT = "2025-05-02-random-variation"
    checkpointer_config = CheckpointerConfig(work_dir=common.save_folder)
    wandb_callback = HeliosWandBCallback(
        name=common.run_name,
        project=WANDB_PROJECT,
        entity=WANDB_USERNAME,
        enabled=True,  # set to False to avoid wandb errors
    )
    PERMANENT_SAVE_INTERVAL = 2500
    EPHERMERAL_SAVE_INTERVAL = 250
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
            eval_interval=Duration.epochs(10),
        ),
        "mados": DownstreamTaskConfig(
            dataset="mados",
            batch_size=128,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=False,
            probe_lr=0.01,
            eval_interval=Duration.epochs(20),
        ),
        "sen1floods11": DownstreamTaskConfig(
            dataset="sen1floods11",
            batch_size=128,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            probe_lr=0.1,
            eval_interval=Duration.epochs(20),
        ),
        "pastis": DownstreamTaskConfig(
            dataset="pastis",
            batch_size=8,
            num_workers=2,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            probe_lr=0.1,
            eval_interval=Duration.epochs(20),
        ),
        "pastis-r": DownstreamTaskConfig(
            dataset="pastis-r",
            batch_size=8,
            num_workers=2,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            probe_lr=0.1,
            eval_interval=Duration.epochs(20),
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
        .with_callback("beaker", BeakerCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=PERMANENT_SAVE_INTERVAL,
                ephemeral_save_interval=EPHERMERAL_SAVE_INTERVAL,
            ),
        )
    )
    return trainer_config


def build_common_components_limited_modalities(*args: Any) -> CommonComponents:
    """Build the common components for an experiment."""
    config = build_common_components(*args)
    config.training_modalities = [
        Modality.SENTINEL1.name,
        Modality.SENTINEL2_L2A.name,
        Modality.WORLDCOVER.name,
        #     Modality.LANDSAT.name,
        #     Modality.OPENSTREETMAP_RASTER.name,
        #     Modality.SRTM.name,
    ]
    return config


if __name__ == "__main__":
    main(
        common_components_builder=build_common_components_limited_modalities,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )

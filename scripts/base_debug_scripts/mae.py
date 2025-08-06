"""Trying to prototype fitting everything into olmo core."""

import logging
from typing import Any

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

from helios.data.constants import Modality
from helios.data.dataloader import HeliosDataLoaderConfig
from helios.data.dataset import HeliosDatasetConfig
from helios.internal.common import build_common_components
from helios.internal.experiment import CommonComponents, main
from helios.nn.flexihelios import (
    EncoderConfig,
    PoolingType,
    PredictorConfig,
    ReconstructorConfig,
)
from helios.nn.mae import MAEConfig
from helios.train.callbacks import (
    DownstreamEvaluatorCallbackConfig,
    HeliosSpeedMonitorCallback,
    HeliosWandBCallback,
)
from helios.train.callbacks.evaluator_callback import DownstreamTaskConfig
from helios.train.loss import LossConfig
from helios.train.masking import MaskingConfig
from helios.train.train_module.mae import MAETrainModuleConfig

logger = logging.getLogger(__name__)
MAX_PATCH_SIZE = 8  # NOTE: actual patch_size <= max_patch_size
MIN_PATCH_SIZE = 4
USE_4_X_128_DATASET = False

# Training duration constants
TOTAL_EPOCHS = 300
if USE_4_X_128_DATASET:
    TOTAL_EPOCHS = TOTAL_EPOCHS // 4

MAE_MODALITIES = [
    Modality.SENTINEL2_L2A.name,
    Modality.SENTINEL1.name,
    Modality.WORLDCOVER.name,
]


def build_model_config(common: CommonComponents) -> MAEConfig:
    """Build the model config for an experiment."""
    ENCODER_EMBEDDING_SIZE = 192
    DECODER_EMBEDDING_SIZE = 192
    ENCODER_DEPTH = 12
    DECODER_DEPTH = 12
    ENCODER_NUM_HEADS = 3
    DECODER_NUM_HEADS = 3
    MLP_RATIO = 4.0
    encoder_config = EncoderConfig(
        supported_modality_names=common.training_modalities,
        embedding_size=ENCODER_EMBEDDING_SIZE,
        max_patch_size=MAX_PATCH_SIZE,
        min_patch_size=MIN_PATCH_SIZE,
        num_heads=ENCODER_NUM_HEADS,
        depth=ENCODER_DEPTH,
        mlp_ratio=MLP_RATIO,
        drop_path=0.1,
        max_sequence_length=12,
    )
    decoder_config = PredictorConfig(
        encoder_embedding_size=ENCODER_EMBEDDING_SIZE,
        decoder_embedding_size=DECODER_EMBEDDING_SIZE,
        depth=DECODER_DEPTH,
        mlp_ratio=MLP_RATIO,
        num_heads=DECODER_NUM_HEADS,
        max_sequence_length=12,
        supported_modality_names=common.training_modalities,
    )
    reconstructor_config = ReconstructorConfig(
        supported_modality_names=common.training_modalities,
        max_patch_size=MAX_PATCH_SIZE,
        decoder_config=decoder_config,
    )
    model_config = MAEConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        reconstructor_config=reconstructor_config,
    )
    return model_config


def build_train_module_config(
    common: CommonComponents,
) -> MAETrainModuleConfig:
    """Build the train module config for an experiment."""
    LR = 0.002
    RANK_MICROBATCH_SIZE = 32
    ENCODE_RATIO = 0.1
    DECODE_RATIO = 0.9
    WD = 0.02
    optim_config = AdamWConfig(lr=LR, weight_decay=WD)
    masking_config = MaskingConfig(
        strategy_config={
            "type": "random",
            "encode_ratio": ENCODE_RATIO,
            "decode_ratio": DECODE_RATIO,
        }
    )
    mae_loss_config = LossConfig(
        loss_config={
            "type": "mae",
        }
    )
    token_exit_cfg = {modality: 4 for modality in common.training_modalities}
    dp_config = DataParallelConfig(name=DataParallelType.ddp)

    # TODO: would need a scheduler config and registry to be able to change this with overrides
    scheduler = CosWithWarmup(warmup=8000)
    train_module_config = MAETrainModuleConfig(
        # TODO: change name to optim config
        optim_config=optim_config,
        masking_config=masking_config,
        mae_loss_config=mae_loss_config,
        rank_microbatch_size=RANK_MICROBATCH_SIZE,
        token_exit_cfg=token_exit_cfg,
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
    GLOBAL_BATCH_SIZE = 128
    PREFETCH_FACTOR = 4
    TOKEN_BUDGET = 1500
    SAMPLE_HW_P_LIST = list(range(5, 13))

    dataloader_config = HeliosDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=3622,
        work_dir=common.save_folder,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        sampled_hw_p_list=SAMPLE_HW_P_LIST,
        min_patch_size=MIN_PATCH_SIZE,
        max_patch_size=MAX_PATCH_SIZE,
        token_budget=TOKEN_BUDGET,
    )
    return dataloader_config


def build_dataset_config(common: CommonComponents) -> HeliosDatasetConfig:
    """Build the dataset config for an experiment."""
    if USE_4_X_128_DATASET:
        dataset_path = "/weka/dfive-default/helios/dataset/presto/h5py_data_w_missing_timesteps_128_x_4_zstd_3/landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/469892"
    else:
        dataset_path = "/weka/dfive-default/helios/dataset/presto/h5py_data_w_missing_timesteps_zstd_3/landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/117473/"
    return HeliosDatasetConfig(
        h5py_dir=dataset_path,
        training_modalities=common.training_modalities,
        dtype="float32",
        # cache_dir="/helios_cache/osm_sampling",
        # samples_per_sec=4 / NUM_WORKERS,  # 2/ GBS
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """Build the trainer config for an experiment."""
    MAX_DURATION = Duration.epochs(TOTAL_EPOCHS)
    METRICS_COLLECT_INTERVAL = 1
    CANCEL_CHECK_INTERVAL = 1
    LOAD_STRATEGY = LoadStrategy.if_available
    WANDB_USERNAME = "eai-ai2"  # nosec
    WANDB_PROJECT = "helios-debug"
    checkpointer_config = CheckpointerConfig(work_dir=common.save_folder)
    wandb_callback = HeliosWandBCallback(
        name=common.run_name,
        project=WANDB_PROJECT,
        entity=WANDB_USERNAME,
        enabled=True,  # set to False to avoid wandb errors
    )
    garbage_collector_callback = GarbageCollectorCallback(gc_interval=1)
    EVAL_TASKS = {
        "m-eurosat": DownstreamTaskConfig(
            dataset="m-eurosat",
            embedding_batch_size=128,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            eval_interval=Duration.epochs(5),
        ),
        "breizhcrops": DownstreamTaskConfig(
            dataset="breizhcrops",
            embedding_batch_size=128,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            eval_interval=Duration.epochs(50),
            patch_size=1,
        ),
        "pastis": DownstreamTaskConfig(
            dataset="pastis",
            embedding_batch_size=32,
            probe_batch_size=128,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            probe_lr=0.1,
            eval_interval=Duration.epochs(50),
            input_modalities=[Modality.SENTINEL2_L2A.name],
            epochs=50,
        ),
        "pastis_sentinel1": DownstreamTaskConfig(
            dataset="pastis",
            embedding_batch_size=32,
            probe_batch_size=16,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            probe_lr=0.1,
            eval_interval=Duration.epochs(50),
            input_modalities=[Modality.SENTINEL1.name],
            epochs=50,
        ),
        "sickle_sentinel1": DownstreamTaskConfig(
            dataset="sickle",
            embedding_batch_size=32,
            probe_batch_size=16,
            num_workers=2,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            probe_lr=0.0004,
            eval_interval=Duration.epochs(10),
            input_modalities=[Modality.SENTINEL1.name],
            epochs=50,
        ),
        "sickle_landsat": DownstreamTaskConfig(
            dataset="sickle",
            embedding_batch_size=32,
            probe_batch_size=16,
            num_workers=2,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            probe_lr=0.01,
            eval_interval=Duration.epochs(10),
            input_modalities=[Modality.LANDSAT.name],
            epochs=50,
        ),
        "pastis_r": DownstreamTaskConfig(
            dataset="pastis",
            embedding_batch_size=32,
            probe_batch_size=128,
            num_workers=2,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            probe_lr=0.1,
            eval_interval=Duration.epochs(20),
            input_modalities=[Modality.SENTINEL1.name, Modality.SENTINEL2_L2A.name],
            epochs=50,
        ),
        "mados": DownstreamTaskConfig(
            dataset="mados",
            embedding_batch_size=128,
            probe_batch_size=128,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=False,
            probe_lr=0.1,
            eval_interval=Duration.epochs(10),
        ),
        "sen1floods11": DownstreamTaskConfig(
            dataset="sen1floods11",
            embedding_batch_size=128,
            probe_batch_size=128,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            probe_lr=0.1,
            eval_interval=Duration.epochs(10),
        ),
        "m-bigearthnet": DownstreamTaskConfig(
            dataset="m-bigearthnet",
            embedding_batch_size=64,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            eval_interval=Duration.epochs(20),
        ),
        "m-brick-kiln": DownstreamTaskConfig(
            dataset="m-brick-kiln",
            embedding_batch_size=128,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            eval_interval=Duration.epochs(20),
        ),
        "m-so2sat": DownstreamTaskConfig(
            dataset="m-so2sat",
            embedding_batch_size=128,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            eval_interval=Duration.epochs(20),
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
    )
    return trainer_config


def build_common_components_mae(*args: Any) -> CommonComponents:
    """Build the common components for an experiment."""
    common = build_common_components(*args)
    return CommonComponents(
        run_name=common.run_name,
        save_folder=common.save_folder,
        launch=common.launch,
        training_modalities=MAE_MODALITIES,
    )


if __name__ == "__main__":
    main(
        common_components_builder=build_common_components_mae,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
    )

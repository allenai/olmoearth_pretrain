"""Script for v0 sweep for Galileo."""

import logging

from olmo_core.config import DType
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
from shared import (
    TRAINING_MODALITIES,
    build_dataloader_config,
    build_model_config_builder,
    build_train_module_config_builder,
    build_visualize_config,
)

from helios.data.concat import HeliosConcatDatasetConfig
from helios.data.constants import Modality
from helios.data.dataset import HeliosDatasetConfig
from helios.internal.common import build_common_components
from helios.internal.experiment import CommonComponents, main
from helios.nn.flexihelios import (
    PoolingType,
)
from helios.train.callbacks import (
    DownstreamEvaluatorCallbackConfig,
    HeliosSpeedMonitorCallback,
    HeliosWandBCallback,
)
from helios.train.callbacks.evaluator_callback import DownstreamTaskConfig

logger = logging.getLogger(__name__)


def build_dataset_config(common: CommonComponents) -> HeliosDatasetConfig:
    """Build the dataset config for an experiment."""
    dataset_configs = [
        # presto
        HeliosDatasetConfig(
            h5py_dir="/weka/dfive-default/helios/dataset/presto/h5py_data_w_missing_timesteps_zstd_3/landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/117473/",
            training_modalities=TRAINING_MODALITIES,
            # use_samples_with_missing_supported_modalities=False,
            dtype=DType.float32,
            # cache_dir="/helios_cache/presto",
        ),
        # osm_sampling
        HeliosDatasetConfig(
            h5py_dir="/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_zstd_3/landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/285288/",
            training_modalities=TRAINING_MODALITIES,
            # use_samples_with_missing_supported_modalities=False,
            dtype=DType.float32,
            # cache_dir="/helios_cache/osm_sampling",
        ),
    ]
    return HeliosConcatDatasetConfig(dataset_configs=dataset_configs)


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """Build the trainer config for an experiment."""
    MAX_DURATION = Duration.epochs(200)
    METRICS_COLLECT_INTERVAL = 10
    CANCEL_CHECK_INTERVAL = 25
    LOAD_STRATEGY = LoadStrategy.if_available
    WANDB_USERNAME = "eai-ai2"  # nosec
    WANDB_PROJECT = "v0-sweep-rerun"
    PERMANENT_SAVE_INTERVAL = 5000
    EPHERMERAL_SAVE_INTERVAL = 250

    checkpointer_config = CheckpointerConfig(work_dir=common.save_folder)
    wandb_callback = HeliosWandBCallback(
        name=common.run_name,
        project=WANDB_PROJECT,
        entity=WANDB_USERNAME,
        enabled=True,  # set to False to avoid wandb errors
    )
    # Safe to collect every step for now
    garbage_collector_callback = GarbageCollectorCallback(gc_interval=1)
    logger.warning("WANDB Distribution Uploads are disabled for Debugging")
    EVAL_TASKS = {
        "m-eurosat": DownstreamTaskConfig(
            dataset="m-eurosat",
            batch_size=128,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            eval_interval=Duration.epochs(5),
        ),
        "breizhcrops": DownstreamTaskConfig(
            dataset="breizhcrops",
            batch_size=128,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            eval_interval=Duration.epochs(50),
            patch_size=1,
        ),
        "pastis": DownstreamTaskConfig(
            dataset="pastis",
            batch_size=8,
            num_workers=2,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            probe_lr=0.1,
            eval_interval=Duration.epochs(50),
            input_modalities=["sentinel2"],
        ),
        "sickle-sentinel1": DownstreamTaskConfig(
            dataset="sickle",
            batch_size=8,
            num_workers=2,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            probe_lr=0.01,
            eval_interval=Duration.epochs(10),
            input_modalities=["sentinel1"],
        ),
        "sickle-landsat": DownstreamTaskConfig(
            dataset="sickle",
            batch_size=8,
            num_workers=2,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            probe_lr=0.01,
            eval_interval=Duration.epochs(10),
            input_modalities=[Modality.LANDSAT.name],
        ),
        "mados": DownstreamTaskConfig(
            dataset="mados",
            batch_size=128,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=False,
            probe_lr=0.1,
            eval_interval=Duration.epochs(10),
        ),
        "sen1floods11": DownstreamTaskConfig(
            dataset="sen1floods11",
            batch_size=128,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            probe_lr=0.1,
            eval_interval=Duration.epochs(10),
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


if __name__ == "__main__":
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config_builder(
            separate_attention=False, model="galileo"
        ),
        train_module_config_builder=build_train_module_config_builder(model="galileo"),
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )

"""Launch script for evaluation allowing you to easily run all the evals for your model by just pointing at your training script."""

import importlib.util
import os
import sys
from logging import getLogger

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

from helios.data.constants import Modality
from helios.internal.experiment import (
    CommonComponents,
    main,
)
from helios.nn.flexihelios import (
    PoolingType,
)
from helios.train.callbacks import (
    DownstreamEvaluatorCallbackConfig,
    HeliosSpeedMonitorCallback,
    HeliosWandBCallback,
)
from helios.train.callbacks.evaluator_callback import DownstreamTaskConfig

logger = getLogger(__name__)


def load_user_module(path):
    """Load the user module from the given path."""
    logger.info(f"Loading user module from {path}")
    spec = importlib.util.spec_from_file_location("user_module", path)
    user_mod = importlib.util.module_from_spec(spec)
    sys.modules["user_module"] = user_mod
    spec.loader.exec_module(user_mod)
    return user_mod


EVAL_TASKS = {
    "m_forestnet": DownstreamTaskConfig(
        dataset="m-forestnet",
        embedding_batch_size=128,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=False,
        eval_interval=Duration.epochs(5),
    ),
    "m_eurosat": DownstreamTaskConfig(
        dataset="m-eurosat",
        embedding_batch_size=128,
        num_workers=0,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,  # True, #False,
        eval_interval=Duration.epochs(5),
    ),
    "m_bigearthnet": DownstreamTaskConfig(
        dataset="m-bigearthnet",
        embedding_batch_size=64,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        eval_interval=Duration.epochs(5),
    ),
    "m_so2sat": DownstreamTaskConfig(
        dataset="m-so2sat",
        embedding_batch_size=128,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        eval_interval=Duration.epochs(5),
    ),
    "m_brick_kiln": DownstreamTaskConfig(
        dataset="m-brick-kiln",
        embedding_batch_size=128,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        eval_interval=Duration.epochs(5),
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
    "pastis_sentinel2": DownstreamTaskConfig(
        dataset="pastis",
        embedding_batch_size=32,
        probe_batch_size=8,
        num_workers=2,
        pooling_type=PoolingType.MAX,
        norm_stats_from_pretrained=True,
        probe_lr=0.1,
        eval_interval=Duration.epochs(50),
        input_modalities=[Modality.SENTINEL2_L2A.name],
        epochs=50,
    ),
    "breizhcrops": DownstreamTaskConfig(
        dataset="breizhcrops",
        embedding_batch_size=128,
        probe_batch_size=128,
        num_workers=0,
        pooling_type=PoolingType.MAX,
        norm_stats_from_pretrained=True,
        eval_interval=Duration.epochs(50),
        patch_size=1,
        eval_mode="linear_probe",
        probe_lr=0.1,
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
    "m_sa_crop_type": DownstreamTaskConfig(
        dataset="m-sa-crop-type",
        embedding_batch_size=32,
        probe_batch_size=8,
        num_workers=2,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=False,
        probe_lr=0.1,
        eval_interval=Duration.epochs(10),
    ),
    "m_cashew_plant": DownstreamTaskConfig(
        dataset="m-cashew-plant",
        embedding_batch_size=32,
        probe_batch_size=8,
        num_workers=2,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        probe_lr=0.1,
        eval_interval=Duration.epochs(10),
    ),
    "cropharvest_Togo_12_sentinel2": DownstreamTaskConfig(
        dataset="cropharvest_Togo_12",
        embedding_batch_size=128,
        num_workers=2,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        eval_interval=Duration.epochs(20),
        input_modalities=[Modality.SENTINEL2_L2A.name],
        patch_size=1,
        eval_mode="linear_probe",
        probe_lr=0.1,
        epochs=50,
    ),
    "cropharvest_Togo_12_sentinel1": DownstreamTaskConfig(
        dataset="cropharvest_Togo_12",
        embedding_batch_size=128,
        num_workers=2,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        eval_interval=Duration.epochs(20),
        input_modalities=[Modality.SENTINEL1.name],
        patch_size=1,
        eval_mode="linear_probe",
        probe_lr=0.1,
        epochs=50,
    ),
    # example of "in season" cropland mapping - 6 indicates only the
    # first 6 timesteps are passed to the model
    "cropharvest_Peoples_Republic_of_China_6": DownstreamTaskConfig(
        dataset="cropharvest_People's Republic of China_6",
        embedding_batch_size=128,
        num_workers=2,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        eval_interval=Duration.epochs(20),
        input_modalities=[Modality.SENTINEL2_L2A.name],
        patch_size=1,
        eval_mode="linear_probe",
        probe_lr=0.1,
        epochs=50,
    ),
    "cropharvest_Peoples_Republic_of_China_6_sentinel1": DownstreamTaskConfig(
        dataset="cropharvest_People's Republic of China_6",
        embedding_batch_size=128,
        num_workers=2,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        eval_interval=Duration.epochs(20),
        input_modalities=[Modality.SENTINEL1.name],
        patch_size=1,
        eval_mode="linear_probe",
        probe_lr=0.1,
        epochs=50,
    ),
    "cropharvest_Togo_12_sentinel2_sentinel1": DownstreamTaskConfig(
        dataset="cropharvest_Togo_12",
        embedding_batch_size=128,
        num_workers=2,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        eval_interval=Duration.epochs(20),
        input_modalities=[Modality.SENTINEL2_L2A.name, Modality.SENTINEL1.name],
        patch_size=1,
        eval_mode="linear_probe",
        probe_lr=0.1,
        epochs=50,
    ),
    "cropharvest_Peoples_Republic_of_China_6_sentinel2_sentinel1_sentinel2": DownstreamTaskConfig(
        dataset="cropharvest_People's Republic of China_6",
        embedding_batch_size=128,
        num_workers=2,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        eval_interval=Duration.epochs(20),
        input_modalities=[
            Modality.SENTINEL2_L2A.name,
            Modality.SENTINEL1.name,
        ],
        patch_size=1,
        eval_mode="linear_probe",
        probe_lr=0.1,
        epochs=50,
    ),
}


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """Build the trainer config for an experiment."""
    MAX_DURATION = Duration.epochs(300)
    METRICS_COLLECT_INTERVAL = 10
    CANCEL_CHECK_INTERVAL = 1
    LOAD_STRATEGY = LoadStrategy.if_available
    WANDB_USERNAME = "eai-ai2"  # nosec
    WANDB_PROJECT = "helios_in_loop_evals"
    PERMANENT_SAVE_INTERVAL = 5000
    EPHERMERAL_SAVE_INTERVAL = 250
    checkpointer_config = CheckpointerConfig(work_dir=common.save_folder)
    wandb_callback = HeliosWandBCallback(
        name=common.run_name,
        project=WANDB_PROJECT,
        entity=WANDB_USERNAME,
        enabled=True,  # set to False to avoid wandb errors
    )
    # Safe to collect everys tep for now
    garbage_collector_callback = GarbageCollectorCallback(gc_interval=1)
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
                eval_on_startup=True,
                cancel_after_first_eval=True,
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
    module_path = os.environ.get("TRAIN_SCRIPT_PATH")
    if module_path is None:
        raise ValueError("TRAIN_SCRIPT_PATH environment variable must be set")
    user_mod = load_user_module(module_path)

    # 3) Inject all of the builder names into your namespace
    build_common_components = user_mod.build_common_components
    build_model_config = user_mod.build_model_config
    build_train_module_config = user_mod.build_train_module_config
    build_dataset_config = user_mod.build_dataset_config
    build_dataloader_config = user_mod.build_dataloader_config
    build_visualize_config = user_mod.build_visualize_config
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )

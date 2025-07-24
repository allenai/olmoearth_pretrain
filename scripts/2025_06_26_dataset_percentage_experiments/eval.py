"""Launch script for evaluation.

The eval.sh calls this script with all the different checkpoints we had for this same
model architecture related to fixed modality masking.
"""

from latent_mim_all_data import (
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_model_config,
    build_train_module_config,
    build_visualize_config,
)
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


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """Build the trainer config for an experiment."""
    MAX_DURATION = Duration.epochs(300)
    METRICS_COLLECT_INTERVAL = 10
    CANCEL_CHECK_INTERVAL = 1
    LOAD_STRATEGY = LoadStrategy.if_available
    WANDB_USERNAME = "eai-ai2"  # nosec
    WANDB_PROJECT = "2025_06_30_dataset_percentage__eval"
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
    EVAL_TASKS = {
        "m-eurosat": DownstreamTaskConfig(
            dataset="m-eurosat",
            embedding_batch_size=128,
            num_workers=0,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            eval_interval=Duration.epochs(5),
        ),
        "m-bigearthnet": DownstreamTaskConfig(
            dataset="m-bigearthnet",
            embedding_batch_size=64,
            num_workers=4,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            eval_interval=Duration.epochs(5),
        ),
        "m-so2sat": DownstreamTaskConfig(
            dataset="m-so2sat",
            embedding_batch_size=128,
            num_workers=4,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            eval_interval=Duration.epochs(5),
        ),
        "m-brick-kiln": DownstreamTaskConfig(
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
        "sickle_sentinel1": DownstreamTaskConfig(
            dataset="sickle",
            embedding_batch_size=32,
            probe_batch_size=16,
            num_workers=2,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            probe_lr=0.01,
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
        "sickle_sentinel1_landsat": DownstreamTaskConfig(
            dataset="sickle",
            embedding_batch_size=32,
            probe_batch_size=16,
            num_workers=2,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            probe_lr=0.002,
            eval_interval=Duration.epochs(10),
            input_modalities=[Modality.SENTINEL1.name, Modality.LANDSAT.name],
            epochs=50,
        ),
        "m_sa_crop_type": DownstreamTaskConfig(
            dataset="m-sa-crop-type",
            embedding_batch_size=32,
            probe_batch_size=8,
            num_workers=2,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
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
        "pastis_sentinel2": DownstreamTaskConfig(
            dataset="pastis",
            embedding_batch_size=32,
            probe_batch_size=8,
            num_workers=2,
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
            probe_batch_size=8,
            num_workers=2,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            probe_lr=0.1,
            eval_interval=Duration.epochs(50),
            input_modalities=[Modality.SENTINEL1.name],
            epochs=50,
        ),
        "pastis_sentinel1_sentinel2": DownstreamTaskConfig(
            dataset="pastis",
            embedding_batch_size=32,
            probe_batch_size=8,
            num_workers=2,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            probe_lr=0.1,
            eval_interval=Duration.epochs(20),
            input_modalities=[Modality.SENTINEL1.name, Modality.SENTINEL2_L2A.name],
            epochs=50,
        ),
        "breizhcrops": DownstreamTaskConfig(
            dataset="breizhcrops",
            embedding_batch_size=128,
            probe_batch_size=128,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            eval_interval=Duration.epochs(50),
            patch_size=1,
            eval_mode="linear_probe",
            probe_lr=0.1,
            epochs=50,
        ),
        "cropharvest_Togo_12": DownstreamTaskConfig(
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
        # example of "in season" cropland mapping - 6 indicates only the
        # first 6 timesteps are passed to the model
        "cropharvest_People's Republic of China_6": DownstreamTaskConfig(
            dataset="cropharvest_People's Republic of China_6",
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
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )

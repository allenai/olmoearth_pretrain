"""Base script for this collection of experiments."""

import logging

from olmo_core.config import Config, DType
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
from upath import UPath

from helios.data.concat import HeliosConcatDatasetConfig
from helios.data.dataloader import HeliosDataLoaderConfig
from helios.data.dataset import HeliosDatasetConfig
from helios.data.normalize import Strategy
from helios.data.transform import TransformConfig
from helios.internal.common import build_common_components
from helios.internal.experiment import CommonComponents, HeliosVisualizeConfig, main
from helios.nn.flexihelios import EncoderConfig, PoolingType, PredictorConfig
from helios.nn.latent_mim import LatentMIMConfig
from helios.train.callbacks import (
    DownstreamEvaluatorCallbackConfig,
    HeliosSpeedMonitorCallback,
    HeliosWandBCallback,
)
from helios.train.callbacks.evaluator_callback import DownstreamTaskConfig
from helios.train.loss import LossConfig
from helios.train.masking import MaskingConfig
from helios.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)
MAX_PATCH_SIZE = 8
MIN_PATCH_SIZE = 1


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model config for an experiment."""
    # Fixed model parameters
    ENCODER_EMBEDDING_SIZE = 192
    DECODER_EMBEDDING_SIZE = 192
    ENCODER_DEPTH = 12
    DECODER_DEPTH = 12
    ENCODER_NUM_HEADS = 3
    DECODER_NUM_HEADS = 3
    MLP_RATIO = 4.0
    encoder_config = EncoderConfig(
        supported_modality_names=common.supported_modality_names,
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
        supported_modality_names=common.supported_modality_names,
        learnable_channel_embeddings=True,
    )
    model_config = LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
    )
    return model_config


def build_train_module_config(
    common: CommonComponents,
) -> LatentMIMTrainModuleConfig:
    """Build the train module config for an experiment."""
    LR = 2e-3
    RANK_MICROBATCH_SIZE = 128
    ENCODE_RATIO = 0.1
    DECODE_RATIO = 0.74
    WD = 2e-3
    optim_config = AdamWConfig(lr=LR, weight_decay=WD)
    masking_config = MaskingConfig(
        strategy_config={
            "type": "random",
            "encode_ratio": ENCODE_RATIO,
            "decode_ratio": DECODE_RATIO,
        }
    )
    loss_config = LossConfig(
        loss_config={
            "type": "patch_discrimination_new",  # TODO: Should be registered via enum names
        }
    )
    TRANSFORM_TYPE = "flip_and_rotate"
    transform_config = TransformConfig(
        transform_type=TRANSFORM_TYPE,
    )
    token_exit_cfg = {modality: 0 for modality in common.supported_modality_names}

    WARMUP_EPOCHS = 10
    dp_config = DataParallelConfig(
        name=DataParallelType.fsdp,
        param_dtype=DType.bfloat16,
        reduce_dtype=DType.float32,
    )

    # TODO: would need a scheduler config and registry to be able to change this with overrides
    scheduler = CosWithWarmup()
    train_module_config = LatentMIMTrainModuleConfig(
        optim_config=optim_config,
        transform_config=transform_config,
        masking_config=masking_config,
        warmup_duration=Duration.epochs(WARMUP_EPOCHS),
        loss_config=loss_config,
        rank_microbatch_size=RANK_MICROBATCH_SIZE,
        token_exit_cfg=token_exit_cfg,
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
    TOKEN_BUDGET = 1500

    SAMPLE_HW_P_LIST = list(range(5, 13))

    dataloader_config = HeliosDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=123456,
        work_dir=common.save_folder,
        num_workers=NUM_WORKERS,
        sampled_hw_p_list=SAMPLE_HW_P_LIST,
        min_patch_size=MIN_PATCH_SIZE,
        max_patch_size=MAX_PATCH_SIZE,
        token_budget=TOKEN_BUDGET,
    )
    return dataloader_config


def build_dataset_config(common: CommonComponents) -> Config:
    """Build the dataset config for an experiment."""
    dataset_configs = [
        HeliosDatasetConfig(
            h5py_dir="/weka/dfive-default/helios/dataset/osm_sampling/h5py_data/latlon_sentinel1_sentinel2_l2a_worldcover/255819",
            tile_path=None,
            supported_modality_names=common.supported_modality_names,
            dtype=DType.float32,
        ),
        HeliosDatasetConfig(
            h5py_dir="/weka/dfive-default/helios/dataset/presto/h5py_data/latlon_sentinel1_sentinel2_l2a_worldcover/102686",
            tile_path=None,
            supported_modality_names=common.supported_modality_names,
            dtype=DType.float32,
        ),
    ]
    return HeliosConcatDatasetConfig(dataset_configs=dataset_configs)


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """Build the trainer config for an experiment."""
    MAX_DURATION = Duration.epochs(200)
    METRICS_COLLECT_INTERVAL = 10
    CANCEL_CHECK_INTERVAL = 1
    LOAD_STRATEGY = LoadStrategy.if_available
    WANDB_USERNAME = "eai-ai2"  # nosec
    WANDB_PROJECT = "2025_04_09_helios_multidataset"
    PERMANENT_SAVE_INTERVAL = 5000
    EPHERMERAL_SAVE_INTERVAL = 250
    checkpointer_config = CheckpointerConfig(
        work_dir=common.save_folder,
    )
    wandb_callback = HeliosWandBCallback(
        name=common.run_name,
        project=WANDB_PROJECT,
        entity=WANDB_USERNAME,
        cancel_check_interval=100,  # checks for cancel tag on wandb
        enabled=True,  # set to False to avoid wandb errors
    )
    # Safe to collect everys tep for now
    garbage_collector_callback = GarbageCollectorCallback(gc_interval=1)
    EVAL_TASKS = {
        "m-eurosat": DownstreamTaskConfig(
            dataset="m-eurosat",
            batch_size=128,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            eval_interval=Duration.epochs(5),
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
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=PERMANENT_SAVE_INTERVAL,
                ephemeral_save_interval=EPHERMERAL_SAVE_INTERVAL,
            ),
        )
    )
    return trainer_config


def build_visualize_config(common: CommonComponents) -> HeliosVisualizeConfig:
    """Build the visualize config for an experiment."""
    return HeliosVisualizeConfig(
        num_samples=None,
        output_dir=str(UPath(common.save_folder) / "visualizations"),
        normalize_strategy=Strategy.PREDEFINED,
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

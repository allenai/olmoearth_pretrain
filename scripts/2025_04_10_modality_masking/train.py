"""Trying to prototype fitting everything into olmo core."""

import logging

from olmo_core.config import DType
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

from helios.data.dataloader import HeliosDataLoaderConfig
from helios.data.dataset import HeliosDatasetConfig
from helios.data.normalize import Strategy
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
    ENCODER_EMBEDDING_SIZE = 128
    DECODER_EMBEDDING_SIZE = 128
    ENCODER_DEPTH = 4
    DECODER_DEPTH = 4
    ENCODER_NUM_HEADS = 8
    DECODER_NUM_HEADS = 8
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
    LR = 0.002
    RANK_MICROBATCH_SIZE = 64
    ENCODE_RATIO = 0.5
    DECODE_RATIO = 0.5
    WD = 0.02
    optim_config = AdamWConfig(lr=LR, weight_decay=WD)
    masking_config = MaskingConfig(
        strategy_config={
            "type": "selectable_modality",
            "encode_ratio": ENCODE_RATIO,
            "decode_ratio": DECODE_RATIO,
            "decodable_modalities": ["worldcover"],
            "fully_mask_modalities": ["sentinel2_l2a", "sentinel1"],
            "max_to_mask": 1,
        }
    )
    loss_config = LossConfig(
        loss_config={
            "type": "patch_discrimination_new",  # TODO: Should be registered via enum names
        }
    )
    token_exit_cfg = {modality: 0 for modality in common.supported_modality_names}

    WARMUP_EPOCHS = 20
    dp_config = DataParallelConfig(name=DataParallelType.ddp)

    # TODO: would need a scheduler config and registry to be able to change this with overrides
    scheduler = CosWithWarmup()
    train_module_config = LatentMIMTrainModuleConfig(
        optim_config=optim_config,
        masking_config=masking_config,
        warmup_duration=Duration.epochs(WARMUP_EPOCHS),
        loss_config=loss_config,
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
    GLOBAL_BATCH_SIZE = 512
    TOKEN_BUDGET = 1500

    SAMPLE_HW_P_LIST = list(range(5, 13))

    dataloader_config = HeliosDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=3622,
        work_dir=common.save_folder,
        num_workers=NUM_WORKERS,
        sampled_hw_p_list=SAMPLE_HW_P_LIST,
        min_patch_size=MIN_PATCH_SIZE,
        max_patch_size=MAX_PATCH_SIZE,
        token_budget=TOKEN_BUDGET,
    )
    return dataloader_config


def build_dataset_config(common: CommonComponents) -> HeliosDatasetConfig:
    """Build the dataset config for an experiment."""
    # NOTE: Change this directory based on the supported modalities
    h5py_dir = "/weka/dfive-default/helios/dataset/presto/h5py_data/latlon_sentinel1_sentinel2_l2a_worldcover/102686"
    return HeliosDatasetConfig(
        h5py_dir=h5py_dir,
        supported_modality_names=common.supported_modality_names,
        use_samples_with_missing_supported_modalities=True,
        dtype=DType.float32,
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """Build the trainer config for an experiment."""
    MAX_DURATION = Duration.epochs(300)
    METRICS_COLLECT_INTERVAL = 1
    CANCEL_CHECK_INTERVAL = 1
    LOAD_STRATEGY = LoadStrategy.if_available
    WANDB_USERNAME = "eai-ai2"  # nosec
    WANDB_PROJECT = "2025_04_10_modality_masking"
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

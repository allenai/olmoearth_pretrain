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
from helios.internal.common import build_common_components
from helios.internal.experiment import CommonComponents, HeliosVisualizeConfig, main
from helios.internal.utils import MODEL_SIZE_ARGS
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

base_model_args = MODEL_SIZE_ARGS["base_super_shallow_decoder"]


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model config for an experiment."""
    ENCODER_EMBEDDING_SIZE = base_model_args["encoder_embedding_size"]
    DECODER_EMBEDDING_SIZE = base_model_args["decoder_embedding_size"]
    ENCODER_DEPTH = base_model_args["encoder_depth"]
    DECODER_DEPTH = base_model_args["decoder_depth"]
    ENCODER_NUM_HEADS = base_model_args["encoder_num_heads"]
    DECODER_NUM_HEADS = base_model_args["decoder_num_heads"]
    MLP_RATIO = base_model_args["mlp_ratio"]
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
    model_config = LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
    )
    return model_config


def build_train_module_config(
    common: CommonComponents,
) -> LatentMIMTrainModuleConfig:
    """Build the train module config for an experiment."""
    LR = 0.0001
    RANK_MICROBATCH_SIZE = 32
    ENCODE_RATIO = 0.1
    DECODE_RATIO = 0.75
    WD = 0.02
    optim_config = AdamWConfig(lr=LR, weight_decay=WD)
    masking_config = MaskingConfig(
        strategy_config={
            "type": "modality_cross_space",
            "max_unmasking_bandsets": 20,
            "min_encoding_bandsets": 2,
            "max_encoding_bandsets": 6,
            "encode_ratio": ENCODE_RATIO,
            "decode_ratio": DECODE_RATIO,
        }
    )
    loss_config = LossConfig(
        loss_config={
            "type": "patch_discrimination_new",  # TODO: Should be registered via enum namecds
        }
    )
    token_exit_cfg = {modality: 0 for modality in common.training_modalities}

    WARMUP_EPOCHS = 10
    dp_config = DataParallelConfig(name=DataParallelType.ddp)

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

    NUM_WORKERS = 8
    GLOBAL_BATCH_SIZE = 512
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
    h5py_dir = "/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_gzip_3/landsat_naip_10_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/334699"
    return HeliosDatasetConfig(
        h5py_dir=h5py_dir,
        use_samples_with_missing_supported_modalities=True,
        training_modalities=common.training_modalities,
        dtype=DType.float32,
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """Build the trainer config for an experiment."""
    MAX_DURATION = Duration.epochs(300)
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
        "m-bigearthnet": DownstreamTaskConfig(
            dataset="m-bigearthnet",
            batch_size=64,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            eval_interval=Duration.epochs(20),
        ),
        "m-brick-kiln": DownstreamTaskConfig(
            dataset="m-brick-kiln",
            batch_size=128,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            eval_interval=Duration.epochs(20),
        ),
        "m-so2sat": DownstreamTaskConfig(
            dataset="m-so2sat",
            batch_size=128,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            eval_interval=Duration.epochs(20),
        ),
        "breizhcrops": DownstreamTaskConfig(
            dataset="breizhcrops",
            batch_size=128,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            eval_interval=Duration.epochs(20),
            patch_size=1,
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
            input_modalities=["sentinel2"],
        ),
        # "pastis-r": DownstreamTaskConfig(
        #     dataset="pastis",
        #     batch_size=8,
        #     num_workers=2,
        #     pooling_type=PoolingType.MEAN,
        #     norm_stats_from_pretrained=True,
        #     probe_lr=0.1,
        #     eval_interval=Duration.epochs(20),
        #     input_modalities=["sentinel1", "sentinel2"],
        # ),
        # "sickle": DownstreamTaskConfig(
        #     dataset="sickle",
        #     batch_size=8,
        #     num_workers=2,
        #     pooling_type=PoolingType.MEAN,
        #     norm_stats_from_pretrained=True,
        #     probe_lr=0.1,
        #     eval_interval=Duration.epochs(20),
        #     input_modalities=[Modality.LANDSAT.name],
        # ),
        # "sickle-r": DownstreamTaskConfig(
        #     dataset="sickle",
        #     batch_size=8,
        #     num_workers=2,
        #     pooling_type=PoolingType.MEAN,
        #     norm_stats_from_pretrained=True,
        #     probe_lr=0.1,
        #     eval_interval=Duration.epochs(20),
        #     input_modalities=[Modality.LANDSAT.name, "sentinel1", "sentinel2"],
        # ),
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

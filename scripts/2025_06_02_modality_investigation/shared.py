"""Script for v0 sweep."""

import logging
from collections.abc import Callable

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
from helios.nn.flexihelios import (
    EncoderConfig,
    PoolingType,
    PredictorConfig,
    ReconstructorConfig,
)
from helios.nn.galileo import GalileoConfig
from helios.nn.latent_mim import LatentMIMConfig
from helios.nn.st_model import STEncoderConfig, STPredictorConfig
from helios.train.callbacks import (
    DownstreamEvaluatorCallbackConfig,
    HeliosSpeedMonitorCallback,
    HeliosWandBCallback,
)
from helios.train.callbacks.evaluator_callback import DownstreamTaskConfig
from helios.train.loss import LossConfig
from helios.train.masking import MaskingConfig
from helios.train.train_module.contrastive_latentmim import (
    ContrastiveLatentMIMTrainModuleConfig,
)
from helios.train.train_module.galileo import GalileoTrainModuleConfig
from helios.train.train_module.latent_mim import LatentMIMTrainModuleConfig
from helios.train.train_module.train_module import HeliosTrainModuleConfig

logger = logging.getLogger(__name__)

MAX_PATCH_SIZE = 8  # NOTE: actual patch_size <= max_patch_size
MIN_PATCH_SIZE = 1

MAX_SEQUENCE_LENGTH = 12

TRAINING_MODALITIES = [
    # Modality.SENTINEL2_L2A.name,
    Modality.SENTINEL1.name,
    Modality.WORLDCOVER.name,
    Modality.SRTM.name,
    Modality.LATLON.name,
    Modality.LANDSAT.name,
    Modality.OPENSTREETMAP_RASTER.name,
]

ENCODER_EMBEDDING_SIZE = 768
DECODER_EMBEDDING_SIZE = 768
ENCODER_DEPTH = 12
DECODER_DEPTH = 12
ENCODER_NUM_HEADS = 12
DECODER_NUM_HEADS = 12
MLP_RATIO = 4.0


def build_encoder_config(separate_attention: bool = False) -> Config:
    """Build encoder configs."""
    kwargs = {
        "supported_modality_names": TRAINING_MODALITIES,
        "embedding_size": ENCODER_EMBEDDING_SIZE,
        "max_patch_size": MAX_PATCH_SIZE,
        "num_heads": ENCODER_NUM_HEADS,
        "depth": ENCODER_DEPTH,
        "mlp_ratio": MLP_RATIO,
        "drop_path": 0.1,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "use_channel_embs": True,
    }
    if separate_attention:
        return STEncoderConfig(**kwargs)
    else:
        return EncoderConfig(**kwargs)


def build_decoder_config(separate_attention: bool = False) -> Config:
    """Build decoder configs."""
    kwargs = {
        "encoder_embedding_size": ENCODER_EMBEDDING_SIZE,
        "decoder_embedding_size": DECODER_EMBEDDING_SIZE,
        "depth": DECODER_DEPTH,
        "mlp_ratio": MLP_RATIO,
        "num_heads": DECODER_NUM_HEADS,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "supported_modality_names": TRAINING_MODALITIES,
        "learnable_channel_embeddings": True,
    }
    if separate_attention:
        return STPredictorConfig(**kwargs)
    else:
        return PredictorConfig(**kwargs)


def build_reconstructor_config(separate_attention: bool = False) -> Config:
    """Build reconstructor configs."""
    return ReconstructorConfig(
        supported_modality_names=[
            m for m in TRAINING_MODALITIES if m != Modality.LATLON.name
        ],
        max_patch_size=MAX_PATCH_SIZE,
        decoder_config=build_decoder_config(separate_attention),
    )


def build_model_config(
    separate_attention: bool = False, model: str = "galileo"
) -> Config:
    """Build the model config for an experiment."""
    encoder_config = build_encoder_config(separate_attention)
    decoder_config = build_decoder_config(separate_attention)
    reconstructor_config = build_reconstructor_config(separate_attention)
    if model == "galileo":
        return GalileoConfig(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            reconstructor_config=reconstructor_config,
        )
    elif model == "latentmim" or model == "contrastive_latentmim":
        return LatentMIMConfig(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            reconstructor_config=reconstructor_config,
        )
    else:
        raise ValueError(
            f"model must be galileo, latentmim, or contrastive_latentmim, got {model}"
        )


def build_model_config_builder(
    separate_attention: bool = False, model: str = "galileo"
) -> Callable[[CommonComponents], Config]:
    """Builder for model config builders."""
    return lambda common: build_model_config(separate_attention, model)


def build_train_module_config(model: str = "galileo") -> HeliosTrainModuleConfig:
    """Build the train module config for an experiment."""
    LR = 0.0001
    RANK_MICROBATCH_SIZE = 32
    ENCODE_RATIO = 0.1
    DECODE_RATIO = 0.75
    WD = 0.02
    WARMUP_EPOCHS = 10

    optim_config = AdamWConfig(lr=LR, weight_decay=WD)
    masking_config = MaskingConfig(
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
    loss_config = LossConfig(
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
            "weight": 0.1,
        }
    )
    mae_loss_config = LossConfig(
        loss_config={
            "type": "mae",
            "loss_function": "SmoothL1Loss",
            "beta": 0.02,
            "weight": 0.1,
        }
    )
    token_exit_cfg_galileo = {
        Modality.SENTINEL2_L2A.name: ENCODER_DEPTH,
        Modality.LATLON.name: ENCODER_DEPTH,
        Modality.SENTINEL1.name: ENCODER_DEPTH,
        Modality.WORLDCOVER.name: 0,
        Modality.SRTM.name: int(ENCODER_DEPTH / 2),
        Modality.OPENSTREETMAP_RASTER.name: 0,
        Modality.LANDSAT.name: ENCODER_DEPTH,
    }
    if any(modality not in token_exit_cfg_galileo for modality in TRAINING_MODALITIES):
        raise ValueError(
            f"All modalities must be in token_exit_cfg_a: {TRAINING_MODALITIES}"
        )
    token_exit_cfg_zero = {modality: 0 for modality in TRAINING_MODALITIES}
    dp_config = DataParallelConfig(name=DataParallelType.fsdp)

    # TODO: would need a scheduler config and registry to be able to change this with overrides
    scheduler = CosWithWarmup()
    if model == "galileo":
        return GalileoTrainModuleConfig(
            # TODO: change name to optim config
            optim_config=optim_config,
            warmup_duration=Duration.epochs(WARMUP_EPOCHS),
            masking_config_a=masking_config,
            masking_config_b=masking_config_b,
            loss_config_a=loss_config,
            loss_config_b=loss_config_b,
            contrastive_config=contrastive_config,
            mae_loss_config=mae_loss_config,
            rank_microbatch_size=RANK_MICROBATCH_SIZE,
            token_exit_cfg_a=token_exit_cfg_galileo,
            token_exit_cfg_b=token_exit_cfg_zero,
            autocast_precision=DType.bfloat16,
            max_grad_norm=1.0,
            ema_decay=[1.0, 1.0],
            dp_config=dp_config,
            scheduler=scheduler,
        )
    elif model == "latentmim":
        return LatentMIMTrainModuleConfig(
            optim_config=optim_config,
            masking_config=masking_config,
            warmup_duration=Duration.epochs(WARMUP_EPOCHS),
            loss_config=loss_config,
            mae_loss_config=mae_loss_config,
            rank_microbatch_size=RANK_MICROBATCH_SIZE,
            token_exit_cfg=token_exit_cfg_zero,
            autocast_precision=DType.bfloat16,
            max_grad_norm=1.0,
            dp_config=dp_config,
            scheduler=scheduler,
        )
    elif model == "contrastive_latentmim":
        return ContrastiveLatentMIMTrainModuleConfig(
            optim_config=optim_config,
            masking_config=masking_config,
            warmup_duration=Duration.epochs(WARMUP_EPOCHS),
            loss_config=loss_config,
            mae_loss_config=mae_loss_config,
            rank_microbatch_size=RANK_MICROBATCH_SIZE,
            token_exit_cfg=token_exit_cfg_zero,
            autocast_precision=DType.bfloat16,
            contrastive_config=contrastive_config,
            max_grad_norm=1.0,
            dp_config=dp_config,
            scheduler=scheduler,
        )
    else:
        raise ValueError(f"model must be galileo or latentmim, got {model}")


def build_train_module_config_builder(
    model: str = "galileo",
) -> Callable[[CommonComponents], HeliosTrainModuleConfig]:
    """Builder for train module config builders."""
    return lambda common: build_train_module_config(model)


def build_dataloader_config(common: CommonComponents) -> HeliosDataLoaderConfig:
    """Build the dataloader config for an experiment."""
    # things should be set during building
    # TODO: Include collate function here

    NUM_WORKERS = 16
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
    # Should the dataloader build the config or take an object?
    return dataloader_config


def build_dataset_config(common: CommonComponents) -> HeliosDatasetConfig:
    """Build the dataset config for an experiment."""
    dataset_configs = [
        # presto
        HeliosDatasetConfig(
            h5py_dir="/weka/dfive-default/helios/dataset/presto/h5py_data_w_missing_timesteps_zstd_3/landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/117473/",
            training_modalities=TRAINING_MODALITIES,
            # use_samples_with_missing_supported_modalities=False,
            dtype=DType.float32,
            cache_dir="/helios_cache/presto",
        ),
        # osm_sampling
        HeliosDatasetConfig(
            h5py_dir="/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_zstd_3/landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/285288/",
            training_modalities=TRAINING_MODALITIES,
            # use_samples_with_missing_supported_modalities=False,
            dtype=DType.float32,
            cache_dir="/helios_cache/osm_sampling",
        ),
        # osmbig
        # HeliosDatasetConfig(
        #    h5py_dir="/weka/dfive-default/helios/dataset/osmbig/h5py_data_w_missing_timesteps_zstd_3/landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/324482/",
        #    training_modalities=TRAINING_MODALITIES,
        #    # use_samples_with_missing_supported_modalities=False,
        #    dtype=DType.float32,
        #    cache_dir="/helios_cache/osmbig",
        # ),
    ]
    return HeliosConcatDatasetConfig(dataset_configs=dataset_configs)


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """Build the trainer config for an experiment."""
    MAX_DURATION = Duration.epochs(300)
    METRICS_COLLECT_INTERVAL = 1
    CANCEL_CHECK_INTERVAL = 1
    LOAD_STRATEGY = LoadStrategy.if_available
    WANDB_USERNAME = "eai-ai2"  # nosec
    WANDB_PROJECT = "2025_06_02_modality_investigation"
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
        "sickle_sentinel1": DownstreamTaskConfig(
            dataset="sickle",
            embedding_batch_size=32,
            probe_batch_size=16,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            probe_lr=0.01,
            eval_interval=Duration.epochs(5),
            input_modalities=["sentinel1"],
        ),
        "pastis_sentinel1": DownstreamTaskConfig(
            dataset="pastis",
            embedding_batch_size=32,
            probe_batch_size=16,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            probe_lr=0.01,
            eval_interval=Duration.epochs(5),
            input_modalities=["sentinel1"],
            epochs=50,
        ),
        "sickle_landsat": DownstreamTaskConfig(
            dataset="sickle",
            embedding_batch_size=32,
            probe_batch_size=32,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            probe_lr=0.01,
            eval_interval=Duration.epochs(5),
            input_modalities=[Modality.LANDSAT.name],
        ),
        "sickle_landsat_sentinel1": DownstreamTaskConfig(
            dataset="sickle",
            embedding_batch_size=32,
            probe_batch_size=8,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            probe_lr=0.002,
            eval_interval=Duration.epochs(5),
            input_modalities=[Modality.LANDSAT.name, "sentinel1"],
        ),
        "sen1floods11": DownstreamTaskConfig(
            dataset="sen1floods11",
            embedding_batch_size=128,
            probe_batch_size=128,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            probe_lr=0.1,
            eval_interval=Duration.epochs(5),
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
        model_config_builder=build_model_config_builder(),
        train_module_config_builder=build_train_module_config_builder(),
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )

"""Trying to prototype fitting everything into olmo core."""

import logging

from olmo_core.config import DType
from olmo_core.distributed.parallel.data_parallel import (
    DataParallelConfig,
    DataParallelType,
)
from olmo_core.optim import AdamWConfig
from olmo_core.optim.scheduler import WSD
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
from upath import UPath

from helios.data.concat import HeliosConcatDatasetConfig
from helios.data.constants import Modality
from helios.data.dataloader import HeliosDataLoaderConfig
from helios.data.dataset import HeliosDatasetConfig
from helios.internal.common import build_common_components
from helios.internal.experiment import CommonComponents, HeliosVisualizeConfig, main
from helios.internal.utils import MODEL_SIZE_ARGS
from helios.nn.flexihelios import (
    EncoderConfig,
    PoolingType,
    PredictorConfig,
)
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

MAX_PATCH_SIZE = 8
MIN_PATCH_SIZE = 1
model_size = MODEL_SIZE_ARGS["base_shallow_decoder"]


def build_model_config(common: CommonComponents) -> GalileoConfig:
    """Build the model config for an experiment."""
    encoder_config = EncoderConfig(
        embedding_size=model_size["encoder_embedding_size"],
        num_heads=model_size["encoder_num_heads"],
        depth=model_size["encoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        supported_modality_names=common.training_modalities,
        max_patch_size=MAX_PATCH_SIZE,
        drop_path=0.1,
        max_sequence_length=12,
    )
    decoder_config = PredictorConfig(
        encoder_embedding_size=model_size["encoder_embedding_size"],
        decoder_embedding_size=model_size["decoder_embedding_size"],
        depth=model_size["decoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        num_heads=model_size["decoder_num_heads"],
        supported_modality_names=common.training_modalities,
        max_sequence_length=12,
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
    scheduler = WSD(
        decay_steps=50000,
        decay_fraction=None,
    )

    masking_config = MaskingConfig(
        strategy_config={
            "type": "modality_cross_space_time",
            "encode_ratio": 0.1,
            "decode_ratio": 0.85,
            "allow_encoding_decoding_same_bandset": True,
            "min_decoded_bandsets": 6,
            "only_decode_modalities": [
                Modality.OPENSTREETMAP_RASTER.name,
                Modality.WORLDCOVER.name,
            ],
        }
    )
    masking_config_b = MaskingConfig(
        strategy_config={
            "type": "random",
            "encode_ratio": 0.1,
            "decode_ratio": 0.85,
        }
    )
    optim_config = AdamWConfig(lr=0.0001, weight_decay=0.02)
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
            "weight": 0.05,
        }
    )
    token_exit_cfg_galileo = {
        Modality.SENTINEL2_L2A.name: model_size["encoder_depth"],
        Modality.LATLON.name: model_size["encoder_depth"],
        Modality.SENTINEL1.name: model_size["encoder_depth"],
        Modality.WORLDCOVER.name: 0,
        Modality.SRTM.name: int(model_size["encoder_depth"] / 2),
        Modality.OPENSTREETMAP_RASTER.name: 0,
        Modality.LANDSAT.name: model_size["encoder_depth"],
    }
    token_exit_cfg_zero = {modality: 0 for modality in common.training_modalities}
    return GalileoTrainModuleConfig(
        optim_config=optim_config,
        warmup_duration=Duration.steps(8000),
        masking_config_a=masking_config,
        masking_config_b=masking_config_b,
        loss_config_a=loss_config,
        loss_config_b=loss_config_b,
        contrastive_config=contrastive_config,
        mae_loss_config=None,
        rank_microbatch_size=64,
        token_exit_cfg_a=token_exit_cfg_galileo,
        token_exit_cfg_b=token_exit_cfg_zero,
        autocast_precision=None,
        max_grad_norm=1.0,
        scheduler=scheduler,
        ema_decay=(1.0, 1.0),
        dp_config=DataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
    )


def build_dataloader_config(common: CommonComponents) -> HeliosDataLoaderConfig:
    """Build the dataloader config for an experiment."""
    # things should be set during building

    return HeliosDataLoaderConfig(
        num_workers=16,
        global_batch_size=512,
        token_budget=1500,
        prefetch_factor=4,
        sampled_hw_p_list=list(range(5, 13)),
        min_patch_size=MIN_PATCH_SIZE,
        max_patch_size=MAX_PATCH_SIZE,
        work_dir=common.save_folder,
        seed=3622,
    )


def build_dataset_config(common: CommonComponents) -> HeliosDatasetConfig:
    """Build the dataset config for an experiment."""
    dataset_configs = [
        # presto
        HeliosDatasetConfig(
            h5py_dir="/weka/dfive-default/helios/dataset/presto/h5py_data_w_missing_timesteps_128_x_4_zstd_3/landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/469892",
            training_modalities=common.training_modalities,
        ),
        # osm_sampling
        HeliosDatasetConfig(
            h5py_dir="/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_128_x_4_zstd_3/landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/1141152",
            training_modalities=common.training_modalities,
        ),
    ]
    return HeliosConcatDatasetConfig(dataset_configs=dataset_configs)


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """Build the trainer config for an experiment."""
    MAX_DURATION = Duration.epochs(75)
    METRICS_COLLECT_INTERVAL = 10
    CANCEL_CHECK_INTERVAL = 1
    LOAD_STRATEGY = LoadStrategy.if_available
    WANDB_USERNAME = "eai-ai2"  # nosec
    WANDB_PROJECT = "v0.2_sweep"
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
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            eval_interval=Duration.steps(4000),
        ),
        "pastis": DownstreamTaskConfig(
            dataset="pastis",
            embedding_batch_size=32,
            probe_batch_size=8,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            probe_lr=0.1,
            eval_interval=Duration.steps(20000),
            input_modalities=[Modality.SENTINEL2_L2A.name],
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

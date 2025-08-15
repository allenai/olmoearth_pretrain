"""Latent mim cross random masking."""

import logging

from olmo_core.config import DType
from olmo_core.distributed.parallel.data_parallel import (
    DataParallelConfig,
    DataParallelType,
)
from olmo_core.optim import AdamWConfig, ConstantWithWarmup
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


def build_model_config(common: CommonComponents) -> GalileoConfig:
    """Build the model config for an experiment."""
    model_size = MODEL_SIZE_ARGS["base_shallow_decoder"]

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
    decoder_b_config = PredictorConfig(
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
        decoder_b_config=decoder_b_config,
    )
    return model_config


def build_train_module_config(
    common: CommonComponents,
) -> GalileoTrainModuleConfig:
    """Build the train module config for an experiment."""
    # scheduler = ConstantWithWarmup(warmup_steps=8000)
    masking_strategy_no_maps = {
        "type": "modality_cross_random",
        "encode_ratio": 0.5,
        "decode_ratio": 0.5,
        "allow_encoding_decoding_same_bandset": True,
        "always_exclude_modalities": [
            Modality.OPENSTREETMAP_RASTER.name,
            Modality.WORLDCOVER.name,
            Modality.SRTM.name,
        ],
    }
    masking_strategy_maps = {
        "type": "modality_cross_random",
        "encode_ratio": 0.5,
        "decode_ratio": 0.5,
        "min_encoded_bandsets": None,  # use all encodable modalities
        "allow_encoding_decoding_same_bandset": True,
        "only_decode_modalities": [
            Modality.OPENSTREETMAP_RASTER.name,
            Modality.WORLDCOVER.name,
            Modality.SRTM.name,
        ],
        "never_decode_modalities": [
            Modality.SENTINEL1.name,
            Modality.SENTINEL2_L2A.name,
            Modality.LANDSAT.name,
        ],
    }

    scheduler = ConstantWithWarmup(warmup_steps=8000)
    return GalileoTrainModuleConfig(
        optim_config=AdamWConfig(lr=0.0001, weight_decay=0.02, fused=True),
        rank_microbatch_size=64,  # Can be 256 on titan, needs to be <= 64 (i think) on jupiter
        token_exit_cfg_a={modality: 0 for modality in common.training_modalities},
        token_exit_cfg_b={modality: 0 for modality in common.training_modalities},
        max_grad_norm=1.0,
        loss_config_a=LossConfig(loss_config={"type": "patch_discrimination_new"}),
        loss_config_b=LossConfig(loss_config={"type": "patch_discrimination_new"}),
        masking_config_a=MaskingConfig(strategy_config=masking_strategy_no_maps),
        masking_config_b=MaskingConfig(strategy_config=masking_strategy_maps),
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
            h5py_dir="/weka/dfive-default/helios/dataset/presto/h5py_data_w_missing_timesteps_zstd_3_128_x_4/era5_10_landsat_naip_10_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/469728",
            training_modalities=common.training_modalities,
        ),
        # osm_sampling
        HeliosDatasetConfig(
            h5py_dir="/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_zstd_3_128_x_4/gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/1141152",
            training_modalities=common.training_modalities,
        ),
        # osmbig
        HeliosDatasetConfig(
            h5py_dir="/weka/dfive-default/helios/dataset/osmbig/h5py_data_w_missing_timesteps_zstd_3_128_x_4/landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/1297928",
            training_modalities=common.training_modalities,
        ),
        # presto neighbor
        HeliosDatasetConfig(
            h5py_dir="/weka/dfive-default/helios/dataset/presto_neighbor/h5py_data_w_missing_timesteps_zstd_3_128_x_4/landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/3507748",
            training_modalities=common.training_modalities,
        ),
        # worldcover
        HeliosDatasetConfig(
            h5py_dir="/weka/dfive-default/helios/dataset/worldcover_sampling/h5py_data_w_missing_timesteps_zstd_3_128_x_4/landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/6370580",
            training_modalities=common.training_modalities,
        ),
    ]
    return HeliosConcatDatasetConfig(dataset_configs=dataset_configs)


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """Build the trainer config for an experiment."""
    MAX_DURATION = Duration.epochs(400)
    METRICS_COLLECT_INTERVAL = 10
    CANCEL_CHECK_INTERVAL = 25
    LOAD_STRATEGY = LoadStrategy.if_available
    WANDB_USERNAME = "eai-ai2"  # nosec
    WANDB_PROJECT = "2025_08_07_map_cotrain"
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
        "mados": DownstreamTaskConfig(
            dataset="mados",
            embedding_batch_size=128,
            probe_batch_size=128,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=False,
            probe_lr=0.01,
            epochs=50,
            eval_interval=Duration.steps(10000),
        ),
        "m_cashew_plant": DownstreamTaskConfig(
            dataset="m-cashew-plant",
            embedding_batch_size=32,
            probe_batch_size=8,
            num_workers=2,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            probe_lr=0.1,
            eval_interval=Duration.steps(10000),
        ),
        "m_so2sat": DownstreamTaskConfig(
            dataset="m-so2sat",
            embedding_batch_size=128,
            num_workers=4,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=False,
            eval_interval=Duration.steps(10000),
        ),
        "cropharvest_Togo_12_sentinel2": DownstreamTaskConfig(
            dataset="cropharvest_Togo_12",
            embedding_batch_size=128,
            num_workers=2,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            input_modalities=[Modality.SENTINEL2_L2A.name],
            patch_size=1,
            eval_mode="linear_probe",
            probe_lr=0.1,
            epochs=50,
            eval_interval=Duration.steps(20000),
        ),
        "cropharvest_Togo_12_sentinel1": DownstreamTaskConfig(
            dataset="cropharvest_Togo_12",
            embedding_batch_size=128,
            num_workers=2,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            input_modalities=[Modality.SENTINEL1.name],
            patch_size=1,
            eval_mode="linear_probe",
            probe_lr=0.1,
            epochs=50,
            eval_interval=Duration.steps(20000),
        ),
        # "mados_attnpool": DownstreamTaskConfig(
        #     dataset="mados",
        #     embedding_batch_size=128,
        #     probe_batch_size=128,
        #     num_workers=8,
        #     pooling_type=PoolingType.MEAN,
        #     norm_stats_from_pretrained=False,
        #     probe_lr=0.01,
        #     epochs=50,
        #     eval_interval=Duration.steps(10000),
        #     probe_type=ProbeType.ATTNPOOL,
        # ),
        # "m_cashew_plant_attnpool": DownstreamTaskConfig(
        #     dataset="m-cashew-plant",
        #     embedding_batch_size=32,
        #     probe_batch_size=8,
        #     num_workers=2,
        #     pooling_type=PoolingType.MEAN,
        #     norm_stats_from_pretrained=True,
        #     probe_lr=0.1,
        #     eval_interval=Duration.steps(10000),
        #     probe_type=ProbeType.ATTNPOOL,
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

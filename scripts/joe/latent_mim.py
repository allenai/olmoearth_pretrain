"""Trying to prototype fitting everything into olmo core."""

import logging

from olmo_core.config import DType
from olmo_core.distributed.parallel.data_parallel import (
    DataParallelConfig,
    DataParallelType,
)
from olmo_core.internal.common import get_beaker_username
from olmo_core.optim import AdamWConfig
from olmo_core.optim.scheduler import ConstantWithWarmup
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
from helios.internal.common import (
    build_launch_config,
    get_root_dir,
)
from helios.internal.experiment import (
    CommonComponents,
    HeliosVisualizeConfig,
    SubCmd,
    main,
)
from helios.internal.utils import MODEL_SIZE_ARGS
from helios.nn.flexihelios import (
    EncoderConfig,
    PoolingType,
    PredictorConfig,
)
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
    model_size = MODEL_SIZE_ARGS["base_shallow_decoder"]

    model_config = LatentMIMConfig(
        encoder_config=EncoderConfig(
            embedding_size=model_size["encoder_embedding_size"],
            num_heads=model_size["encoder_num_heads"],
            depth=model_size["encoder_depth"],
            mlp_ratio=model_size["mlp_ratio"],
            supported_modality_names=common.training_modalities,
            max_patch_size=MAX_PATCH_SIZE,
            min_patch_size=MIN_PATCH_SIZE,
            drop_path=0.1,
            max_sequence_length=12,
        ),
        decoder_config=PredictorConfig(
            encoder_embedding_size=model_size["encoder_embedding_size"],
            decoder_embedding_size=model_size["decoder_embedding_size"],
            depth=model_size["decoder_depth"],
            mlp_ratio=model_size["mlp_ratio"],
            num_heads=model_size["decoder_num_heads"],
            supported_modality_names=common.training_modalities,
            max_sequence_length=12,
        ),
    )
    return model_config


def build_train_module_config(
    common: CommonComponents,
) -> LatentMIMTrainModuleConfig:
    """Build the train module config for an experiment."""
    return LatentMIMTrainModuleConfig(
        optim_config=AdamWConfig(lr=0.0001, weight_decay=0.02),
        rank_microbatch_size=64,  # Can be 256 on titan, needs to be <= 64 (i think) on jupiter
        masking_config=MaskingConfig(
            strategy_config={
                "type": "modality_cross_random",
                "encode_ratio": 0.75,
                "decode_ratio": 0.25,
                "allow_encoding_decoding_same_bandset": True,
            }
        ),
        loss_config=LossConfig(
            loss_config={
                "type": "patch_discrimination_new",
            }
        ),
        token_exit_cfg={modality: 0 for modality in common.training_modalities},
        max_grad_norm=1.0,
        scheduler=ConstantWithWarmup(warmup=2000),
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
        prefetch_factor=2,
        sampled_hw_p_list=list(range(5, 13)),
        min_patch_size=MIN_PATCH_SIZE,
        max_patch_size=MAX_PATCH_SIZE,
        work_dir=common.save_folder,
        seed=3622,
    )


def build_dataset_config(common: CommonComponents) -> HeliosDatasetConfig:
    """Build the dataset config for an experiment."""
    return HeliosConcatDatasetConfig(
        dataset_configs=[
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
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """Build the trainer config for an experiment."""
    return (
        TrainerConfig(
            work_dir=common.save_folder,
            load_strategy=LoadStrategy.if_available,
            save_folder=common.save_folder,
            cancel_check_interval=10,
            metrics_collect_interval=10,
            max_duration=Duration.epochs(100),
            checkpointer=CheckpointerConfig(work_dir=common.save_folder),
        )
        .with_callback(
            "wandb",
            HeliosWandBCallback(
                name=common.run_name,
                project="helios-joer",
                entity="eai-ai2",  # nosec
                enabled=True,  # set to False to avoid wandb errors
            ),
        )
        .with_callback("speed_monitor", HeliosSpeedMonitorCallback())
        .with_callback("gpu_memory_monitor", GPUMemoryMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks={
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
                },
            ),
        )
        .with_callback("garbage_collector", GarbageCollectorCallback(gc_interval=1))
        .with_callback("beaker", BeakerCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=5000,
                ephemeral_save_interval=250,
            ),
        )
    )


def build_visualize_config(common: CommonComponents) -> HeliosVisualizeConfig:
    """Build the visualize config for an experiment."""
    return HeliosVisualizeConfig(
        num_samples=None,
        output_dir=str(UPath(common.save_folder) / "visualizations"),
        std_multiplier=2.0,
    )


def build_common_components(
    script: str,
    cmd: SubCmd,
    run_name: str,
    cluster: str,
    overrides: list[str],
) -> CommonComponents:
    """Build the common components for an experiment."""
    cmd_to_launch = SubCmd.train
    if cmd == SubCmd.launch_prep:
        cmd_to_launch = SubCmd.prep
    return CommonComponents(
        run_name=run_name,
        save_folder=f"{get_root_dir(cluster)}/checkpoints/{get_beaker_username().lower()}/{run_name}",
        launch=build_launch_config(
            name=f"{run_name}-{cmd_to_launch}",
            cmd=[script, cmd_to_launch, run_name, cluster, *overrides],
            clusters=cluster,
            nccl_debug=False,
        ),
        training_modalities=[
            Modality.LATLON.name,
            Modality.SENTINEL2_L2A.name,
            Modality.SENTINEL1.name,
            Modality.LANDSAT.name,
            # Modality.WORLDCOVER.name,
            # Modality.SRTM.name,
            # Modality.OPENSTREETMAP_RASTER.name,
            # Modality.NAIP_10.name,
        ],
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

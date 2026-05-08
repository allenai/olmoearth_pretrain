# ruff: noqa: D103

r"""ViT-L MAE with scaling-style downstream evals and embedding diagnostics.

Hyperparameters match `base_mae.py` except encoder/decoder width and depth
(`large_shallow_decoder`). Override learning rate via CLI if needed for stability.

Launch:
    python3 scripts/vnext/mae_scaling/large_mae.py dry_run run_name local

    python3 scripts/vnext/mae_scaling/large_mae.py launch run_name ai2/jupiter \
        --launch.num_gpus=8 \
        --trainer.callbacks.wandb.project=2026_05_07_scaling_investigation

Default Beaker launch targets ai2/jupiter and ai2/ceres at urgent priority (override via CLI).
"""

from olmo_core.config import DType
from olmo_core.distributed.parallel.data_parallel import (
    DataParallelConfig,
    DataParallelType,
)
from olmo_core.launch.beaker import BeakerPriority
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
from upath import UPath

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig
from olmoearth_pretrain.evals.metrics import EvalMetric
from olmoearth_pretrain.internal.all_evals import EMBED_DIAG_TASKS
from olmoearth_pretrain.internal.common import (
    build_common_components as build_common_components_default,
)
from olmoearth_pretrain.internal.experiment import (
    CommonComponents,
    OlmoEarthVisualizeConfig,
    SubCmd,
    main,
)
from olmoearth_pretrain.internal.utils import MODEL_SIZE_ARGS
from olmoearth_pretrain.nn.flexi_vit import (
    EncoderConfig,
    PoolingType,
    PredictorConfig,
    ReconstructorConfig,
)
from olmoearth_pretrain.nn.mae import MAEConfig
from olmoearth_pretrain.train.callbacks import (
    DownstreamEvaluatorCallbackConfig,
    OlmoEarthSpeedMonitorCallback,
    OlmoEarthWandBCallback,
)
from olmoearth_pretrain.train.callbacks.evaluator_callback import DownstreamTaskConfig
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import MaskingConfig
from olmoearth_pretrain.train.train_module.mae import MAETrainModuleConfig

MAX_PATCH_SIZE = 8
MIN_PATCH_SIZE = 1
WANDB_PROJECT = "2026_05_07_scaling_investigation"


def get_masking_config() -> MaskingConfig:
    return MaskingConfig(
        strategy_config={
            "type": "modality_cross_random",
            "encode_ratio": 0.5,
            "decode_ratio": 0.5,
            "allow_encoding_decoding_same_bandset": True,
            "only_decode_modalities": [
                "worldcover",
                "srtm",
                "openstreetmap_raster",
                "wri_canopy_height_map",
                "cdl",
                "worldcereal",
            ],
        }
    )


def build_common_components(
    script: str,
    cmd: SubCmd,
    run_name: str,
    cluster: str,
    overrides: list[str],
) -> CommonComponents:
    config = build_common_components_default(script, cmd, run_name, cluster, overrides)
    config.training_modalities = [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
        Modality.LANDSAT.name,
        Modality.WORLDCOVER.name,
        Modality.SRTM.name,
        Modality.OPENSTREETMAP_RASTER.name,
        Modality.WRI_CANOPY_HEIGHT_MAP.name,
        Modality.CDL.name,
        Modality.WORLDCEREAL.name,
    ]
    if config.launch is not None:
        config.launch.num_gpus = 8
        config.launch.clusters = ["ai2/jupiter", "ai2/ceres"]
        config.launch.priority = BeakerPriority.urgent
    return config


def build_model_config(common: CommonComponents) -> MAEConfig:
    model_size = MODEL_SIZE_ARGS["large_shallow_decoder"]

    encoder_config = EncoderConfig(
        embedding_size=model_size["encoder_embedding_size"],
        num_heads=model_size["encoder_num_heads"],
        depth=model_size["encoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        supported_modality_names=common.training_modalities,
        max_patch_size=MAX_PATCH_SIZE,
        drop_path=0.1,
        max_sequence_length=12,
        use_linear_patch_embed=False,
        use_flash_attn=True,
    )
    decoder_config = PredictorConfig(
        encoder_embedding_size=model_size["encoder_embedding_size"],
        decoder_embedding_size=model_size["decoder_embedding_size"],
        depth=model_size["decoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        num_heads=model_size["decoder_num_heads"],
        supported_modality_names=common.training_modalities,
        max_sequence_length=12,
        use_flash_attn=True,
    )
    reconstructor_config = ReconstructorConfig(
        decoder_config=decoder_config,
        supported_modality_names=common.training_modalities,
        max_patch_size=MAX_PATCH_SIZE,
    )
    return MAEConfig(
        encoder_config=encoder_config,
        reconstructor_config=reconstructor_config,
    )


def build_train_module_config(common: CommonComponents) -> MAETrainModuleConfig:
    return MAETrainModuleConfig(
        optim_config=AdamWConfig(lr=0.0001, weight_decay=0.02, fused=False),
        rank_microbatch_size=32,
        masking_config=get_masking_config(),
        mae_loss_config=LossConfig(
            loss_config={"type": "mae", "loss_function": "SmoothL1Loss", "beta": 0.1}
        ),
        token_exit_cfg={modality: 0 for modality in common.training_modalities},
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=8000),
        dp_config=DataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
    )


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    return OlmoEarthDataLoaderConfig(
        num_workers=16,
        global_batch_size=512,
        token_budget=2250,
        prefetch_factor=4,
        sampled_hw_p_list=list(range(1, 13)),
        min_patch_size=MIN_PATCH_SIZE,
        max_patch_size=MAX_PATCH_SIZE,
        work_dir=common.save_folder,
        seed=3622,
        num_masked_views=1,
        masking_config=get_masking_config(),
    )


def build_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    return OlmoEarthDatasetConfig(
        h5py_dir="/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_zstd_3_128_x_4/cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_worldpop_wri_canopy_height_map/1138828",
        training_modalities=common.training_modalities,
    )


EVAL_TASKS = {
    "m-eurosat": DownstreamTaskConfig(
        dataset="m-eurosat",
        embedding_batch_size=128,
        num_workers=0,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        eval_interval=Duration.steps(4000),
    ),
    "m_so2sat": DownstreamTaskConfig(
        dataset="m-so2sat",
        embedding_batch_size=128,
        num_workers=8,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        eval_interval=Duration.steps(20000),
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
    "m-eurosat_10pct": DownstreamTaskConfig(
        dataset="m-eurosat",
        embedding_batch_size=128,
        num_workers=0,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        eval_interval=Duration.steps(4000),
        max_train_samples=2000,
    ),
    "pastis_10pct": DownstreamTaskConfig(
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
        max_train_samples=500,
    ),
    "m_bigearthnet": DownstreamTaskConfig(
        dataset="m-bigearthnet",
        embedding_batch_size=64,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        eval_interval=Duration.steps(20000),
        input_modalities=[Modality.SENTINEL2_L2A.name],
        primary_metric=EvalMetric.MACRO_F1,
    ),
    "m_bigearthnet_10pct": DownstreamTaskConfig(
        dataset="m-bigearthnet",
        embedding_batch_size=64,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        eval_interval=Duration.steps(20000),
        input_modalities=[Modality.SENTINEL2_L2A.name],
        primary_metric=EvalMetric.MACRO_F1,
        max_train_samples=5000,
    ),
    **EMBED_DIAG_TASKS,
}


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    return (
        TrainerConfig(
            work_dir=common.save_folder,
            load_strategy=LoadStrategy.if_available,
            save_folder=common.save_folder,
            cancel_check_interval=25,
            metrics_collect_interval=10,
            max_duration=Duration.epochs(300),
            checkpointer=CheckpointerConfig(work_dir=common.save_folder),
        )
        .with_callback(
            "wandb",
            OlmoEarthWandBCallback(
                name=common.run_name,
                project=WANDB_PROJECT,
                entity="eai-ai2",
                enabled=True,
            ),
        )
        .with_callback("speed_monitor", OlmoEarthSpeedMonitorCallback())
        .with_callback("gpu_memory_monitor", GPUMemoryMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(tasks=EVAL_TASKS),
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


def build_visualize_config(common: CommonComponents) -> OlmoEarthVisualizeConfig:
    return OlmoEarthVisualizeConfig(
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

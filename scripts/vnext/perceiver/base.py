"""Set-Latent Perceiver (SLP) pretraining launch config (ViT-B scale).

Self-contained SSL encoder (see ``docs/perceiver_encoder_spec.md`` and
``olmoearth_pretrain/nn/set_latent_perceiver.py``). Unlike the FlexiViT/LatentMIM
baselines, the SLP masks internally and computes its own soft-InfoNCE loss, so
the dataloader runs with ``num_masked_views=1`` and a trivial masking config (the
masks are ignored; validity is derived from ``MISSING_VALUE``).

Note: ERA5 is not present in the v1.2 corpus below, so the SLP is configured for
the Sentinel-2 / Sentinel-1 / Landsat modalities that the corpus provides. Add
``era5_10`` to both ``training_modalities`` and ``supported_modality_names`` when
running against a corpus that includes it.
"""

import logging

from olmo_core.config import DType
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

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig
from olmoearth_pretrain.evals.datasets.normalize import NormMethod
from olmoearth_pretrain.evals.metrics import EvalMetric
from olmoearth_pretrain.internal.common import (
    build_common_components as build_common_components_default,
)
from olmoearth_pretrain.internal.experiment import (
    CommonComponents,
    OlmoEarthVisualizeConfig,
    SubCmd,
    main,
)
from olmoearth_pretrain.nn.flexi_vit import PoolingType
from olmoearth_pretrain.nn.set_latent_perceiver import SetLatentPerceiverConfig
from olmoearth_pretrain.train.callbacks import (
    DownstreamEvaluatorCallbackConfig,
    OlmoEarthSpeedMonitorCallback,
    OlmoEarthWandBCallback,
)
from olmoearth_pretrain.train.callbacks.evaluator_callback import (
    DownstreamTaskConfig,
    EvalMode,
)
from olmoearth_pretrain.train.masking import MaskingConfig
from olmoearth_pretrain.train.train_module.set_latent_perceiver import (
    SetLatentPerceiverTrainModuleConfig,
)

logger = logging.getLogger(__name__)

# SLP tokenizes on the stored 10 m grid with an 80 m token (patch_px = 8), so the
# dataloader emits samples whose H/W are multiples of 8 (no internal padding).
PATCH_SIZE = 8

SLP_MODALITIES = [
    Modality.SENTINEL2_L2A.name,
    Modality.SENTINEL1.name,
    Modality.LANDSAT.name,
]

# ViT-B scale (spec S3): d=768, heads=12, K=1024.
SLP_SIZES = {
    "nano": dict(
        dim=256,
        heads=8,
        latents=256,
        nested_latents=(64, 128, 256),
        self_depth_per_read=2,
        level2_depth=1,
        decoder_depth=2,
    ),
    "base": dict(
        dim=768,
        heads=12,
        latents=1024,
        nested_latents=(128, 256, 512, 1024),
        self_depth_per_read=4,
        level2_depth=2,
        decoder_depth=2,
    ),
}


def _masking_config() -> MaskingConfig:
    """Trivial masking config for the dataloader (the SLP masks internally)."""
    return MaskingConfig(strategy_config={"type": "random"})


def build_common_components(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """Build the common components for an experiment."""
    config = build_common_components_default(script, cmd, run_name, cluster, overrides)
    config.training_modalities = list(SLP_MODALITIES)
    config.tokenization_config = None
    return config


def build_train_module_config(
    common: CommonComponents,
) -> SetLatentPerceiverTrainModuleConfig:
    """Build the SLP train module config."""
    return SetLatentPerceiverTrainModuleConfig(
        optim_config=AdamWConfig(lr=0.0001, weight_decay=0.02, fused=False),
        rank_microbatch_size=16,
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=8000),
        dp_config=DataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
    )


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """Build the dataloader config (single masked view; SLP masks internally)."""
    return OlmoEarthDataLoaderConfig(
        num_workers=16,
        global_batch_size=256,
        token_budget=2250,
        prefetch_factor=4,
        sampled_hw_p_list=list(range(1, 13)),
        min_patch_size=PATCH_SIZE,
        max_patch_size=PATCH_SIZE,
        work_dir=common.save_folder,
        seed=3622,
        num_masked_views=1,
        masking_config=_masking_config(),
    )


def build_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """Build the dataset config for an experiment."""
    return OlmoEarthDatasetConfig(
        h5py_dir="/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_zstd_3_128_x_4/cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_worldpop_wri_canopy_height_map/1138828",
        training_modalities=common.training_modalities,
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """Build the trainer config for an experiment."""
    MAX_DURATION = Duration.epochs(300)
    checkpointer_config = CheckpointerConfig(work_dir=common.save_folder)
    wandb_callback = OlmoEarthWandBCallback(
        name=common.run_name,
        project="perceiver_slp",
        entity="eai-ai2",
        enabled=True,
    )
    EVAL_TASKS = {
        "m-eurosat": DownstreamTaskConfig(
            dataset="m-eurosat",
            embedding_batch_size=128,
            num_workers=0,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            norm_method=NormMethod.NORM_NO_CLIP_2_STD,
            input_modalities=[Modality.SENTINEL2_L2A.name],
            eval_mode=EvalMode.KNN,
            primary_metric=EvalMetric.ACCURACY,
            eval_interval=Duration.steps(4000),
        ),
        "mados": DownstreamTaskConfig(
            dataset="mados",
            embedding_batch_size=128,
            probe_batch_size=128,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=False,
            norm_method=NormMethod.NORM_NO_CLIP_2_STD,
            probe_lr=0.01,
            eval_interval=Duration.steps(4000),
            input_modalities=[Modality.SENTINEL2_L2A.name],
            eval_mode=EvalMode.LINEAR_PROBE,
            primary_metric=EvalMetric.MICRO_F1,
        ),
    }
    return (
        TrainerConfig(
            work_dir=common.save_folder,
            load_strategy=LoadStrategy.if_available,
            save_folder=common.save_folder,
            cancel_check_interval=25,
            metrics_collect_interval=10,
            max_duration=MAX_DURATION,
            checkpointer=checkpointer_config,
        )
        .with_callback("wandb", wandb_callback)
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
            CheckpointerCallback(save_interval=5000, ephemeral_save_interval=250),
        )
    )


def build_visualize_config(common: CommonComponents) -> OlmoEarthVisualizeConfig:
    """Build the visualize config for an experiment."""
    return OlmoEarthVisualizeConfig(
        num_samples=None,
        output_dir=str(f"{common.save_folder}/visualizations"),
        std_multiplier=2.0,
    )


def build_size_model_config(
    common: CommonComponents, size_name: str
) -> SetLatentPerceiverConfig:
    """Build an SLP model config for the given size preset."""
    size = SLP_SIZES[size_name]
    return SetLatentPerceiverConfig(
        supported_modality_names=list(common.training_modalities),
        token_extent_m=float(PATCH_SIZE * 10),  # 80 m tokens on the 10 m grid
        # v1.2 corpus dates span 2015-2024; units are years since 2020. Eval
        # dates outside this span fall back to the trained "unknown date" null.
        trained_years=(-5.0, 5.0),
        **size,
    )


def build_model_config(common: CommonComponents) -> SetLatentPerceiverConfig:
    """Build the default (ViT-B scale) SLP model config."""
    return build_size_model_config(common, "base")


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

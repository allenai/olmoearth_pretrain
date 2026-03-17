"""ERA5 decode-target tokenization ablations.

Based on base_band_dropout_no_s1_drop_random_time.py with ERA5 added as a decode-only target.
Ablates ERA5 tokenization strategies:

1. era5_default: All 6 ERA5 bands as one bandset (default tokenization)
2. era5_temp_dewpoint: Only temperature + dewpoint as one bandset (drops other bands)
3. era5_multi_bandset: 4 bandsets — temp+dewpoint, precipitation, wind (u+v), pressure
"""

import logging
import os
import sys

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
from olmoearth_pretrain.nn.flexi_vit import PoolingType
from olmoearth_pretrain.nn.flexihelios import EncoderConfig, PredictorConfig
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.nn.tokenization import ModalityTokenization, TokenizationConfig
from olmoearth_pretrain.train.callbacks import (
    DownstreamEvaluatorCallbackConfig,
    OlmoEarthSpeedMonitorCallback,
    OlmoEarthWandBCallback,
)
from olmoearth_pretrain.train.callbacks.evaluator_callback import DownstreamTaskConfig
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import MaskingConfig
from olmoearth_pretrain.train.train_module.contrastive_latentmim import (
    ContrastiveLatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

MAX_PATCH_SIZE = 8
MIN_PATCH_SIZE = 1
RANDOM_BAND_DROPOUT_MAX_RATE = 0.3

# --- Tokenization overrides (S2, Landsat same as base) ---

S2_SINGLE_BANDSET = ModalityTokenization(
    band_groups=[
        [
            "B02",
            "B03",
            "B04",
            "B08",
            "B05",
            "B06",
            "B07",
            "B8A",
            "B11",
            "B12",
            "B01",
            "B09",
        ],
    ]
)

LANDSAT_SINGLE_BANDSET = ModalityTokenization(
    band_groups=[
        ["B8", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"],
    ]
)

# --- ERA5 tokenization variants ---

ERA5_TEMP_DEWPOINT = ModalityTokenization(
    band_groups=[
        ["2m-temperature", "2m-dewpoint-temperature"],
    ]
)

ERA5_MULTI_BANDSET = ModalityTokenization(
    band_groups=[
        ["2m-temperature", "2m-dewpoint-temperature"],
        ["total-precipitation"],
        ["10m-u-component-of-wind", "10m-v-component-of-wind"],
        ["surface-pressure"],
    ]
)

# --- Modality lists ---

ONLY_DECODE_MODALITIES = [
    Modality.WORLDCOVER.name,
    Modality.SRTM.name,
    Modality.OPENSTREETMAP_RASTER.name,
    Modality.WRI_CANOPY_HEIGHT_MAP.name,
    Modality.CDL.name,
    Modality.WORLDCEREAL.name,
    Modality.ERA5_10.name,
]

BAND_DROPOUT_MODALITIES = [
    Modality.SENTINEL2_L2A.name,
    Modality.LANDSAT.name,
]

TRAINING_MODALITIES = [
    Modality.SENTINEL2_L2A.name,
    Modality.SENTINEL1.name,
    Modality.LANDSAT.name,
    Modality.WORLDCOVER.name,
    Modality.SRTM.name,
    Modality.OPENSTREETMAP_RASTER.name,
    Modality.WRI_CANOPY_HEIGHT_MAP.name,
    Modality.CDL.name,
    Modality.WORLDCEREAL.name,
    Modality.ERA5_10.name,
]


# --- Helpers ---


def _tokenization_config(
    era5_override: ModalityTokenization | None = None,
) -> TokenizationConfig:
    overrides: dict[str, ModalityTokenization] = {
        "sentinel2_l2a": S2_SINGLE_BANDSET,
        "landsat": LANDSAT_SINGLE_BANDSET,
    }
    if era5_override is not None:
        overrides["era5_10"] = era5_override
    return TokenizationConfig(overrides=overrides)


def _masking_config(
    tokenization_config: TokenizationConfig | None = None,
) -> MaskingConfig:
    return MaskingConfig(
        strategy_config={
            "type": "random_time_with_decode",
            "encode_ratio": 0.5,
            "decode_ratio": 0.5,
            "only_decode_modalities": ONLY_DECODE_MODALITIES,
        },
        tokenization_config=tokenization_config,
    )


def _build_common(
    script: str,
    cmd: SubCmd,
    run_name: str,
    cluster: str,
    overrides: list[str],
    era5_override: ModalityTokenization | None = None,
) -> CommonComponents:
    config = build_common_components_default(script, cmd, run_name, cluster, overrides)
    config.training_modalities = TRAINING_MODALITIES
    config.tokenization_config = _tokenization_config(era5_override)
    return config


def _build_train_module(
    common: CommonComponents,
    era5_weight: float | None = None,
) -> ContrastiveLatentMIMTrainModuleConfig:
    loss_cfg: dict = {
        "type": "modality_patch_discrimination_masked_negatives",
        "tau": 0.1,
        "same_target_threshold": 0.999,
        "mask_negatives_for_modalities": ONLY_DECODE_MODALITIES,
    }
    if era5_weight is not None:
        loss_cfg["modality_weights"] = {Modality.ERA5_10.name: era5_weight}
    return ContrastiveLatentMIMTrainModuleConfig(
        optim_config=AdamWConfig(lr=0.0001, weight_decay=0.02, fused=False),
        rank_microbatch_size=64,
        masking_config=_masking_config(common.tokenization_config),
        loss_config=LossConfig(loss_config=loss_cfg),
        contrastive_config=LossConfig(
            loss_config={
                "type": "InfoNCE",
                "weight": 0.05,
            }
        ),
        token_exit_cfg={modality: 0 for modality in common.training_modalities},
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=8000),
        ema_decay=(1.0, 1.0),
        dp_config=DataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
    )


def _build_dataloader(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
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
        num_masked_views=2,
        masking_config=_masking_config(common.tokenization_config),
    )


def _build_model(common: CommonComponents) -> LatentMIMConfig:
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
        tokenization_config=common.tokenization_config,
        band_dropout_rate=RANDOM_BAND_DROPOUT_MAX_RATE,
        random_band_dropout=True,
        band_dropout_modalities=BAND_DROPOUT_MODALITIES,
    )
    decoder_config = PredictorConfig(
        encoder_embedding_size=model_size["encoder_embedding_size"],
        decoder_embedding_size=model_size["decoder_embedding_size"],
        depth=model_size["decoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        num_heads=model_size["decoder_num_heads"],
        supported_modality_names=common.training_modalities,
        max_sequence_length=12,
        tokenization_config=common.tokenization_config,
    )
    return LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
    )


def build_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """Build the dataset config."""
    return OlmoEarthDatasetConfig(
        h5py_dir="/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_zstd_3_128_x_4/cdl_era5_10_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_wri_canopy_height_map/1138828",
        training_modalities=common.training_modalities,
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """Build the trainer config."""
    MAX_DURATION = Duration.epochs(300)
    METRICS_COLLECT_INTERVAL = 10
    CANCEL_CHECK_INTERVAL = 25
    LOAD_STRATEGY = LoadStrategy.if_available
    WANDB_USERNAME = "eai-ai2"  # nosec
    WANDB_PROJECT = "2026_03_16_era5_tokenization_ablation"
    PERMANENT_SAVE_INTERVAL = 5000
    EPHERMERAL_SAVE_INTERVAL = 250
    checkpointer_config = CheckpointerConfig(work_dir=common.save_folder)
    wandb_callback = OlmoEarthWandBCallback(
        name=common.run_name,
        project=WANDB_PROJECT,
        entity=WANDB_USERNAME,
        enabled=True,
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
                save_interval=PERMANENT_SAVE_INTERVAL,
                ephemeral_save_interval=EPHERMERAL_SAVE_INTERVAL,
            ),
        )
    )
    return trainer_config


def build_visualize_config(common: CommonComponents) -> OlmoEarthVisualizeConfig:
    """Build the visualize config."""
    return OlmoEarthVisualizeConfig(
        num_samples=None,
        output_dir=str(f"{common.save_folder}/visualizations"),
        std_multiplier=2.0,
    )


# ============================================================
# Experiment 1: ERA5 default tokenization (all 6 bands, 1 bandset)
# ============================================================


def build_common_exp1(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """ERA5 default tokenization (all 6 bands, 1 bandset)."""
    return _build_common(script, cmd, run_name, cluster, overrides, era5_override=None)


# ============================================================
# Experiment 2: ERA5 temp + dewpoint only (1 bandset, 2 bands)
# ============================================================


def build_common_exp2(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """ERA5 temp + dewpoint only (1 bandset, 2 bands)."""
    return _build_common(
        script, cmd, run_name, cluster, overrides, era5_override=ERA5_TEMP_DEWPOINT
    )


# ============================================================
# Experiment 3: ERA5 multi-bandset (4 bandsets)
# ============================================================


def build_common_exp3(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """ERA5 multi-bandset (4 bandsets)."""
    return _build_common(
        script, cmd, run_name, cluster, overrides, era5_override=ERA5_MULTI_BANDSET
    )


# ============================================================
# Experiment 4: ERA5 default tokenization + ERA5 weight 0.5
# ============================================================


def build_common_exp4(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """ERA5 default tokenization + ERA5 weight 0.5."""
    return _build_common(script, cmd, run_name, cluster, overrides, era5_override=None)


def build_train_module_exp4(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """Build train module with ERA5 weight 0.5."""
    return _build_train_module(common, era5_weight=0.5)


# ============================================================
# Experiment 5: ERA5 default tokenization + ERA5 weight 0.1
# ============================================================


def build_common_exp5(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """ERA5 default tokenization + ERA5 weight 0.1."""
    return _build_common(script, cmd, run_name, cluster, overrides, era5_override=None)


def build_train_module_exp5(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """Build train module with ERA5 weight 0.1."""
    return _build_train_module(common, era5_weight=0.1)


# ============================================================
# Entry point — select experiment via EXPERIMENT env var
# ============================================================

EXPERIMENTS: dict[str, tuple] = {
    "era5_default": (build_common_exp1, _build_train_module),
    "era5_temp_dewpoint": (build_common_exp2, _build_train_module),
    "era5_multi_bandset": (build_common_exp3, _build_train_module),
    "era5_default_weight_0.5": (build_common_exp4, build_train_module_exp4),
    "era5_default_weight_0.1": (build_common_exp5, build_train_module_exp5),
}

if __name__ == "__main__":
    exp_key = os.environ.get("EXPERIMENT", "era5_default")

    if exp_key not in EXPERIMENTS:
        print(f"Unknown experiment: {exp_key}")
        print(f"Available: {list(EXPERIMENTS.keys())}")
        sys.exit(1)

    common_builder, train_module_builder = EXPERIMENTS[exp_key]

    logger.info(f"Running experiment: {exp_key}")
    main(
        common_components_builder=common_builder,
        model_config_builder=_build_model,
        train_module_config_builder=train_module_builder,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=_build_dataloader,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )

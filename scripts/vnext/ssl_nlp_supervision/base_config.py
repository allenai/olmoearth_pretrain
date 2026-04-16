"""SSL + NLP Supervision experiment config.

Combines:
- Contrastive Latent MIM (self-supervised) on encode-decode modalities (S2, S1, Landsat)
- Text-conditioned NLP supervision on map modalities (WorldCover, SRTM, OSM, CDL, etc.)

Map modalities are NOT in the encoder's supported_modality_names — they are
purely ground-truth targets.  The NLP supervision decoder uses a CrossAttnDecoder
conditioned on SigLIP text embeddings to predict map values from encoder output.

Experiments:
- mode_b: text_condition_regression=True  (all map modalities through CLIP decoder)
- mode_a: text_condition_regression=False (regression via direct heads, classification via CLIP)
"""

import copy
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
from olmoearth_pretrain.nn.flexihelios import EncoderConfig, PredictorConfig
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.nn.nlp_supervision import NLPSupervisionDecoderConfig
from olmoearth_pretrain.nn.pooling import PoolingType
from olmoearth_pretrain.nn.tokenization import ModalityTokenization, TokenizationConfig
from olmoearth_pretrain.open_set.model.cross_attn_decoder import CrossAttnDecoderConfig
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
    NLPSupervisionTrainConfig,
)

logger = logging.getLogger(__name__)

MAX_PATCH_SIZE = 8
MIN_PATCH_SIZE = 1

# Encode-decode modalities (go through encoder + MIM decoder).
ENCODE_DECODE_MODALITIES = [
    Modality.SENTINEL2_L2A.name,
    Modality.SENTINEL1.name,
    Modality.LANDSAT.name,
]

# Map modalities (targets only — not in encoder, used for NLP supervision GT).
MAP_MODALITIES = [
    Modality.WORLDCOVER.name,
    Modality.SRTM.name,
    Modality.OPENSTREETMAP_RASTER.name,
    Modality.WRI_CANOPY_HEIGHT_MAP.name,
    Modality.CDL.name,
    Modality.WORLDCEREAL.name,
]

# All modalities the dataloader should provide.
ALL_TRAINING_MODALITIES = ENCODE_DECODE_MODALITIES + MAP_MODALITIES

# Regression modality names (for mode-a direct heads).
REGRESSION_MODALITY_NAMES = [
    Modality.SRTM.name,
    Modality.WRI_CANOPY_HEIGHT_MAP.name,
    Modality.CDL.name,
]

# --- Tokenization ---

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


def _tokenization_config() -> TokenizationConfig:
    return TokenizationConfig(
        overrides={
            "sentinel2_l2a": S2_SINGLE_BANDSET,
            "landsat": LANDSAT_SINGLE_BANDSET,
        }
    )


# --- Loss ---

_LOSS_CONFIG_DICT = {
    "type": "modality_patch_discrimination_masked_negatives",
    "tau": 0.1,
    "same_target_threshold": 0.999,
    "mask_negatives_for_modalities": [],  # No decode-only modalities in MIM path.
}

_CONTRASTIVE_CONFIG_DICT = {
    "type": "InfoNCE",
    "weight": 0.1,
}


# --- Masking ---


def _masking_config(
    tokenization_config: TokenizationConfig | None = None,
) -> MaskingConfig:
    return MaskingConfig(
        strategy_config={
            "type": "random_time_with_decode",
            "encode_ratio": 0.5,
            "decode_ratio": 0.5,
            "random_ratio": 0.5,
            "only_decode_modalities": MAP_MODALITIES,
        },
        tokenization_config=tokenization_config,
    )


# ---------------------------------------------------------------------------
# Builder functions matching the `main()` contract
# ---------------------------------------------------------------------------


def build_common_components(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """Build common components — dataloader loads ALL modalities (including maps as GT)."""
    config = build_common_components_default(script, cmd, run_name, cluster, overrides)
    config.training_modalities = ALL_TRAINING_MODALITIES
    config.tokenization_config = _tokenization_config()
    return config


def build_model_config(
    common: CommonComponents,
    text_condition_regression: bool = True,
    band_dropout_rate: float = 0.3,
) -> LatentMIMConfig:
    """Build the model config."""
    model_size = MODEL_SIZE_ARGS["base_shallow_decoder"]
    tokenization_config = common.tokenization_config or _tokenization_config()

    encoder_config = EncoderConfig(
        embedding_size=model_size["encoder_embedding_size"],
        num_heads=model_size["encoder_num_heads"],
        depth=model_size["encoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        supported_modality_names=ENCODE_DECODE_MODALITIES,
        max_patch_size=MAX_PATCH_SIZE,
        drop_path=0.1,
        max_sequence_length=12,
        tokenization_config=tokenization_config,
        band_dropout_rate=band_dropout_rate,
    )
    decoder_config = PredictorConfig(
        encoder_embedding_size=model_size["encoder_embedding_size"],
        decoder_embedding_size=model_size["decoder_embedding_size"],
        depth=model_size["decoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        num_heads=model_size["decoder_num_heads"],
        supported_modality_names=ENCODE_DECODE_MODALITIES,
        max_sequence_length=12,
        tokenization_config=tokenization_config,
    )

    nlp_decoder_config = NLPSupervisionDecoderConfig(
        decoder_config=CrossAttnDecoderConfig(
            dim=512,
            depth=4,
            num_heads=8,
            mlp_ratio=4.0,
            qk_norm=False,
            use_flash_attn=False,
        ),
        text_dim=1152,
        reference_modalities=("sentinel2_l2a", "sentinel1", "landsat"),
        text_condition_regression=text_condition_regression,
        supervision_weight=1.0,
        regression_modality_names=(
            REGRESSION_MODALITY_NAMES if not text_condition_regression else []
        ),
    )

    return LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        nlp_supervision_decoder_config=nlp_decoder_config,
    )


def build_train_module_config(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """Build the training module config."""
    tokenization_config = common.tokenization_config or _tokenization_config()

    return ContrastiveLatentMIMTrainModuleConfig(
        optim_config=AdamWConfig(lr=0.0001, weight_decay=0.02, fused=False),
        rank_microbatch_size=32,
        masking_config=_masking_config(tokenization_config),
        loss_config=LossConfig(loss_config=copy.deepcopy(_LOSS_CONFIG_DICT)),
        contrastive_config=LossConfig(
            loss_config=copy.deepcopy(_CONTRASTIVE_CONFIG_DICT)
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
        nlp_supervision_train_config=NLPSupervisionTrainConfig(
            text_encoder_name="google/siglip2-so400m-patch14-384",
            text_cache_dir="",
            sampler_k_pos=3,
            sampler_k_neg=3,
            sampler_seed=42,
            target_size_source="openstreetmap_raster",
            catalog_sources=MAP_MODALITIES,
        ),
    )


def build_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """Build the dataset config."""
    return OlmoEarthDatasetConfig(
        h5py_dir="/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_zstd_3_128_x_4/cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_worldpop_wri_canopy_height_map/1138828",
        training_modalities=common.training_modalities,
    )


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """Build the dataloader config."""
    tokenization_config = common.tokenization_config or _tokenization_config()

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
        masking_config=_masking_config(tokenization_config),
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """Build the trainer config."""
    MAX_DURATION = Duration.epochs(300)
    WANDB_PROJECT = "2026_04_ssl_nlp_supervision"

    checkpointer_config = CheckpointerConfig(work_dir=common.save_folder)
    wandb_callback = OlmoEarthWandBCallback(
        name=common.run_name,
        project=WANDB_PROJECT,
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
    }

    trainer_config = (
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
            CheckpointerCallback(
                save_interval=5000,
                ephemeral_save_interval=250,
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


# ---------------------------------------------------------------------------
# Experiment variants
# ---------------------------------------------------------------------------


def _build_model_mode_b(common: CommonComponents) -> LatentMIMConfig:
    """Mode (b): all map modalities through the CLIP decoder."""
    return build_model_config(common, text_condition_regression=True)


def _build_model_mode_a(common: CommonComponents) -> LatentMIMConfig:
    """Mode (a): regression via direct heads, classification via CLIP decoder."""
    return build_model_config(common, text_condition_regression=False)


EXPERIMENTS = {
    "ssl_nlp_mode_b": (
        build_common_components,
        _build_model_mode_b,
        build_train_module_config,
        build_dataloader_config,
    ),
    "ssl_nlp_mode_a": (
        build_common_components,
        _build_model_mode_a,
        build_train_module_config,
        build_dataloader_config,
    ),
}

if __name__ == "__main__":
    exp_key = os.environ.get("EXPERIMENT", "ssl_nlp_mode_b")

    if exp_key not in EXPERIMENTS:
        print(f"Unknown experiment: {exp_key}")
        print(f"Available: {list(EXPERIMENTS.keys())}")
        sys.exit(1)

    common_builder, model_builder, train_module_builder, dataloader_builder = (
        EXPERIMENTS[exp_key]
    )

    logger.info(f"Running experiment: {exp_key}")
    main(
        common_components_builder=common_builder,
        model_config_builder=model_builder,
        train_module_config_builder=train_module_builder,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=dataloader_builder,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )

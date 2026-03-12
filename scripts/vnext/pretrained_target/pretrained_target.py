"""Pretrained target encoder experiments.

Uses a frozen OlmoEarth-v1-Base model as the target encoder instead of an EMA copy
of the online encoder. Modeled on exp20 from single_bandset_masked_neg.py.

Experiments:
1. pretrained_target_no_dropout: No band dropout, full depth single pass
2. pretrained_target_random_band_dropout: Uniform(0, 0.3) band dropout, full depth single pass
3. pretrained_target_projection_only: Uniform(0, 0.3) band dropout, projection only (no transformer)
4. pretrained_target_per_modality: Uniform(0, 0.3) band dropout, full depth per-modality forward
"""

import copy
import logging
import sys
from pathlib import Path

# Add single_bandset_band_dropout directory to path for base_token_masked imports
sys.path.insert(0, str(Path(__file__).parent.parent / "single_bandset_band_dropout"))

from base_token_masked import (
    build_common_components as build_common_components_base,
)
from base_token_masked import (
    build_dataset_config,
    build_trainer_config,
    build_visualize_config,
)
from olmo_core.config import DType
from olmo_core.distributed.parallel.data_parallel import (
    DataParallelConfig,
    DataParallelType,
)
from olmo_core.optim import AdamWConfig
from olmo_core.optim.scheduler import CosWithWarmup

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig
from olmoearth_pretrain.internal.experiment import CommonComponents, SubCmd, main
from olmoearth_pretrain.internal.utils import MODEL_SIZE_ARGS
from olmoearth_pretrain.nn.flexi_vit import EncoderConfig, PredictorConfig
from olmoearth_pretrain.nn.pretrained_target_latent_mim import (
    PretrainedTargetLatentMIMConfig,
)
from olmoearth_pretrain.nn.tokenization import ModalityTokenization, TokenizationConfig
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import MaskingConfig
from olmoearth_pretrain.train.train_module.contrastive_latentmim import (
    ContrastiveLatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

MAX_PATCH_SIZE = 8
MIN_PATCH_SIZE = 1
RANDOM_BAND_DROPOUT_MAX_RATE = 0.3

# --- Modality lists ---

ONLY_DECODE_MODALITIES = [
    Modality.WORLDCOVER.name,
    Modality.SRTM.name,
    Modality.OPENSTREETMAP_RASTER.name,
    Modality.WRI_CANOPY_HEIGHT_MAP.name,
    Modality.CDL.name,
    Modality.WORLDCEREAL.name,
]

ONLY_DECODE_MODALITIES_WITH_NDVI_AND_ERA5 = ONLY_DECODE_MODALITIES + [
    Modality.NDVI.name,
    Modality.ERA5_10.name,
]

ENCODABLE_MODALITIES = [
    Modality.SENTINEL2_L2A.name,
    Modality.SENTINEL1.name,
    Modality.LANDSAT.name,
]

# --- Tokenization configs (single bandset for online encoder) ---

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

NDVI_SINGLE_BANDSET = ModalityTokenization(band_groups=[["ndvi"]])


def _single_bandset_tokenization_config() -> TokenizationConfig:
    return TokenizationConfig(
        overrides={
            "sentinel2_l2a": S2_SINGLE_BANDSET,
            "landsat": LANDSAT_SINGLE_BANDSET,
            "ndvi": NDVI_SINGLE_BANDSET,
        }
    )


# --- Loss configs ---

_LOSS_CONFIG_DICT = {
    "type": "modality_patch_discrimination_masked_negatives",
    "tau": 0.1,
    "same_target_threshold": 0.999,
    "mask_negatives_for_modalities": ONLY_DECODE_MODALITIES_WITH_NDVI_AND_ERA5,
}

_CONTRASTIVE_CONFIG_DICT = {
    "type": "InfoNCE",
    "weight": 0.1,
}


def _loss_config() -> LossConfig:
    return LossConfig(loss_config=copy.deepcopy(_LOSS_CONFIG_DICT))


def _contrastive_config() -> LossConfig:
    return LossConfig(loss_config=copy.deepcopy(_CONTRASTIVE_CONFIG_DICT))


# --- Masking config ---


def _masking_config(
    tokenization_config: TokenizationConfig | None = None,
) -> MaskingConfig:
    return MaskingConfig(
        strategy_config={
            "type": "random_time_with_decode",
            "encode_ratio": 0.5,
            "decode_ratio": 0.5,
            "random_ratio": 0.5,
            "only_decode_modalities": ONLY_DECODE_MODALITIES_WITH_NDVI_AND_ERA5,
        },
        tokenization_config=tokenization_config,
    )


# --- Common components ---


def _build_common(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    common = build_common_components_base(script, cmd, run_name, cluster, overrides)
    common.training_modalities = common.training_modalities + [
        Modality.NDVI.name,
        Modality.ERA5_10.name,
    ]
    common.tokenization_config = _single_bandset_tokenization_config()
    return common


# --- Train module ---


def _build_train_module(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    return ContrastiveLatentMIMTrainModuleConfig(
        optim_config=AdamWConfig(lr=0.0001, weight_decay=0.02, fused=False),
        rank_microbatch_size=32,
        masking_config=_masking_config(common.tokenization_config),
        loss_config=_loss_config(),
        contrastive_config=_contrastive_config(),
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


# --- Dataloader ---


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


# --- Model builder ---


def _build_model(
    common: CommonComponents,
    band_dropout_rate: float = 0.0,
    random_band_dropout: bool = False,
    projection_only: bool = False,
    per_modality_forward: bool = False,
) -> PretrainedTargetLatentMIMConfig:
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
        band_dropout_rate=band_dropout_rate,
        random_band_dropout=random_band_dropout,
    )
    # Decoder uses default multi-bandset tokenization (matching pre-trained model).
    # No tokenization_config override = default bandsets.
    decoder_config = PredictorConfig(
        encoder_embedding_size=model_size["encoder_embedding_size"],
        decoder_embedding_size=model_size["decoder_embedding_size"],
        depth=model_size["decoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        num_heads=model_size["decoder_num_heads"],
        supported_modality_names=common.training_modalities,
        max_sequence_length=12,
        output_embedding_size=768,  # Match OlmoEarth-v1-Base embedding size
    )
    return PretrainedTargetLatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        target_encoder_model_id="OlmoEarth-v1-Base",
        projection_only=projection_only,
        per_modality_forward=per_modality_forward,
        encodable_modality_names=ENCODABLE_MODALITIES,
    )


# ============================================================
# Experiment 1: No band dropout, full depth single pass
# ============================================================


def build_common_exp1(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """Build common components for exp1."""
    return _build_common(script, cmd, run_name, cluster, overrides)


def build_train_module_exp1(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """Build train module for exp1."""
    return _build_train_module(common)


def build_model_exp1(common: CommonComponents) -> PretrainedTargetLatentMIMConfig:
    """Build model for exp1."""
    return _build_model(common, band_dropout_rate=0.0, random_band_dropout=False)


def build_dataloader_exp1(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """Build dataloader for exp1."""
    return _build_dataloader(common)


# ============================================================
# Experiment 2: Random band dropout, full depth single pass
# ============================================================


def build_common_exp2(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """Build common components for exp2."""
    return _build_common(script, cmd, run_name, cluster, overrides)


def build_train_module_exp2(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """Build train module for exp2."""
    return _build_train_module(common)


def build_model_exp2(common: CommonComponents) -> PretrainedTargetLatentMIMConfig:
    """Build model for exp2."""
    return _build_model(
        common,
        band_dropout_rate=RANDOM_BAND_DROPOUT_MAX_RATE,
        random_band_dropout=True,
    )


def build_dataloader_exp2(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """Build dataloader for exp2."""
    return _build_dataloader(common)


# ============================================================
# Experiment 3: Random band dropout + projection only
# ============================================================


def build_common_exp3(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """Build common components for exp3."""
    return _build_common(script, cmd, run_name, cluster, overrides)


def build_train_module_exp3(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """Build train module for exp3."""
    return _build_train_module(common)


def build_model_exp3(common: CommonComponents) -> PretrainedTargetLatentMIMConfig:
    """Build model for exp3."""
    return _build_model(
        common,
        band_dropout_rate=RANDOM_BAND_DROPOUT_MAX_RATE,
        random_band_dropout=True,
        projection_only=True,
    )


def build_dataloader_exp3(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """Build dataloader for exp3."""
    return _build_dataloader(common)


# ============================================================
# Experiment 4: Random band dropout + per-modality forward
# ============================================================


def build_common_exp4(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """Build common components for exp4."""
    return _build_common(script, cmd, run_name, cluster, overrides)


def build_train_module_exp4(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """Build train module for exp4."""
    return _build_train_module(common)


def build_model_exp4(common: CommonComponents) -> PretrainedTargetLatentMIMConfig:
    """Build model for exp4."""
    return _build_model(
        common,
        band_dropout_rate=RANDOM_BAND_DROPOUT_MAX_RATE,
        random_band_dropout=True,
        per_modality_forward=True,
    )


def build_dataloader_exp4(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """Build dataloader for exp4."""
    return _build_dataloader(common)


# ============================================================
# Entry point — select experiment via EXPERIMENT env var or arg
# ============================================================

EXPERIMENTS = {
    "pretrained_target_no_dropout": (
        build_common_exp1,
        build_model_exp1,
        build_train_module_exp1,
        build_dataloader_exp1,
    ),
    "pretrained_target_random_band_dropout": (
        build_common_exp2,
        build_model_exp2,
        build_train_module_exp2,
        build_dataloader_exp2,
    ),
    "pretrained_target_projection_only": (
        build_common_exp3,
        build_model_exp3,
        build_train_module_exp3,
        build_dataloader_exp3,
    ),
    "pretrained_target_per_modality": (
        build_common_exp4,
        build_model_exp4,
        build_train_module_exp4,
        build_dataloader_exp4,
    ),
}

if __name__ == "__main__":
    import os

    exp_key = os.environ.get("EXPERIMENT", "pretrained_target_random_band_dropout")

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

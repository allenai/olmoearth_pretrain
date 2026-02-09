"""Single bandset experiments with masked-negatives loss.

Seven experiments:
1. modality_cross_random masking + single bandset S2 (all 12 bands) / Landsat + masked neg loss
2. random_with_decode masking + single bandset S2 (all 12 bands) / Landsat + masked neg loss
3. modality_cross_random masking + single bandset S2 (no 60m: 10 bands) / Landsat + masked neg loss
4. modality_cross_random masking + single bandset S2 (10m only: 4 bands) / Landsat + masked neg loss
5. modality_cross_random masking + single bandset S2 (all 12) / Landsat + masked neg loss + band dropout 0.3
6. modality_cross_random masking + single bandset S2 (all 12) / Landsat + masked neg loss + band dropout 0.5
7. random_with_decode masking + single bandset S2/Landsat + ERA5 decode-only (1 bandset) + masked neg loss
"""

import copy
import logging
import sys

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
from olmoearth_pretrain.nn.flexihelios import EncoderConfig, PredictorConfig
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.nn.tokenization import ModalityTokenization, TokenizationConfig
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import MaskingConfig
from olmoearth_pretrain.train.train_module.contrastive_latentmim import (
    ContrastiveLatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

MAX_PATCH_SIZE = 8
MIN_PATCH_SIZE = 1


# --- Loss configs ---

ONLY_DECODE_MODALITIES = [
    Modality.WORLDCOVER.name,
    Modality.SRTM.name,
    Modality.OPENSTREETMAP_RASTER.name,
    Modality.WRI_CANOPY_HEIGHT_MAP.name,
    Modality.CDL.name,
    Modality.WORLDCEREAL.name,
]

ONLY_DECODE_MODALITIES_WITH_ERA5 = ONLY_DECODE_MODALITIES + [Modality.ERA5_10.name]

_LOSS_CONFIG_DICT = {
    "type": "modality_patch_discrimination_masked_negatives",
    "tau": 0.1,
    "same_target_threshold": 0.999,
    "mask_negatives_for_modalities": ONLY_DECODE_MODALITIES,
}

_CONTRASTIVE_CONFIG_DICT = {
    "type": "InfoNCE",
    "weight": 0.1,
}


def _loss_config() -> LossConfig:
    return LossConfig(loss_config=copy.deepcopy(_LOSS_CONFIG_DICT))


def _contrastive_config() -> LossConfig:
    return LossConfig(loss_config=copy.deepcopy(_CONTRASTIVE_CONFIG_DICT))


# --- Tokenization configs ---

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

S2_SINGLE_BANDSET_NO_60M = ModalityTokenization(
    band_groups=[
        ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12"],
    ]
)

S2_SINGLE_BANDSET_10M_ONLY = ModalityTokenization(
    band_groups=[
        ["B02", "B03", "B04", "B08"],
    ]
)

LANDSAT_SINGLE_BANDSET = ModalityTokenization(
    band_groups=[
        ["B8", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"],
    ]
)


def _tokenization_config(
    s2_config: ModalityTokenization = S2_SINGLE_BANDSET,
) -> TokenizationConfig:
    return TokenizationConfig(
        overrides={
            "sentinel2_l2a": s2_config,
            "landsat": LANDSAT_SINGLE_BANDSET,
        }
    )


def _masking_config(
    masking_type: str, tokenization_config: TokenizationConfig | None = None
) -> MaskingConfig:
    strategy: dict = {
        "type": masking_type,
        "encode_ratio": 0.5,
        "decode_ratio": 0.5,
        "only_decode_modalities": ONLY_DECODE_MODALITIES,
    }
    if masking_type == "modality_cross_random":
        strategy["allow_encoding_decoding_same_bandset"] = True
    return MaskingConfig(
        strategy_config=strategy,
        tokenization_config=tokenization_config,
    )


def _build_common(
    script: str,
    cmd: SubCmd,
    run_name: str,
    cluster: str,
    overrides: list[str],
    s2_config: ModalityTokenization = S2_SINGLE_BANDSET,
) -> CommonComponents:
    common = build_common_components_base(script, cmd, run_name, cluster, overrides)
    common.tokenization_config = _tokenization_config(s2_config=s2_config)
    return common


def _build_train_module(
    common: CommonComponents,
    masking_type: str,
) -> ContrastiveLatentMIMTrainModuleConfig:
    return ContrastiveLatentMIMTrainModuleConfig(
        optim_config=AdamWConfig(lr=0.0001, weight_decay=0.02, fused=False),
        rank_microbatch_size=32,
        masking_config=_masking_config(masking_type, common.tokenization_config),
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


def _build_model(
    common: CommonComponents,
    band_dropout_rate: float = 0.0,
) -> LatentMIMConfig:
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


def _build_dataloader(
    common: CommonComponents,
    masking_type: str,
) -> OlmoEarthDataLoaderConfig:
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
        masking_config=_masking_config(masking_type, common.tokenization_config),
    )


# ============================================================
# Experiment 1: modality_cross_random + single_bandset bandsets
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
    return _build_train_module(common, "modality_cross_random")


def build_dataloader_exp1(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """Build dataloader for exp1."""
    return _build_dataloader(common, "modality_cross_random")


build_model_exp1 = _build_model


# ============================================================
# Experiment 2: random_with_decode + single_bandset bandsets
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
    return _build_train_module(common, "random_with_decode")


def build_dataloader_exp2(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """Build dataloader for exp2."""
    return _build_dataloader(common, "random_with_decode")


build_model_exp2 = _build_model


# ============================================================
# Experiment 3: modality_cross_random + single_bandset bandsets (no B01/B09)
# ============================================================


def build_common_exp3(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """Build common components for exp3."""
    return _build_common(
        script, cmd, run_name, cluster, overrides, s2_config=S2_SINGLE_BANDSET_NO_60M
    )


def build_train_module_exp3(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """Build train module for exp3."""
    return _build_train_module(common, "modality_cross_random")


def build_dataloader_exp3(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """Build dataloader for exp3."""
    return _build_dataloader(common, "modality_cross_random")


build_model_exp3 = _build_model


# ============================================================
# Experiment 4: modality_cross_random + 10m bands only (no 20m/60m)
# ============================================================


def build_common_exp4(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """Build common components for exp4."""
    return _build_common(
        script, cmd, run_name, cluster, overrides, s2_config=S2_SINGLE_BANDSET_10M_ONLY
    )


def build_train_module_exp4(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """Build train module for exp4."""
    return _build_train_module(common, "modality_cross_random")


def build_dataloader_exp4(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """Build dataloader for exp4."""
    return _build_dataloader(common, "modality_cross_random")


build_model_exp4 = _build_model


# ============================================================
# Experiment 5: modality_cross_random + single_bandset + band dropout (0.3)
# ============================================================

BAND_DROPOUT_RATE = 0.3


def build_common_exp5(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """Build common components for exp5."""
    return _build_common(script, cmd, run_name, cluster, overrides)


def build_train_module_exp5(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """Build train module for exp5."""
    return _build_train_module(common, "modality_cross_random")


def build_dataloader_exp5(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """Build dataloader for exp5."""
    return _build_dataloader(common, "modality_cross_random")


def build_model_exp5(common: CommonComponents) -> LatentMIMConfig:
    """Build model for exp5 with band dropout 0.3."""
    return _build_model(common, band_dropout_rate=BAND_DROPOUT_RATE)


# ============================================================
# Experiment 6: modality_cross_random + single_bandset + band dropout (0.5)
# ============================================================

BAND_DROPOUT_RATE_HIGH = 0.5


def build_common_exp6(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """Build common components for exp6."""
    return _build_common(script, cmd, run_name, cluster, overrides)


def build_train_module_exp6(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """Build train module for exp6."""
    return _build_train_module(common, "modality_cross_random")


def build_dataloader_exp6(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """Build dataloader for exp6."""
    return _build_dataloader(common, "modality_cross_random")


def build_model_exp6(common: CommonComponents) -> LatentMIMConfig:
    """Build model for exp6 with band dropout 0.5."""
    return _build_model(common, band_dropout_rate=BAND_DROPOUT_RATE_HIGH)


# ============================================================
# Experiment 7: random_with_decode + single_bandset + ERA5 decode-only
# ============================================================


def _masking_config_era5(
    tokenization_config: TokenizationConfig | None = None,
) -> MaskingConfig:
    return MaskingConfig(
        strategy_config={
            "type": "random_with_decode",
            "encode_ratio": 0.5,
            "decode_ratio": 0.5,
            "only_decode_modalities": ONLY_DECODE_MODALITIES_WITH_ERA5,
        },
        tokenization_config=tokenization_config,
    )


def _loss_config_era5() -> LossConfig:
    return LossConfig(
        loss_config={
            "type": "modality_patch_discrimination_masked_negatives",
            "tau": 0.1,
            "same_target_threshold": 0.999,
            "mask_negatives_for_modalities": ONLY_DECODE_MODALITIES_WITH_ERA5,
        }
    )


def build_common_exp7(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """Build common components for exp7 (ERA5 decode-only)."""
    common = _build_common(script, cmd, run_name, cluster, overrides)
    common.training_modalities = common.training_modalities + [Modality.ERA5_10.name]
    return common


def build_train_module_exp7(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """Build train module for exp7 (ERA5 decode-only)."""
    return ContrastiveLatentMIMTrainModuleConfig(
        optim_config=AdamWConfig(lr=0.0001, weight_decay=0.02, fused=False),
        rank_microbatch_size=32,
        masking_config=_masking_config_era5(common.tokenization_config),
        loss_config=_loss_config_era5(),
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


def build_dataloader_exp7(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """Build dataloader for exp7 (ERA5 decode-only)."""
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
        masking_config=_masking_config_era5(common.tokenization_config),
    )


build_model_exp7 = _build_model


# ============================================================
# Entry point — select experiment via EXPERIMENT env var or arg
# ============================================================

EXPERIMENTS = {
    "single_bandset_cross_random_masked_neg": (
        build_common_exp1,
        build_model_exp1,
        build_train_module_exp1,
        build_dataloader_exp1,
    ),
    "single_bandset_random_decode_masked_neg": (
        build_common_exp2,
        build_model_exp2,
        build_train_module_exp2,
        build_dataloader_exp2,
    ),
    "single_bandset_no60m_cross_random_masked_neg": (
        build_common_exp3,
        build_model_exp3,
        build_train_module_exp3,
        build_dataloader_exp3,
    ),
    "single_bandset_10m_only_cross_random_masked_neg": (
        build_common_exp4,
        build_model_exp4,
        build_train_module_exp4,
        build_dataloader_exp4,
    ),
    "single_bandset_band_dropout_cross_random_masked_neg": (
        build_common_exp5,
        build_model_exp5,
        build_train_module_exp5,
        build_dataloader_exp5,
    ),
    "single_bandset_band_dropout_0.5_cross_random_masked_neg": (
        build_common_exp6,
        build_model_exp6,
        build_train_module_exp6,
        build_dataloader_exp6,
    ),
    "single_bandset_era5_decode_only_masked_neg": (
        build_common_exp7,
        build_model_exp7,
        build_train_module_exp7,
        build_dataloader_exp7,
    ),
}

if __name__ == "__main__":
    import os

    exp_key = os.environ.get("EXPERIMENT", "single_bandset_cross_random_masked_neg")

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

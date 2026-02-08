"""Single bandset experiments with masked-negatives loss.

Three experiments:
1. modality_cross_random masking + single_bandset S2/Landsat bandsets + masked neg loss
2. random_with_decode masking + single_bandset S2/Landsat bandsets + masked neg loss
3. modality_cross_random masking + single_bandset S2/Landsat (no B01/B09) + masked neg loss
"""

import copy
import logging
import sys

from base_token_masked import (
    build_common_components as build_common_components_base,
    build_dataloader_config as build_dataloader_config_base,
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

S2_COLLAPSED_ALL = ModalityTokenization(
    band_groups=[
        ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"],
    ]
)

S2_COLLAPSED_NO_60M = ModalityTokenization(
    band_groups=[
        ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12"],
    ]
)

LANDSAT_COLLAPSED = ModalityTokenization(
    band_groups=[
        ["B8", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"],
    ]
)


def _tokenization_config(drop_60m: bool = False) -> TokenizationConfig:
    s2 = S2_COLLAPSED_NO_60M if drop_60m else S2_COLLAPSED_ALL
    return TokenizationConfig(
        overrides={
            "sentinel2_l2a": s2,
            "landsat": LANDSAT_COLLAPSED,
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
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str],
    drop_60m: bool = False,
) -> CommonComponents:
    common = build_common_components_base(script, cmd, run_name, cluster, overrides)
    common.tokenization_config = _tokenization_config(drop_60m=drop_60m)
    return common


def _build_train_module(
    common: CommonComponents, masking_type: str,
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


# ============================================================
# Experiment 1: modality_cross_random + single_bandset bandsets
# ============================================================

def build_common_exp1(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    return _build_common(script, cmd, run_name, cluster, overrides)


def build_train_module_exp1(common: CommonComponents) -> ContrastiveLatentMIMTrainModuleConfig:
    return _build_train_module(common, "modality_cross_random")


def build_dataloader_exp1(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    return build_dataloader_config_base(common)


build_model_exp1 = _build_model


# ============================================================
# Experiment 2: random_with_decode + single_bandset bandsets
# ============================================================

def build_common_exp2(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    return _build_common(script, cmd, run_name, cluster, overrides)


def build_train_module_exp2(common: CommonComponents) -> ContrastiveLatentMIMTrainModuleConfig:
    return _build_train_module(common, "random_with_decode")


def build_dataloader_exp2(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    return build_dataloader_config_base(common)


build_model_exp2 = _build_model


# ============================================================
# Experiment 3: modality_cross_random + single_bandset bandsets (no B01/B09)
# ============================================================

def build_common_exp3(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    return _build_common(script, cmd, run_name, cluster, overrides, drop_60m=True)


def build_train_module_exp3(common: CommonComponents) -> ContrastiveLatentMIMTrainModuleConfig:
    return _build_train_module(common, "modality_cross_random")


def build_dataloader_exp3(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    return build_dataloader_config_base(common)


build_model_exp3 = _build_model


# ============================================================
# Entry point — select experiment via EXPERIMENT env var or arg
# ============================================================

EXPERIMENTS = {
    "single_bandset_cross_random_masked_neg": (
        build_common_exp1, build_model_exp1, build_train_module_exp1,
        build_dataloader_exp1,
    ),
    "single_bandset_random_decode_masked_neg": (
        build_common_exp2, build_model_exp2, build_train_module_exp2,
        build_dataloader_exp2,
    ),
    "single_bandset_no60m_cross_random_masked_neg": (
        build_common_exp3, build_model_exp3, build_train_module_exp3,
        build_dataloader_exp3,
    ),
}

if __name__ == "__main__":
    import os

    exp_key = os.environ.get("EXPERIMENT", "single_bandset_cross_random_masked_neg")

    if exp_key not in EXPERIMENTS:
        print(f"Unknown experiment: {exp_key}")
        print(f"Available: {list(EXPERIMENTS.keys())}")
        sys.exit(1)

    common_builder, model_builder, train_module_builder, dataloader_builder = EXPERIMENTS[exp_key]

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

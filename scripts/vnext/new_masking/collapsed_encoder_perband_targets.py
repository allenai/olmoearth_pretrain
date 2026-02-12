"""Collapsed bandsets for encoder, per-band (unique) bandsets for targets.

This experiment uses collapsed (all-bands-in-one-token) tokenization for the
online encoder, while the target encoder uses per-band tokenization where each
band is its own bandset/token. A RetokenizingPredictor bridges the two.

Change from baseline:
- Online encoder: S2 uses 1 token (all 12 bands), S1 uses 1 token (2 bands),
  Landsat uses 1 token (all 11 bands)
- Target encoder: S2 uses 12 tokens (one per band), S1 uses 2 tokens (one per band),
  Landsat uses 11 tokens (one per band)
- Decoder: RetokenizingPredictor that re-tokenizes from collapsed -> per-band
- Masking: performed on collapsed bandsets (encoder's tokenization)
"""

import logging

from new_masking_script import (
    build_common_components as build_common_components_base,
)
from new_masking_script import (
    build_dataloader_config,
    build_dataset_config,
    build_train_module_config,
    build_trainer_config,
    build_visualize_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, SubCmd, main
from olmoearth_pretrain.internal.utils import MODEL_SIZE_ARGS
from olmoearth_pretrain.nn.flexihelios import (
    EncoderConfig,
    PredictorConfig,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.nn.retokenizing_predictor import RetokenizingPredictorConfig
from olmoearth_pretrain.nn.tokenization import (
    ModalityTokenization,
    TokenizationConfig,
)

logger = logging.getLogger(__name__)

MAX_PATCH_SIZE = 8
MIN_PATCH_SIZE = 1

# Per-band tokenization: each band becomes its own token
_S2_BANDS = [
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
]
_S1_BANDS = ["vv", "vh"]
_LANDSAT_BANDS = ["B8", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"]


def _get_collapsed_tokenization() -> TokenizationConfig:
    """Collapsed tokenization: all bands in one token per modality."""
    return TokenizationConfig(
        overrides={
            "sentinel2_l2a": ModalityTokenization(band_groups=[_S2_BANDS]),
            "sentinel1": ModalityTokenization(band_groups=[_S1_BANDS]),
            "landsat": ModalityTokenization(band_groups=[_LANDSAT_BANDS]),
        }
    )


def _get_perband_tokenization() -> TokenizationConfig:
    """Per-band tokenization: one band per token."""
    return TokenizationConfig(
        overrides={
            "sentinel2_l2a": ModalityTokenization(band_groups=[[b] for b in _S2_BANDS]),
            "sentinel1": ModalityTokenization(band_groups=[[b] for b in _S1_BANDS]),
            "landsat": ModalityTokenization(band_groups=[[b] for b in _LANDSAT_BANDS]),
        }
    )


def build_common_components(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """Build common components with collapsed tokenization for the encoder."""
    common = build_common_components_base(script, cmd, run_name, cluster, overrides)
    # Masking and dataloader use the collapsed (encoder) tokenization
    common.tokenization_config = _get_collapsed_tokenization()
    return common


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build model config: collapsed encoder, per-band target encoder."""
    model_size = MODEL_SIZE_ARGS["base_shallow_decoder"]
    collapsed_tokenization = _get_collapsed_tokenization()
    perband_tokenization = _get_perband_tokenization()

    # Online encoder uses collapsed bandsets
    encoder_config = EncoderConfig(
        embedding_size=model_size["encoder_embedding_size"],
        num_heads=model_size["encoder_num_heads"],
        depth=model_size["encoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        supported_modality_names=common.training_modalities,
        max_patch_size=MAX_PATCH_SIZE,
        drop_path=0.1,
        max_sequence_length=12,
        tokenization_config=collapsed_tokenization,
    )

    # Target encoder uses per-band bandsets
    target_encoder_config = EncoderConfig(
        embedding_size=model_size["encoder_embedding_size"],
        num_heads=model_size["encoder_num_heads"],
        depth=model_size["encoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        supported_modality_names=common.training_modalities,
        max_patch_size=MAX_PATCH_SIZE,
        drop_path=0.1,
        max_sequence_length=12,
        tokenization_config=perband_tokenization,
    )

    # Decoder: RetokenizingPredictor bridges collapsed -> per-band
    inner_predictor_config = PredictorConfig(
        encoder_embedding_size=model_size["encoder_embedding_size"],
        decoder_embedding_size=model_size["decoder_embedding_size"],
        depth=model_size["decoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        num_heads=model_size["decoder_num_heads"],
        supported_modality_names=common.training_modalities,
        max_sequence_length=12,
        tokenization_config=perband_tokenization,
    )
    decoder_config = RetokenizingPredictorConfig(
        predictor_config=inner_predictor_config,
        input_tokenization_config=collapsed_tokenization,
    )

    model_config = LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        target_encoder_config=target_encoder_config,
    )
    return model_config


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

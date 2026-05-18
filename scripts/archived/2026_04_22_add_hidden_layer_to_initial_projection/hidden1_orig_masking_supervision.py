"""hidden1 + orig masking + pixel supervision head on decode-only modalities."""

import logging

from hidden1 import (
    BAND_DROPOUT_MODALITIES,
    MAX_PATCH_SIZE,
    PATCH_EMBED_HIDDEN_SIZES,
    RANDOM_BAND_DROPOUT_MAX_RATE,
    build_common_components,
    build_dataset_config,
    build_trainer_config,
    build_visualize_config,
)
from hidden1_orig_masking import build_dataloader_config, build_train_module_config

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.internal.utils import MODEL_SIZE_ARGS
from olmoearth_pretrain.nn.flexihelios import EncoderConfig, PredictorConfig
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.nn.supervision_head import (
    SupervisionHeadConfig,
    SupervisionModalityConfig,
    SupervisionTaskType,
)

logger = logging.getLogger(__name__)

SUPERVISION_WEIGHT = 0.1
TASK_TYPE_WEIGHTS = {
    SupervisionTaskType.CLASSIFICATION: 0.1,
    SupervisionTaskType.BINARY_CLASSIFICATION: 0.1,
    SupervisionTaskType.REGRESSION: 1.0,
}

WORLDCOVER_CLASS_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]

# USDA NASS CDL crop codes (public legend). Raw values are uint8 codes in [0, 255];
# they get normalized as raw/200 by the placeholder computed norm config for CDL
# (mean=100, std=50, std_multiplier=2 → min=0, max=200). If the CDL norm config
# is updated, this divisor must change to match.
_CDL_CODES = [
    1,
    2,
    3,
    4,
    5,
    6,
    10,
    11,
    12,
    13,
    14,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    74,
    75,
    76,
    77,
    81,
    82,
    83,
    87,
    88,
    92,
    111,
    112,
    121,
    122,
    123,
    124,
    131,
    141,
    142,
    143,
    152,
    176,
    190,
    195,
    204,
    205,
    206,
    207,
    208,
    209,
    210,
    211,
    212,
    213,
    214,
    215,
    216,
    217,
    218,
    219,
    220,
    221,
    222,
    223,
    224,
    225,
    226,
    227,
    228,
    229,
    230,
    231,
    232,
    233,
    234,
    235,
    236,
    237,
    238,
    239,
    240,
    241,
    242,
    243,
    244,
    245,
    246,
    247,
    248,
    249,
    250,
    254,
]
CDL_CLASS_VALUES = [code / 200 for code in _CDL_CODES]

SUPERVISION_MODALITY_CONFIGS = {
    "worldcover": SupervisionModalityConfig(
        task_type=SupervisionTaskType.CLASSIFICATION,
        num_output_channels=11,
        weight=SUPERVISION_WEIGHT
        * TASK_TYPE_WEIGHTS[SupervisionTaskType.CLASSIFICATION],
        class_values=WORLDCOVER_CLASS_VALUES,
    ),
    "srtm": SupervisionModalityConfig(
        task_type=SupervisionTaskType.REGRESSION,
        num_output_channels=1,
        weight=SUPERVISION_WEIGHT * TASK_TYPE_WEIGHTS[SupervisionTaskType.REGRESSION],
        norm_pix_loss=False,
    ),
    "openstreetmap_raster": SupervisionModalityConfig(
        task_type=SupervisionTaskType.BINARY_CLASSIFICATION,
        num_output_channels=30,
        weight=SUPERVISION_WEIGHT
        * TASK_TYPE_WEIGHTS[SupervisionTaskType.BINARY_CLASSIFICATION],
    ),
    "wri_canopy_height_map": SupervisionModalityConfig(
        task_type=SupervisionTaskType.REGRESSION,
        num_output_channels=1,
        weight=SUPERVISION_WEIGHT * TASK_TYPE_WEIGHTS[SupervisionTaskType.REGRESSION],
        norm_pix_loss=False,
    ),
    "cdl": SupervisionModalityConfig(
        task_type=SupervisionTaskType.CLASSIFICATION,
        num_output_channels=len(CDL_CLASS_VALUES),
        weight=SUPERVISION_WEIGHT
        * TASK_TYPE_WEIGHTS[SupervisionTaskType.CLASSIFICATION],
        class_values=CDL_CLASS_VALUES,
    ),
    "worldcereal": SupervisionModalityConfig(
        task_type=SupervisionTaskType.BINARY_CLASSIFICATION,
        num_output_channels=8,
        weight=SUPERVISION_WEIGHT
        * TASK_TYPE_WEIGHTS[SupervisionTaskType.BINARY_CLASSIFICATION],
    ),
}


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
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
        tokenization_config=common.tokenization_config,
        band_dropout_rate=RANDOM_BAND_DROPOUT_MAX_RATE,
        random_band_dropout=True,
        band_dropout_modalities=BAND_DROPOUT_MODALITIES,
        patch_embed_hidden_sizes=PATCH_EMBED_HIDDEN_SIZES,
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
    supervision_head_config = SupervisionHeadConfig(
        modality_configs=SUPERVISION_MODALITY_CONFIGS,
    )
    return LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        supervision_head_config=supervision_head_config,
    )


def run() -> None:
    """Run the experiment."""
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )


if __name__ == "__main__":
    run()

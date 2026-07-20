"""Register-grid supervision for the v1.2 register-bottleneck runs.

Adds a :class:`SupervisionHead` with ``register_supervision=True``: the per-modality
heads read the ENCODER's register grid (the Perceiver bottleneck) instead of the
decoder tokens, so the supervision gradient flows straight into the bottleneck --
the representation the decoder and downstream probes consume. Two arms:

* ``regsup``        -- pixel supervision of the decode-only map modalities
  (worldcover, srtm, openstreetmap_raster, wri_canopy_height_map, cdl, worldcereal).
  These are already decode-only training modalities in ``base.py``, so their raw
  targets are in every batch; only the head config is new.
* ``regsup_latlon`` -- additionally regresses the sample's location from the
  MEAN-POOLED register grid, as cartesian coordinates on the unit sphere
  (see ``supervision_head._latlon_unit_xyz_target``). Location is a supervised
  TARGET, never an input: a shortcut-free way to make the registers location-aware
  without requiring coordinates at inference.

Weights follow the archived register-salience precedent
(``hidden1_supervision_register_bottleneck.py``): a low base weight so this is an
inductive nudge, not a competing learning signal, with classification losses scaled
down a further 10x to balance their larger magnitude against L1/MSE. Effective
weights: regression (srtm, canopy, latlon) 0.01; classification / BCE 0.001.

latlon plumbing (the ``regsup_latlon`` arm): latlon must reach the batch without
becoming a model input. It is appended to the DATASET's ``training_modalities``
(``read_h5_file`` only loads listed keys; latlons are stored in every h5 file) and
to the masking strategy's ``only_decode_modalities`` (all-DECODE, no encode split).
The encoder/decoder ``supported_modality_names`` are unchanged, and the tokenizer
intersects batch modalities with supported names, so latlon can never leak into the
model -- only the supervision head reads ``batch.latlon``.

The class-value tables are copied from the archived supervision scripts and assume
the global computed.json/predefined.json normalization (worldcover values map the
11 classes; CDL codes are normalized as ``code / 200``).
"""

import logging

from base import ONLY_DECODE_MODALITIES, _masking_config
from base import build_dataset_config as _base_build_dataset_config
from regbtl_v1_2_faster_common import build_faster_train_module_config
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_dataloader_config as _1fwd_build_dataloader_config,
)

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig
from olmoearth_pretrain.internal.experiment import CommonComponents
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.nn.supervision_head import (
    LATLON_TARGET_DIM,
    SupervisionHeadConfig,
    SupervisionModalityConfig,
    SupervisionTaskType,
)
from olmoearth_pretrain.nn.tokenization import TokenizationConfig
from olmoearth_pretrain.train.masking import MaskingConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

# Low base weight: a register-salience nudge, not a learning signal.
SUPERVISION_WEIGHT = 0.01
# Classification/BCE losses run ~10x larger than the L1/MSE regressions, so they are
# scaled down to contribute comparably (same ratios as the archived supervision runs).
TASK_TYPE_WEIGHTS = {
    SupervisionTaskType.CLASSIFICATION: 0.1,
    SupervisionTaskType.BINARY_CLASSIFICATION: 0.1,
    SupervisionTaskType.REGRESSION: 1.0,
}

WORLDCOVER_CLASS_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]

# USDA NASS CDL crop codes (public legend). Raw values are uint8 codes in [0, 255];
# they get normalized as raw/200 by the placeholder computed norm config for CDL
# (mean=100, std=50, std_multiplier=2 -> min=0, max=200). If the CDL norm config
# is updated, this divisor must change to match. Codes 81/88 are pipeline sentinels
# kept for head-dim compatibility with earlier checkpoints (see the archived
# hidden1_supervision.py TODO).
_CDL_CODES = [
    *range(1, 7),
    *range(10, 15),
    *range(21, 40),
    *range(41, 62),
    *range(63, 73),
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
    *range(204, 251),
    254,
]
CDL_CLASS_VALUES = [code / 200 for code in _CDL_CODES]


def build_supervision_head_config(
    *, include_latlon: bool, base_weight: float = SUPERVISION_WEIGHT
) -> SupervisionHeadConfig:
    """Register-grid supervision head config over the decode-only map modalities.

    ``include_latlon`` adds the unit-sphere location regression read from the
    mean-pooled register grid. ``base_weight`` overrides the low ``SUPERVISION_WEIGHT``
    nudge (kept as the default); it is still scaled per-task by ``TASK_TYPE_WEIGHTS``.
    """

    def _weight(task_type: SupervisionTaskType) -> float:
        return base_weight * TASK_TYPE_WEIGHTS[task_type]

    modality_configs = {
        "worldcover": SupervisionModalityConfig(
            task_type=SupervisionTaskType.CLASSIFICATION,
            num_output_channels=len(WORLDCOVER_CLASS_VALUES),
            weight=_weight(SupervisionTaskType.CLASSIFICATION),
            class_values=WORLDCOVER_CLASS_VALUES,
        ),
        # srtm is the full terrain modality [elevation, slope, aspect_sin, aspect_cos],
        # so the register head regresses all of its bands (num_output_channels tracks
        # Modality.SRTM.num_bands rather than being hard-coded to 1).
        "srtm": SupervisionModalityConfig(
            task_type=SupervisionTaskType.REGRESSION,
            num_output_channels=Modality.SRTM.num_bands,
            weight=_weight(SupervisionTaskType.REGRESSION),
            regression_loss_type="l1",
        ),
        "openstreetmap_raster": SupervisionModalityConfig(
            task_type=SupervisionTaskType.BINARY_CLASSIFICATION,
            num_output_channels=30,
            weight=_weight(SupervisionTaskType.BINARY_CLASSIFICATION),
            pos_weight=True,
        ),
        "wri_canopy_height_map": SupervisionModalityConfig(
            task_type=SupervisionTaskType.REGRESSION,
            num_output_channels=1,
            weight=_weight(SupervisionTaskType.REGRESSION),
            regression_loss_type="l1",
        ),
        "cdl": SupervisionModalityConfig(
            task_type=SupervisionTaskType.CLASSIFICATION,
            num_output_channels=len(CDL_CLASS_VALUES),
            weight=_weight(SupervisionTaskType.CLASSIFICATION),
            class_values=CDL_CLASS_VALUES,
        ),
        "worldcereal": SupervisionModalityConfig(
            task_type=SupervisionTaskType.BINARY_CLASSIFICATION,
            num_output_channels=8,
            weight=_weight(SupervisionTaskType.BINARY_CLASSIFICATION),
            pos_weight=True,
        ),
    }
    if include_latlon:
        modality_configs[Modality.LATLON.name] = SupervisionModalityConfig(
            task_type=SupervisionTaskType.REGRESSION,
            num_output_channels=LATLON_TARGET_DIM,
            weight=_weight(SupervisionTaskType.REGRESSION),
        )
    return SupervisionHeadConfig(
        modality_configs=modality_configs,
        register_supervision=True,
    )


def add_register_supervision(
    config: LatentMIMConfig,
    *,
    include_latlon: bool,
    base_weight: float = SUPERVISION_WEIGHT,
) -> LatentMIMConfig:
    """Attach the register-grid supervision head to a regbtl model config."""
    config.supervision_head_config = build_supervision_head_config(
        include_latlon=include_latlon, base_weight=base_weight
    )
    return config


def _latlon_masking_config(
    tokenization_config: TokenizationConfig | None,
) -> MaskingConfig:
    """Base masking config with latlon added to the only-decode modalities.

    only_decode marks every non-missing latlon token DECODE, keeping it out of the
    encode split; since neither the encoder nor the decoder supports latlon, the
    mask is inert and the modality just rides along in the batch for supervision.
    """
    config = _masking_config(tokenization_config)
    config.strategy_config["only_decode_modalities"] = [
        *ONLY_DECODE_MODALITIES,
        Modality.LATLON.name,
    ]
    return config


def build_latlon_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """Base dataset config, additionally loading latlon from the h5 files."""
    config = _base_build_dataset_config(common)
    config.training_modalities = [
        *config.training_modalities,
        Modality.LATLON.name,
    ]
    return config


def build_latlon_dataloader_config(
    common: CommonComponents,
) -> OlmoEarthDataLoaderConfig:
    """Single-view (1fwd) dataloader whose masking knows latlon is decode-only."""
    config = _1fwd_build_dataloader_config(common)
    config.masking_config = _latlon_masking_config(common.tokenization_config)
    return config


def build_latlon_train_module_config(
    common: CommonComponents,
) -> LatentMIMTrainModuleConfig:
    """Faster (1fwd + fused AdamW + ddp/bf16) train module with latlon-aware masking."""
    config = build_faster_train_module_config(common)
    config.masking_config = _latlon_masking_config(common.tokenization_config)
    return config

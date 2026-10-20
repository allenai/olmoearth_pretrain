"""Multi-encoder arms on the v1.3 recipe: per-modality towers fused in the AttentionPooler.

The question: does giving each INPUT modality (S2, S1, Landsat) its own shallow block
stack ("tower") before the shared trunk / AttentionPooler read match the v1.3 encoder,
while making the model extensible (a new modality = a new tower against a frozen grid)?

This module is the v1.3 release recipe rebuilt on main's API (the v1.3 scripts on
``gabi/allcap-slot-narrow`` predate the Perceiver -> AttentionPooler rename and do not
import here), as the shared base for the arms in this folder:

* v1.2 base (``scripts/official/v1_2/base.py``: single-bandset S2/Landsat tokens,
  hidden patch-embed projection, mixed 3D RoPE, decode-only maps, band dropout),
* the AttentionPooler bottleneck: ``pooler_dim`` = encoder width, ``[read ->
  self-attend] x POOLER_DEPTH`` with per-depth read projections and attention at
  encoder width; the decoder cross-attends the location-token grid with 2D RoPE,
* supervision heads on the location tokens for the six decode-only maps,
* the detached ``[128, 64]`` Matryoshka student (LayerNorm output) distilled with the
  cosine (2-layer back-projection, hidden 256) + Gram head,
* projection-only target encoder, single masked view, DDP + bf16, fused AdamW, the
  v1.3 decorrelated shape sampler.

Arms differ ONLY in the encoder (plain ``Encoder`` vs ``MultiEncoder``) and, for the
``*_moddrop`` arms, in the masking (whole-modality dropout wrapped around the v1.2
masking). Everything else is shared so the comparison is the architecture.

Per the v1.3 convention the architecture is baked into each arm's module (``MODULE_PATH``),
never passed as a CLI override, so an eval job that re-imports the module rebuilds the
same model. Run an arm as ``python scripts/vnext/20261018_multi_encoder/<arm>.py <subcmd> <run>
<cluster> [overrides]``.
"""

import logging
import sys
from dataclasses import fields
from pathlib import Path

# The v1.2 recipe lives under scripts/official.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "official"))

from olmo_core.config import DType  # noqa: E402
from olmo_core.distributed.parallel import (  # noqa: E402
    DataParallelConfig,
    DataParallelType,
)
from v1_2.base import (  # noqa: E402
    ONLY_DECODE_MODALITIES,
    PATCH_EMBED_HIDDEN_SIZES,
    build_common_components,
    build_dataset_config,
    build_size_model_config,
    build_visualize_config,
)
from v1_2.base import _masking_config as _v1_2_masking_config  # noqa: E402
from v1_2.base import (
    build_dataloader_config as _v1_2_build_dataloader_config,  # noqa: E402
)
from v1_2.base import (
    build_train_module_config as _v1_2_build_train_module_config,  # noqa: E402
)
from v1_2.base import build_trainer_config as _v1_2_build_trainer_config  # noqa: E402

from olmoearth_pretrain.data.constants import Modality  # noqa: E402
from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig  # noqa: E402
from olmoearth_pretrain.internal.experiment import CommonComponents  # noqa: E402
from olmoearth_pretrain.nn.encodings import PositionEncoding  # noqa: E402
from olmoearth_pretrain.nn.flexi_vit import (  # noqa: E402
    AttentionPoolerConfig,
    EncoderConfig,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402
from olmoearth_pretrain.nn.location_token_distillation_head import (  # noqa: E402
    LocationTokenDistillationHeadConfig,
)
from olmoearth_pretrain.nn.multi_encoder import MultiEncoderConfig  # noqa: E402
from olmoearth_pretrain.nn.supervision_head import (  # noqa: E402
    SupervisionHeadConfig,
    SupervisionModalityConfig,
    SupervisionTaskType,
)
from olmoearth_pretrain.nn.tokenization import TokenizationConfig  # noqa: E402
from olmoearth_pretrain.train.masking import MaskingConfig  # noqa: E402
from olmoearth_pretrain.train.train_module.latent_mim import (  # noqa: E402
    LatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

WANDB_PROJECT = "20261018_multi_encoder"

# --- architecture (v1.3) ---------------------------------------------------------------
SIZE_NAME = "base_shallow_decoder"
# [read -> self-attend] layers over the location-token grid; also the number of reads.
POOLER_DEPTH = 4
# The decoder cross-attends the spatial (2D) location grid; the encoder keeps 3D RoPE.
DECODER_POSITION_ENCODING = PositionEncoding.AXIAL_2D_ROPE.value
# Student widths (128 with a self-sufficient 64 Matryoshka prefix) and the hidden width
# of its 2-layer cosine back-projection.
STUDENT_DIMS = [128, 64]
BACK_PROJECTION_HIDDEN = 256

# --- multi-encoder ------------------------------------------------------------------------
# One tower per INPUT modality. The decode-only maps are in no group: they never have
# visible tokens, so a tower for them would be dead weight.
INPUT_MODALITIES = [
    Modality.SENTINEL2_L2A.name,
    Modality.SENTINEL1.name,
    Modality.LANDSAT.name,
]
TOWER_GROUPS = {name: [name] for name in INPUT_MODALITIES}
# Whole-modality dropout for the *_moddrop arms: per sample, each input modality's
# visible tokens all become decoder targets with this probability (never all three).
MODALITY_DROPOUT_RATE = 0.3
# random_all_with_decode arms: random masking inside EVERY input modality at the token
# budget the v1.2 masking actually produces (measured 2026-10-19 on S2/S1/Landsat: 1/3 of
# input tokens visible, 1/6 decoded, 1/2 target-only), so compute and memory match the
# current arms and only the holdout structure changes. Under the v1.2 masking S2 has a
# visible token in only ~59% of samples; under this one, in every sample.
RANDOM_ALL_ENCODE_RATIO = 1.0 / 3.0
RANDOM_ALL_DECODE_RATIO = 1.0 / 6.0

# --- location-token supervision (v1.3) ---------------------------------------------------
SUPERVISION_BASE_WEIGHT = 1.0
TASK_TYPE_WEIGHTS = {
    SupervisionTaskType.CLASSIFICATION: 0.1,
    SupervisionTaskType.BINARY_CLASSIFICATION: 0.1,
    SupervisionTaskType.REGRESSION: 1.0,
}
WORLDCOVER_CLASS_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
# USDA NASS CDL crop codes, normalized as raw/200 (CDL norm config mean=100, std=50,
# std_multiplier=2). Codes 81/88 are pipeline sentinels kept for head-dim compatibility.
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

# --- shape sampler (v1.3) -----------------------------------------------------------------
TOKEN_BUDGET = 3072
MIN_TOKENS_PER_INSTANCE = 228
TEMPORAL_BIAS = 2.75
TIME_PRIORITY_PROB = 0.5
SAMPLED_HW_P_LIST = list(range(1, 17)) + [18, 20, 24, 28, 32]
RANK_MICROBATCH_SIZE = 64

# --- in-loop evals ------------------------------------------------------------------------
# The v1.2 catalog evals, probed on the shipped embedding (the d128 student).
EVAL_STUDENT_DIM = 128


# =========================================================================================
# Model
# =========================================================================================


def build_supervision_head_config(
    base_weight: float = SUPERVISION_BASE_WEIGHT,
) -> SupervisionHeadConfig:
    """Location-token supervision heads over the decode-only map modalities (v1.3)."""

    def _weight(task_type: SupervisionTaskType) -> float:
        return base_weight * TASK_TYPE_WEIGHTS[task_type]

    return SupervisionHeadConfig(
        modality_configs={
            "worldcover": SupervisionModalityConfig(
                task_type=SupervisionTaskType.CLASSIFICATION,
                num_output_channels=len(WORLDCOVER_CLASS_VALUES),
                weight=_weight(SupervisionTaskType.CLASSIFICATION),
                class_values=WORLDCOVER_CLASS_VALUES,
            ),
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
    )


def build_v1_3_model_config(
    common: CommonComponents, size_name: str = SIZE_NAME
) -> LatentMIMConfig:
    """The v1.3 model on main's API: v1.2 base + AttentionPooler + supervision + student."""
    config = build_size_model_config(common, size_name, PATCH_EMBED_HIDDEN_SIZES)
    encoder_config = config.encoder_config
    decoder_config = config.decoder_config
    assert isinstance(encoder_config, EncoderConfig)
    width = encoder_config.embedding_size

    encoder_config.attention_pooler_config = AttentionPoolerConfig(
        pooler_dim=width,
        pooler_depth=POOLER_DEPTH,
        per_depth_read_proj=True,
        attn_dim=width,
        student_dims=list(STUDENT_DIMS),
        student_output_norm=True,
    )
    decoder_config.use_attention_pooler = True
    decoder_config.pooler_dim = width
    decoder_config.position_encoding = DECODER_POSITION_ENCODING

    config.projection_only_target = True
    config.supervision_head_config = build_supervision_head_config()
    config.location_token_distillation_head_config = (
        LocationTokenDistillationHeadConfig(
            back_projection_hidden=BACK_PROJECTION_HIDDEN
        )
    )
    return config


def to_multi_encoder(
    config: LatentMIMConfig,
    *,
    tower_depth: int,
    fusion_depth: int,
    tower_groups: dict[str, list[str]] | None = None,
    fusion_embeddings: bool = True,
) -> LatentMIMConfig:
    """Swap the encoder for a MultiEncoder with the same everything-else.

    ``fusion_depth`` replaces the encoder's ``depth`` (the shared blocks after the
    towers); ``tower_depth`` blocks run per group before them. With ``fusion_depth=0``
    the modalities meet only in the AttentionPooler read.
    """
    encoder_config = config.encoder_config
    assert isinstance(encoder_config, EncoderConfig)
    kwargs = {f.name: getattr(encoder_config, f.name) for f in fields(encoder_config)}
    kwargs["depth"] = fusion_depth
    config.encoder_config = MultiEncoderConfig(
        tower_depth=tower_depth,
        tower_groups=dict(TOWER_GROUPS if tower_groups is None else tower_groups),
        fusion_embeddings=fusion_embeddings,
        **kwargs,
    )
    return config


def build_multi_encoder_model_config(
    common: CommonComponents,
    *,
    tower_depth: int,
    fusion_depth: int,
    size_name: str = SIZE_NAME,
) -> LatentMIMConfig:
    """v1.3 model with per-input-modality towers (``tower_depth``) + ``fusion_depth`` shared blocks."""
    return to_multi_encoder(
        build_v1_3_model_config(common, size_name),
        tower_depth=tower_depth,
        fusion_depth=fusion_depth,
    )


# =========================================================================================
# Masking
# =========================================================================================


def masking_config(
    tokenization_config: TokenizationConfig | None,
    *,
    modality_dropout: bool = False,
    random_all: bool = False,
) -> MaskingConfig:
    """v1.2 masking (``random_time_with_decode``), optionally under whole-modality dropout.

    ``random_all=True`` swaps the base strategy for ``random_all_with_decode`` at the
    matched token budget (``RANDOM_ALL_*_RATIO``): every input modality partly visible
    in every sample, maps decode-only.
    """
    base = _v1_2_masking_config(tokenization_config)
    if random_all:
        base = MaskingConfig(
            strategy_config={
                "type": "random_all_with_decode",
                "encode_ratio": RANDOM_ALL_ENCODE_RATIO,
                "decode_ratio": RANDOM_ALL_DECODE_RATIO,
                "only_decode_modalities": list(ONLY_DECODE_MODALITIES),
            },
            tokenization_config=tokenization_config,
        )
    if not modality_dropout:
        return base
    return MaskingConfig(
        strategy_config={
            "type": "modality_dropout",
            "base_strategy": base.strategy_config,
            "dropout_modalities": list(INPUT_MODALITIES),
            "dropout_rate": MODALITY_DROPOUT_RATE,
            "max_dropped": len(INPUT_MODALITIES) - 1,
        },
        tokenization_config=tokenization_config,
    )


# =========================================================================================
# Data + optimization (v1.3)
# =========================================================================================


def build_dataloader_config(
    common: CommonComponents,
    *,
    modality_dropout: bool = False,
    random_all: bool = False,
) -> OlmoEarthDataLoaderConfig:
    """v1.2 dataloader, single masked view, v1.3 decorrelated shape sampler."""
    config = _v1_2_build_dataloader_config(common)
    config.num_masked_views = 1
    config.token_budget = TOKEN_BUDGET
    config.min_tokens_per_instance = MIN_TOKENS_PER_INSTANCE
    config.temporal_bias = TEMPORAL_BIAS
    config.time_priority_prob = TIME_PRIORITY_PROB
    config.sampled_hw_p_list = list(SAMPLED_HW_P_LIST)
    config.masking_config = masking_config(
        common.tokenization_config,
        modality_dropout=modality_dropout,
        random_all=random_all,
    )
    return config


def build_train_module_config(
    common: CommonComponents,
    *,
    modality_dropout: bool = False,
    random_all: bool = False,
) -> LatentMIMTrainModuleConfig:
    """Single-forward-pass train module: fused AdamW, DDP, bf16 autocast (v1.3).

    The v1.2 builder returns the two-view contrastive config; its fields are copied into
    the plain LatentMIM config and the contrastive loss is dropped, as v1.3 does.
    """
    base = _v1_2_build_train_module_config(common)
    config = LatentMIMTrainModuleConfig(
        optim_config=base.optim_config,
        rank_microbatch_size=RANK_MICROBATCH_SIZE,
        transform_config=base.transform_config,
        masking_config=masking_config(
            common.tokenization_config,
            modality_dropout=modality_dropout,
            random_all=random_all,
        ),
        loss_config=base.loss_config,
        mae_loss_config=base.mae_loss_config,
        token_exit_cfg=base.token_exit_cfg,
        max_grad_norm=base.max_grad_norm,
        scheduler=base.scheduler,
        ema_decay=base.ema_decay,
        regularizer_config=base.regularizer_config,
        compile_model=base.compile_model,
        compile_loss=base.compile_loss,
        find_unused_parameters=base.find_unused_parameters,
        state_dict_save_opts=base.state_dict_save_opts,
        state_dict_load_opts=base.state_dict_load_opts,
        # Replicated params, one coalesced fp32 gradient all-reduce per step. compile stays
        # off (it degraded training quality on the v1.3 stacks).
        dp_config=DataParallelConfig(name=DataParallelType.ddp),
        autocast_precision=DType.bfloat16,
    )
    config.optim_config.fused = True
    return config


# =========================================================================================
# Trainer / evals
# =========================================================================================


def build_trainer_config(common: CommonComponents):
    """v1.2 trainer with the catalog evals probed on the d128 student location tokens."""
    trainer_config = _v1_2_build_trainer_config(common)
    trainer_config.callbacks["wandb"].project = WANDB_PROJECT
    evaluator = trainer_config.callbacks["downstream_evaluator"]
    for task in evaluator.tasks.values():
        task.eval_on_student_location_tokens = True
        task.eval_student_dim = EVAL_STUDENT_DIM
    return trainer_config


__all__ = [
    "INPUT_MODALITIES",
    "ONLY_DECODE_MODALITIES",
    "TOWER_GROUPS",
    "build_common_components",
    "build_dataloader_config",
    "build_dataset_config",
    "build_multi_encoder_model_config",
    "build_train_module_config",
    "build_trainer_config",
    "build_v1_3_model_config",
    "build_visualize_config",
    "masking_config",
    "to_multi_encoder",
]

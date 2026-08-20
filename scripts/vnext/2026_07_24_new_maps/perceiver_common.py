"""Shared pieces for the new-maps v1.2 register-bottleneck (Perceiver) runs.

Ports the ``scripts/official/v1_2`` perceiver recipe onto the new-maps base
(``base.py``): the GLO30 DSM + Meta Canopy Height map set and the new H5 file.
The recipe layered here is exactly:

* ``regbtl`` + ``gdyn`` + ``il`` + ``pdproj`` + ``wideread`` -- the Perceiver-style
  spatial register bottleneck. The encoder keeps v1.2's 3D mixed RoPE; the bottleneck
  reads spatially (2D); the decoder cross-attends the 2D register grid. ``wideread``
  runs the bottleneck attention at encoder width (``register_attn_dim=embedding_size``)
  so ``register_dim`` is pure storage width.
* ``faster`` train module -- single forward pass (plain ``LatentMIMTrainModule``, no
  InfoNCE), fused AdamW, projection-only target encoder, replicated DP + bf16 autocast.
* ``newsampling`` at ``psuniform`` -- the decorrelated (grid, timestep) shape sampler
  with the patch-size distribution held UNIFORM over ps=1..8 (the sweep showed the
  ps=1 oversampling is a confound, so uniform is the clean default).
* ``regsup`` at ``w0p1`` -- register-grid supervision of the decode-only maps.

Intentionally omitted vs the referenced official file: the ``tanchor`` temporal-anchored
read and the time-conditioned ``ndvi`` arm (and the ``latlon`` arm).

This module is self-contained (mirroring how ``base.py`` is a self-contained copy)
so it never imports from ``scripts/official/v1_2``, which is hard-wired to the old
``srtm`` / ``wri_canopy_height_map`` maps and the old H5 file.
"""

import logging

from base import build_dataloader_config as _base_build_dataloader_config
from base import build_model_config as _base_build_model_config
from base import build_train_module_config as _base_build_train_module_config
from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelConfig, DataParallelType

from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig
from olmoearth_pretrain.internal.experiment import CommonComponents
from olmoearth_pretrain.nn.encodings import PositionEncoding
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.nn.supervision_head import (
    SupervisionHeadConfig,
    SupervisionModalityConfig,
    SupervisionTaskType,
)
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

# Clusters the in-loop eval Beaker jobs may run on (mirrors base_faster.py).
LOOP_EVAL_CLUSTERS = ["ai2/jupiter", "ai2/ceres"]


# --------------------------------------------------------------------------- #
# Register bottleneck model builders (port of regbtl_v1_2_common +            #
# regbtl_v1_2_faster_common, on top of the new-maps base model config).       #
# --------------------------------------------------------------------------- #

# Latent-transformer depth over the register grid; with interleave this is also the
# number of cross-attention reads ([read -> self] x4).
REGISTER_LATENT_DEPTH = 4
# The decoder cross-attends the (spatial) register grid, so it runs with 2D RoPE while
# the encoder keeps 3D. "rope" == 2D axial RoPE.
DECODER_POSITION_ENCODING = PositionEncoding.AXIAL_2D_ROPE.value


def build_regbtl_model_config(
    common: CommonComponents,
    *,
    latent_self_attn: bool,
    register_dim: int,
) -> LatentMIMConfig:
    """New-maps base + spatial register bottleneck: ``gdyn`` + ``il`` + ``pdproj``."""
    config = _base_build_model_config(common)
    encoder_config = config.encoder_config
    decoder_config = config.decoder_config

    for sub_config in (encoder_config, decoder_config):
        sub_config.use_register_bottleneck = True
        sub_config.register_dim = register_dim

    # gdyn: dynamic single-latent grid that matches the patch grid at forward time.
    encoder_config.register_grid_size = 0
    # il: interleave reads with the latent transformer ([read -> self] per layer).
    encoder_config.register_interleave = True
    # pdproj: each read block gets its own input norm + K/V projection.
    encoder_config.register_per_depth_read_proj = True
    encoder_config.register_latent_depth = REGISTER_LATENT_DEPTH
    encoder_config.register_latent_self_attn = latent_self_attn

    # The decoder cross-attends the spatial (2D) register grid; the encoder stays 3D.
    decoder_config.position_encoding = DECODER_POSITION_ENCODING

    return config


def build_wideread_regbtl_model_config(
    common: CommonComponents,
    *,
    latent_self_attn: bool,
    register_dim: int,
) -> LatentMIMConfig:
    """Register bottleneck whose read/latent attention runs at ENCODER width.

    ``register_attn_dim = embedding_size`` decouples the bottleneck's attention from
    ``register_dim`` (reads run with the encoder's own 12x64 head shape and consume the
    K/V source at full encoder width), so ``register_dim`` becomes purely the storage
    width. ``register_num_heads`` is left unset so the bottleneck inherits the encoder's
    head count. Projection-only target encoder (valid because the base uses all-zero
    token exits + ema_decay=(1.0, 1.0)).
    """
    config = build_regbtl_model_config(
        common, latent_self_attn=latent_self_attn, register_dim=register_dim
    )
    config.encoder_config.register_attn_dim = config.encoder_config.embedding_size
    config.projection_only_target = True
    return config


# --------------------------------------------------------------------------- #
# Faster train module (port of the 1fwd + fused AdamW + ddp/bf16 recipe).      #
# --------------------------------------------------------------------------- #


def build_faster_train_module_config(
    common: CommonComponents,
) -> LatentMIMTrainModuleConfig:
    """Single-forward-pass, fused-AdamW, replicated-DP + bf16 train module.

    The new-maps base builder returns a ``ContrastiveLatentMIMTrainModuleConfig`` that
    runs two forward passes per batch to feed InfoNCE. This recipe is ``noic``, so the
    second pass is dead work: we copy the base fields into a plain
    :class:`LatentMIMTrainModuleConfig` (one forward pass), drop the contrastive term,
    and switch FSDP -> replicated DDP + bf16 autocast. The base optimizer is already
    ``fused=True``.
    """
    base = _base_build_train_module_config(common)
    return LatentMIMTrainModuleConfig(
        optim_config=base.optim_config,
        rank_microbatch_size=base.rank_microbatch_size,
        transform_config=base.transform_config,
        masking_config=base.masking_config,
        loss_config=base.loss_config,
        mae_loss_config=base.mae_loss_config,
        token_exit_cfg=base.token_exit_cfg,
        max_grad_norm=base.max_grad_norm,
        scheduler=base.scheduler,
        ema_decay=base.ema_decay,
        dp_config=DataParallelConfig(name=DataParallelType.ddp),
        regularizer_config=base.regularizer_config,
        autocast_precision=DType.bfloat16,
        compile_model=base.compile_model,
        compile_loss=base.compile_loss,
        find_unused_parameters=base.find_unused_parameters,
        state_dict_save_opts=base.state_dict_save_opts,
        state_dict_load_opts=base.state_dict_load_opts,
    )


def build_1fwd_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """New-maps base dataloader, but a single masked view for the 1fwd train module.

    The plain :class:`LatentMIMTrainModule` expects ``(patch_size, MaskedOlmoEarthSample)``;
    the base config's ``num_masked_views=2`` would yield the contrastive two-view tuple.
    """
    config = _base_build_dataloader_config(common)
    config.num_masked_views = 1
    return config


# --------------------------------------------------------------------------- #
# Newsampling shape-sampler knobs (port of regbtl_v1_2_newsampling_common).    #
# --------------------------------------------------------------------------- #

# w0p1: 10x the original 0.01 supervision base weight.
SUPERVISION_BASE_WEIGHT = 0.1

# A multiple of 256; with the decode-only maps excluded from the budget this fits the
# full 12 months up to a 9x9 register grid.
TOKEN_BUDGET = 3072
# Token floor; with maps excluded this is 3*hw^2*t, so 228 drops hw<=2 and forces small
# grids onto long sequences.
MIN_TOKENS_PER_INSTANCE = 228
# Skews the timestep draw toward the maximum of its feasible window (weight t**bias).
TEMPORAL_BIAS = 2.75
# Half of batches sample timesteps first (then a grid that fits); half sample grid first.
TIME_PRIORITY_PROB = 0.5
# Base grids 1..16 plus a coarse incremental tail; the token floor drops hw<=2.
SAMPLED_HW_P_LIST = list(range(1, 17)) + [18, 20, 24, 28, 32]
# Uniform over ps=1..8 -- the psuniform arm (isolates the ps=1 oversampling confound out
# of the newsampling recipe).
UNIFORM_PATCH_SIZE_PROBS = [1.0 / 8] * 8
# Base 64 (1 microbatch/step); drop to 32 if the larger budget OOMs (throughput
# unchanged, micro only affects memory). Launch with expandable_segments to be safe.
RANK_MICROBATCH_SIZE = 64


def apply_new_sampling(config: OlmoEarthDataLoaderConfig) -> OlmoEarthDataLoaderConfig:
    """Set the decorrelated shape-sampling knobs on a dataloader config in place.

    Leaves ``patch_size_probs`` untouched (dataloader default = uniform); pair with
    :func:`apply_uniform_patch_sizes` to pin the uniform distribution explicitly.
    """
    config.token_budget = TOKEN_BUDGET
    config.exclude_only_decode_from_budget = True
    config.min_tokens_per_instance = MIN_TOKENS_PER_INSTANCE
    config.temporal_bias = TEMPORAL_BIAS
    config.time_priority_prob = TIME_PRIORITY_PROB
    config.sampled_hw_p_list = SAMPLED_HW_P_LIST
    return config


def apply_uniform_patch_sizes(
    config: OlmoEarthDataLoaderConfig,
) -> OlmoEarthDataLoaderConfig:
    """Pin patch-size sampling to uniform over ps=1..8 (the psuniform arm)."""
    config.patch_size_probs = list(UNIFORM_PATCH_SIZE_PROBS)
    return config


def apply_microbatch(config: LatentMIMTrainModuleConfig) -> LatentMIMTrainModuleConfig:
    """Set the rank microbatch size in place so the larger budget fits memory."""
    config.rank_microbatch_size = RANK_MICROBATCH_SIZE
    return config


# --------------------------------------------------------------------------- #
# Register-grid supervision (port of regbtl_v1_2_regsup_common) for NEW maps.  #
# --------------------------------------------------------------------------- #

# Low base weight nudge; classification/BCE scaled 10x down to balance against L1/MSE.
TASK_TYPE_WEIGHTS = {
    SupervisionTaskType.CLASSIFICATION: 0.1,
    SupervisionTaskType.BINARY_CLASSIFICATION: 0.1,
    SupervisionTaskType.REGRESSION: 1.0,
}

WORLDCOVER_CLASS_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]

# USDA NASS CDL crop codes normalized as raw/200 (see regbtl_v1_2_regsup_common).
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

# GLO30 is a 3-band DSM (elevation, slope, aspect). Supervise ELEVATION only (band 0),
# the direct analog of the old single-band SRTM elevation target: aspect is circular
# (0 deg ~= 360 deg, -1 flat sentinel) and plain L1/MSE is lossy on a wrap-around angle,
# so it is left unsupervised rather than regressed naively.
GLO30_ELEVATION_BAND_INDEX = 0


def build_supervision_head_config(
    *, base_weight: float = SUPERVISION_BASE_WEIGHT
) -> SupervisionHeadConfig:
    """Register-grid supervision over the new-maps decode-only modalities.

    Same six-modality set as the official regsup head, with the two swapped maps:
    ``srtm`` -> ``glo30`` (elevation band only) and ``wri_canopy_height_map`` ->
    ``meta_canopy_height``. No latlon and no NDVI arm.
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
        "glo30": SupervisionModalityConfig(
            task_type=SupervisionTaskType.REGRESSION,
            num_output_channels=1,
            weight=_weight(SupervisionTaskType.REGRESSION),
            regression_loss_type="l1",
            target_band_index=GLO30_ELEVATION_BAND_INDEX,
        ),
        "openstreetmap_raster": SupervisionModalityConfig(
            task_type=SupervisionTaskType.BINARY_CLASSIFICATION,
            num_output_channels=30,
            weight=_weight(SupervisionTaskType.BINARY_CLASSIFICATION),
            pos_weight=True,
        ),
        "meta_canopy_height": SupervisionModalityConfig(
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
    return SupervisionHeadConfig(
        modality_configs=modality_configs,
        register_supervision=True,
    )


def add_register_supervision(
    config: LatentMIMConfig, *, base_weight: float = SUPERVISION_BASE_WEIGHT
) -> LatentMIMConfig:
    """Attach the new-maps register-grid supervision head to a regbtl model config."""
    config.supervision_head_config = build_supervision_head_config(
        base_weight=base_weight
    )
    return config


# --------------------------------------------------------------------------- #
# In-loop eval routing.                                                        #
# --------------------------------------------------------------------------- #


def route_loop_evals_to_beaker(trainer_config, module_path: str):
    """Route the base in-loop evals through non-blocking Beaker jobs.

    Keeps the new-maps base eval catalog (m-eurosat, so2sat, mados, pastis, yemen_crop,
    fifty_cities S2 + S1S2) but makes each due evaluator launch a Beaker job that
    evaluates the just-saved checkpoint. ``beaker_eval_module_path`` points at the
    launching script so the eval job rebuilds the matching architecture from its
    ``build_model_config``.
    """
    evaluator = trainer_config.callbacks["downstream_evaluator"]
    evaluator.run_as_beaker_job = True
    evaluator.beaker_eval_module_path = module_path
    evaluator.beaker_eval_clusters = list(LOOP_EVAL_CLUSTERS)
    return trainer_config

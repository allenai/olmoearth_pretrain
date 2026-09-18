"""OlmoEarth v1.3: the register-bottleneck teacher with a distilled 128-d student.

v1.3 is the v1.2 recipe (``scripts/official/v1_2/base.py``: hidden patch-embed
projection, mixed 3D RoPE, decode-only map modalities) with two additions and a
faster training stack. Everything below is what the release run
``regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsamp_psuniform_stunorm_mlpgram1``
(W&B project ``2026_08_26_student_norm``) trained with; the v1.2 config is imported
rather than copied so the two versions cannot drift apart silently.

**Aggregation -- the spatial register bottleneck** (``regbtl``). A Perceiver-style
read of the patch tokens into a register grid that is a purely spatial summary:

* the grid has no temporal axis, so the RoPE is asymmetric: the encoder keeps v1.2's
  3D mixed RoPE over ``(t, row, col)``, the bottleneck reads with 2D ``(row, col)``
  RoPE, and the decoder cross-attends the register grid with 2D RoPE (mask tokens
  recover their timestep from the additive slot-index + month encodings);
* a single learned latent cloned to match the patch grid at forward time, so
  windowing stays arbitrary;
* interleaved reads with per-depth read projections, ``[read -> self-attn] x 4``;
* ``register_dim=768`` with attention at ENCODER width (``register_attn_dim``), so
  the register width is purely the storage width;
* register supervision: per-modality heads on the register grid predict the
  decode-only map modalities (worldcover, srtm, openstreetmap, canopy height, cdl,
  worldcereal) at ``base_weight=1.0``, scaled 0.1x for the classification/BCE heads.

**Compaction -- the distilled student** (``proj128lin``). A detached per-cell
``Linear(768, 128)`` on the registers, ending in a LayerNorm (``stunorm``), trained
by cosine (through a 2-layer MLP back-projection, hidden 256) and a Gram term at
each Matryoshka prefix of ``[128, 64]``. Stop-gradient: the teacher trains exactly
as it would without the student. The student IS the shipped embedding.

**Training stack.** Single forward pass per batch with the plain LatentMIM train
module (no instance-contrastive loss), fused AdamW, projection-only target encoder
(valid because the v1.2 base uses all-zero token exits and ``ema_decay=(1.0, 1.0)``),
replicated DP + bf16 autocast, and the decorrelated shape sampler: timesteps drawn
independently of the grid, biased toward the full year, a token floor, uniform
patch sizes 1..8, and the decode-only maps excluded from the encoder token budget.

**In-loop evals** run as separate Beaker jobs (``run_as_beaker_job=True``) that
rebuild this module from ``MODULE_PATH``, so the architecture is baked in here and
must not be passed as CLI overrides. The eval set is the AEF balanced-trial datasets
(kNN twins, which carry the AEF protocol) plus year-aligned PASTIS, on S1+S2+Landsat,
scored on the student at both widths.

The report ablations live in ``ablations/`` and import from here.
"""

import logging
import sys
from dataclasses import replace
from pathlib import Path

# v1.3 builds on the v1.2 config, which lives one directory over.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from olmo_core.config import DType  # noqa: E402
from olmo_core.distributed.parallel import (  # noqa: E402
    DataParallelConfig,
    DataParallelType,
)
from olmo_core.train.common import Duration  # noqa: E402
from v1_2.base import (  # noqa: E402
    PATCH_EMBED_HIDDEN_SIZES,
    build_common_components,
    build_dataset_config,
    build_visualize_config,
)
from v1_2.base import (
    build_dataloader_config as _v1_2_build_dataloader_config,  # noqa: E402
)
from v1_2.base import (
    build_size_model_config as _v1_2_build_size_model_config,  # noqa: E402
)
from v1_2.base import (
    build_train_module_config as _v1_2_build_train_module_config,  # noqa: E402
)
from v1_2.base import build_trainer_config as _v1_2_build_trainer_config  # noqa: E402

from olmoearth_pretrain.data.constants import Modality  # noqa: E402
from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig  # noqa: E402
from olmoearth_pretrain.internal.all_evals import (  # noqa: E402
    AEF_SUPPLEMENTAL_YEAR_ALIGNED,
    EMBEDDING_EVAL_TASKS,
)
from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.encodings import PositionEncoding  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402
from olmoearth_pretrain.nn.supervision_head import (  # noqa: E402
    SupervisionHeadConfig,
    SupervisionModalityConfig,
    SupervisionTaskType,
)
from olmoearth_pretrain.train.train_module.latent_mim import (  # noqa: E402
    LatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/base.py"

# --- architecture --------------------------------------------------------------------
# Encoder/decoder size preset: the v1.2 Base encoder with the shallow decoder.
ENCODER_SIZE_NAME = "base_shallow_decoder"
# Teacher (primary bottleneck) width.
REGISTER_DIM = 768
# Latent-transformer depth over the register grid; with interleaving this is also the
# number of cross-attention reads.
REGISTER_LATENT_DEPTH = 4
# The decoder cross-attends the spatial register grid, so it runs 2D axial RoPE while
# the encoder keeps 3D.
DECODER_POSITION_ENCODING = PositionEncoding.AXIAL_2D_ROPE.value
# Student widths: 128 with its first 64 dims trained as a self-sufficient Matryoshka
# prefix (own back-projection and Gram term), so one artifact serves both by truncation.
PROJECTION_DIMS = [128, 64]
# Hidden width of the 2-layer back-projection heads (training-only, discarded at
# inference). SimReg's (m, 2m, d) rule at m = 128, held fixed across prefixes.
BACK_PROJECTION_HIDDEN = 256

# --- register supervision ---------------------------------------------------------------
SUPERVISION_BASE_WEIGHT = 1.0
# Classification/BCE losses run ~10x larger than the L1 regressions, so they are scaled
# down to contribute comparably.
TASK_TYPE_WEIGHTS = {
    SupervisionTaskType.CLASSIFICATION: 0.1,
    SupervisionTaskType.BINARY_CLASSIFICATION: 0.1,
    SupervisionTaskType.REGRESSION: 1.0,
}
WORLDCOVER_CLASS_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
# USDA NASS CDL crop codes (public legend). Raw values are uint8 codes in [0, 255],
# normalized as raw/200 by the CDL norm config (mean=100, std=50, std_multiplier=2 ->
# min=0, max=200); if that config changes, this divisor must change with it. Codes
# 81/88 are pipeline sentinels kept for head-dim compatibility with earlier checkpoints.
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

# --- shape sampler ------------------------------------------------------------------------
# Encoder token budget per instance. With the maps excluded this fits the full 12 months
# at grids up to hw=9 (3*9^2*12 = 2916).
TOKEN_BUDGET = 3072
# Minimum tokens a shape must cost (3*hw^2*t with maps excluded): drops hw<=2 and forces
# small grids onto long sequences.
MIN_TOKENS_PER_INSTANCE = 228
# Skews the timestep draw toward the maximum of its feasible window (weight t**bias).
TEMPORAL_BIAS = 2.75
# Half of batches sample timesteps first (then a grid that fits); half sample grid first.
TIME_PRIORITY_PROB = 0.5
# Base grids 1..16 plus a coarse tail; the token floor drops hw<=2 and large grids
# naturally carry few timesteps.
SAMPLED_HW_P_LIST = list(range(1, 17)) + [18, 20, 24, 28, 32]
RANK_MICROBATCH_SIZE = 64

# --- in-loop evals -------------------------------------------------------------------------
LOOP_EVAL_CLUSTERS = ["ai2/jupiter", "ai2/ceres"]
# Must be a multiple of the checkpointer save_interval (5000). 80k: the 18-task student
# job needs the time, and consecutive eval jobs sharing one resumed W&B run silently
# drop the overlapping writer's rows.
STUDENT_LOOP_EVAL_INTERVAL_STEPS = 80000
# The eight year-aligned AEF datasets' kNN twins (which carry AEF's balanced-trial
# protocol) plus year-aligned PASTIS, all on unmasked S1+S2+Landsat. Names are looked up
# in the canonical registry so a typo raises at import.
AEFTRIAL_LOOP_EVAL_NAMES = tuple(
    f"{dataset}_ws16_ps1_sentinel1_sentinel2_landsat_knn"
    for dataset in AEF_SUPPLEMENTAL_YEAR_ALIGNED
) + ("pastis_year_aligned_ws16_ps1_sentinel1_sentinel2_landsat",)


# =========================================================================================
# Model
# =========================================================================================


def build_register_bottleneck_model_config(
    common: CommonComponents,
    *,
    register_dim: int,
    size_name: str = ENCODER_SIZE_NAME,
) -> LatentMIMConfig:
    """v1.2 base + the spatial register bottleneck, without supervision or a student.

    ``register_dim`` is the register (storage) width; attention runs at encoder width
    regardless. The projection-only target encoder is set here because it is part of
    the training stack every v1.3 run uses.
    """
    config = _v1_2_build_size_model_config(common, size_name, PATCH_EMBED_HIDDEN_SIZES)
    encoder_config = config.encoder_config
    decoder_config = config.decoder_config

    for sub_config in (encoder_config, decoder_config):
        sub_config.use_register_bottleneck = True
        sub_config.register_dim = register_dim

    # Interleave reads with the latent transformer ([read -> self] per layer), each read
    # block with its own input norm + K/V projection.
    encoder_config.register_per_depth_read_proj = True
    encoder_config.register_latent_depth = REGISTER_LATENT_DEPTH
    # Bottleneck attention at encoder width: register_dim is purely the storage width.
    encoder_config.register_attn_dim = encoder_config.embedding_size

    # The decoder cross-attends the spatial (2D) register grid; the encoder stays 3D.
    decoder_config.position_encoding = DECODER_POSITION_ENCODING

    config.projection_only_target = True
    return config


def build_supervision_head_config(
    base_weight: float = SUPERVISION_BASE_WEIGHT,
) -> SupervisionHeadConfig:
    """Register-grid supervision heads over the decode-only map modalities."""

    def _weight(task_type: SupervisionTaskType) -> float:
        return base_weight * TASK_TYPE_WEIGHTS[task_type]

    modality_configs = {
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
    return SupervisionHeadConfig(
        modality_configs=modality_configs,
    )


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """The v1.3 model: d768 supervised teacher + detached linear [128, 64] student."""
    config = build_register_bottleneck_model_config(common, register_dim=REGISTER_DIM)
    config.supervision_head_config = build_supervision_head_config()
    # Heads attach to the teacher registers only; the student trains by distillation.

    encoder_config = config.encoder_config
    encoder_config.register_projection_dims = list(PROJECTION_DIMS)
    # LayerNorm on the student output, at the full student width (LN(z)[:64] is what a
    # truncating consumer reads, so that is what is trained).
    encoder_config.register_projection_output_norm = True
    encoder_config.register_back_projection_hidden = BACK_PROJECTION_HIDDEN
    return config


# =========================================================================================
# Data + optimization
# =========================================================================================


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """v1.2 dataloader, single masked view, decorrelated shape sampler."""
    config = _v1_2_build_dataloader_config(common)
    # One masked view: the plain LatentMIM train module runs one forward pass per batch.
    config.num_masked_views = 1
    config.token_budget = TOKEN_BUDGET
    config.min_tokens_per_instance = MIN_TOKENS_PER_INSTANCE
    config.temporal_bias = TEMPORAL_BIAS
    config.time_priority_prob = TIME_PRIORITY_PROB
    config.sampled_hw_p_list = list(SAMPLED_HW_P_LIST)
    return config


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """Single-forward-pass train module with fused AdamW, replicated DP and bf16.

    The v1.2 builder returns a contrastive config that runs two forward passes per
    batch. Its fields are copied into the plain :class:`LatentMIMTrainModuleConfig`
    (one pass) rather than re-declared, so this stays in lockstep with v1.2; only the
    contrastive-specific fields are dropped.
    """
    base = _v1_2_build_train_module_config(common)
    config = LatentMIMTrainModuleConfig(
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
        dp_config=base.dp_config,
        regularizer_config=base.regularizer_config,
        autocast_precision=base.autocast_precision,
        compile_model=base.compile_model,
        compile_loss=base.compile_loss,
        find_unused_parameters=base.find_unused_parameters,
        state_dict_save_opts=base.state_dict_save_opts,
        state_dict_load_opts=base.state_dict_load_opts,
    )
    config.optim_config.fused = True
    # Replicated params with one coalesced fp32 gradient all-reduce per step. torch.compile
    # is deliberately off: it degraded training quality on both the FSDP and DDP stacks.
    config.dp_config = DataParallelConfig(name=DataParallelType.ddp)
    config.autocast_precision = DType.bfloat16
    config.rank_microbatch_size = RANK_MICROBATCH_SIZE
    return config


# =========================================================================================
# In-loop evals
# =========================================================================================


def aeftrial_loop_eval_tasks(interval_steps: int) -> dict:
    """The AEF-trial + PASTIS tasks at ``interval_steps``, PASTIS first.

    Eval jobs at urgent priority get preempted mid-run and the trailing tasks are the
    ones that lose their metrics, so the segmentation readout runs ahead of the
    classification block.
    """
    tasks = {
        name: replace(
            EMBEDDING_EVAL_TASKS[name], eval_interval=Duration.steps(interval_steps)
        )
        for name in AEFTRIAL_LOOP_EVAL_NAMES
    }
    ordered = sorted(tasks, key=lambda n: (not n.startswith("pastis"), n))
    return {name: tasks[name] for name in ordered}


def route_loop_evals_through_beaker(trainer_config, module_path: str, tasks: dict):
    """REPLACE the trainer's eval set with ``tasks`` and run each as a Beaker job.

    This discards the shared eval catalog on purpose: v1.3 runs are judged on the
    embedding product, and the catalog evals would inflate the eval job's runtime.
    ``module_path`` is what the eval job re-imports to rebuild the model.
    """
    evaluator = trainer_config.callbacks["downstream_evaluator"]
    evaluator.tasks = tasks
    evaluator.run_as_beaker_job = True
    evaluator.beaker_eval_module_path = module_path
    evaluator.beaker_eval_clusters = list(LOOP_EVAL_CLUSTERS)
    return trainer_config


def set_student_loop_evals(
    trainer_config,
    module_path: str,
    *,
    interval_steps: int = STUDENT_LOOP_EVAL_INTERVAL_STEPS,
    projection_dims: tuple[int, ...] = (128, 64),
):
    """AEF trials + PASTIS on the student at every width in ``projection_dims``.

    Every task probes the detached student (``eval_on_projected_registers``); the d768
    teacher is not scored, since these runs are judged on the shipped embedding. Order
    is the priority order: the shipped width first, PASTIS first within each width.
    """
    base_tasks = aeftrial_loop_eval_tasks(interval_steps)
    tasks = {
        f"{name}_proj{dim}": replace(
            task, eval_on_projected_registers=True, eval_projection_dim=dim
        )
        for dim in projection_dims
        for name, task in base_tasks.items()
    }
    return route_loop_evals_through_beaker(trainer_config, module_path, tasks)


def build_trainer_config(common: CommonComponents):
    """v1.2 trainer + the student in-loop evals routed through Beaker."""
    return set_student_loop_evals(_v1_2_build_trainer_config(common), MODULE_PATH)


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

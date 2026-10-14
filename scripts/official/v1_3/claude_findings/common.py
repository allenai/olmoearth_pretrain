"""Shared builders for the v1.3 control recipes on the merged (main-API) code.

The v1.3 per-pixel location-token runs ``v1_3_rc_ld1_pixtgt_pix512`` and
``v1_3_rc_ld1_pix512`` were trained at 9f5a25b02 (``gabi/rc-pix512-sweep`` lineage)
from ``scripts/official/v1_3/base.py`` + ``ablations/rc_pix512.py`` +
``ablations/rc_ld1_pix512.py`` (+ ``rc_pixtgt_pix512.py``). Those scripts use the
pre-rename Perceiver API and no longer import on this branch, so this module rebuilds
the same configs in main's API:

* ``PerceiverConfig`` -> ``AttentionPoolerConfig`` (``register_dim`` -> ``pooler_dim``,
  ``latent_depth`` -> ``pooler_depth``); the decoder's ``use_perceiver`` /
  ``register_dim`` -> ``use_attention_pooler`` / ``pooler_dim``;
  ``RegisterDistillationHeadConfig`` -> ``LocationTokenDistillationHeadConfig``.
* Per-pixel latents with a random stride under a 512-latent budget
  (``pixel_latents`` / ``random_latent_stride`` / ``max_latents=512``) -> the
  dataloader draws ``location_query_patch_size`` per rank batch under
  ``max_location_queries=512`` (uniform over the divisors of the patch size that fit,
  the patch size always allowed), and the encoder takes it as a forward argument.
* ``eval_latent_stride=1`` (one latent per pixel at eval) -> every in-loop eval task
  at patch size > 1 sets ``location_query_patch_size=1``.
* Eval tasks' ``eval_on_student_registers`` -> ``eval_on_student_location_tokens``.

``scripts/official/v1_3/claude_findings/diff_against_trained.py`` checks the result
field by field against each run's step667200 ``config.json``.
"""

import sys
from dataclasses import replace
from pathlib import Path

# v1.3 builds on the v1.2 config, two directories up.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
from olmoearth_pretrain.internal.experiment import CommonComponents  # noqa: E402
from olmoearth_pretrain.nn.encodings import PositionEncoding  # noqa: E402
from olmoearth_pretrain.nn.flexi_vit import AttentionPoolerConfig  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402
from olmoearth_pretrain.nn.location_token_distillation_head import (  # noqa: E402
    LocationTokenDistillationHeadConfig,
)
from olmoearth_pretrain.nn.supervision_head import (  # noqa: E402
    SupervisionHeadConfig,
    SupervisionModalityConfig,
    SupervisionTaskType,
)
from olmoearth_pretrain.train.train_module.latent_mim import (  # noqa: E402
    LatentMIMTrainModuleConfig,
)

__all__ = [
    "build_common_components",
    "build_dataset_config",
    "build_visualize_config",
    "build_model_config",
    "build_dataloader_config",
    "build_train_module_config",
    "build_trainer_config",
]

# --- architecture (v1_3/base.py + rc_pix512 + rc_ld1_pix512) ------------------------------
ENCODER_SIZE_NAME = "base_shallow_decoder"
POOLER_DIM = 768
# rc_ld1_*: one [read -> self-attend] layer (the RC base had 4).
POOLER_DEPTH = 1
DECODER_POSITION_ENCODING = PositionEncoding.AXIAL_2D_ROPE.value
PROJECTION_DIMS = [128, 64]
BACK_PROJECTION_HIDDEN = 256
# rc_pix512: per-sample location-query budget for the random per-batch query patch size.
MAX_LOCATION_QUERIES = 512

# --- supervision (v1_3/base.py) ------------------------------------------------------------
SUPERVISION_BASE_WEIGHT = 1.0
TASK_TYPE_WEIGHTS = {
    SupervisionTaskType.CLASSIFICATION: 0.1,
    SupervisionTaskType.BINARY_CLASSIFICATION: 0.1,
    SupervisionTaskType.REGRESSION: 1.0,
}
WORLDCOVER_CLASS_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
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

# --- shape sampler (v1_3/base.py) ----------------------------------------------------------
TOKEN_BUDGET = 3072
MIN_TOKENS_PER_INSTANCE = 228
TEMPORAL_BIAS = 2.75
TIME_PRIORITY_PROB = 0.5
SAMPLED_HW_P_LIST = list(range(1, 17)) + [18, 20, 24, 28, 32]
RANK_MICROBATCH_SIZE = 64

# --- in-loop evals (v1_3/base.py + pure_perceiver_mix.build_mix_trainer_config) ----------
LOOP_EVAL_CLUSTERS = ["ai2/jupiter", "ai2/ceres"]
STUDENT_LOOP_EVAL_INTERVAL_STEPS = 80000
AEFTRIAL_LOOP_EVAL_NAMES = tuple(
    f"{dataset}_ws16_ps1_sentinel1_sentinel2_landsat_knn"
    for dataset in AEF_SUPPLEMENTAL_YEAR_ALIGNED
) + ("pastis_year_aligned_ws16_ps1_sentinel1_sentinel2_landsat",)
REGISTER_EVAL_TASKS = ("m-eurosat", "pastis")
PS2_STUDENT_EVAL_DATASETS = (
    "pastis_year_aligned",
    "africa_crop_mask_year_aligned",
    "ethiopia_crops_year_aligned",
)
WANDB_PROJECT = "20260921_perceiver_shapes"


# =========================================================================================
# Model
# =========================================================================================


def build_supervision_head_config(
    base_weight: float = SUPERVISION_BASE_WEIGHT,
) -> SupervisionHeadConfig:
    """Location-token supervision heads over the decode-only map modalities."""

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
    return SupervisionHeadConfig(modality_configs=modality_configs)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """rc_ld1_pix512's model: v1.2 base + 1-layer AttentionPooler + [128, 64] student."""
    config = _v1_2_build_size_model_config(
        common, ENCODER_SIZE_NAME, PATCH_EMBED_HIDDEN_SIZES
    )
    encoder_config = config.encoder_config
    decoder_config = config.decoder_config
    encoder_config.attention_pooler_config = AttentionPoolerConfig(
        pooler_dim=POOLER_DIM,
        pooler_depth=POOLER_DEPTH,
        per_depth_read_proj=True,
        attn_dim=encoder_config.embedding_size,
        student_dims=list(PROJECTION_DIMS),
        student_output_norm=True,
    )
    decoder_config.use_attention_pooler = True
    decoder_config.pooler_dim = POOLER_DIM
    decoder_config.position_encoding = DECODER_POSITION_ENCODING
    config.projection_only_target = True
    config.supervision_head_config = build_supervision_head_config()
    config.location_token_distillation_head_config = (
        LocationTokenDistillationHeadConfig(
            back_projection_hidden=BACK_PROJECTION_HIDDEN
        )
    )
    return config


# =========================================================================================
# Data + optimization
# =========================================================================================


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """v1.2 dataloader, single masked view, decorrelated shape sampler, 512-query budget."""
    config = _v1_2_build_dataloader_config(common)
    config.num_masked_views = 1
    config.token_budget = TOKEN_BUDGET
    config.min_tokens_per_instance = MIN_TOKENS_PER_INSTANCE
    config.temporal_bias = TEMPORAL_BIAS
    config.time_priority_prob = TIME_PRIORITY_PROB
    config.sampled_hw_p_list = list(SAMPLED_HW_P_LIST)
    config.max_location_queries = MAX_LOCATION_QUERIES
    return config


def build_train_module_config(
    common: CommonComponents, *, pixel_targets: bool = False
) -> LatentMIMTrainModuleConfig:
    """v1_3/base.py's single-forward-pass train module (+ optional pixel MIM targets)."""
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
    config.dp_config = DataParallelConfig(name=DataParallelType.ddp)
    config.autocast_precision = DType.bfloat16
    config.rank_microbatch_size = RANK_MICROBATCH_SIZE
    config.pixel_targets = pixel_targets
    return config


# =========================================================================================
# In-loop evals
# =========================================================================================


def _per_pixel(task):
    """One location query per pixel at the task's patch size (rc_pix512's eval stride 1)."""
    if task.patch_size > 1:
        return replace(task, location_query_patch_size=1)
    return task


def aeftrial_loop_eval_tasks(interval_steps: int) -> dict:
    """The AEF-trial + PASTIS tasks at ``interval_steps``, PASTIS first."""
    tasks = {
        name: replace(
            EMBEDDING_EVAL_TASKS[name], eval_interval=Duration.steps(interval_steps)
        )
        for name in AEFTRIAL_LOOP_EVAL_NAMES
    }
    ordered = sorted(tasks, key=lambda n: (not n.startswith("pastis"), n))
    return {name: tasks[name] for name in ordered}


def build_trainer_config(
    common: CommonComponents,
    module_path: str,
    *,
    ps4_student_evals: bool = True,
    ps2_student_evals: bool = True,
):
    """rc_ld1_pix512's in-loop evals, run as Beaker jobs that re-import ``module_path``.

    Student AEF + PASTIS at patch size 1 (d128 / d64), m-eurosat + pastis on the d768
    location tokens, the d128 student at patch size 4 per pixel on AEF + PASTIS, and at
    patch size 2 per pixel on PASTIS + two AEF tasks.
    """
    trainer_config = _v1_2_build_trainer_config(common)
    evaluator = trainer_config.callbacks["downstream_evaluator"]
    catalog = evaluator.tasks
    register_tasks = {
        name: _per_pixel(
            replace(
                catalog[name],
                eval_interval=Duration.steps(STUDENT_LOOP_EVAL_INTERVAL_STEPS),
            )
        )
        for name in REGISTER_EVAL_TASKS
    }
    base_tasks = aeftrial_loop_eval_tasks(STUDENT_LOOP_EVAL_INTERVAL_STEPS)
    tasks = {
        f"{name}_proj{dim}": replace(
            task, eval_on_student_location_tokens=True, eval_student_dim=dim
        )
        for dim in PROJECTION_DIMS
        for name, task in base_tasks.items()
    }
    tasks.update(register_tasks)
    if ps4_student_evals:
        for name, task in base_tasks.items():
            assert "_ps1_" in name, name
            tasks[name.replace("_ps1_", "_ps4_") + "_proj128"] = replace(
                task,
                patch_size=4,
                location_query_patch_size=1,
                eval_on_student_location_tokens=True,
                eval_student_dim=128,
            )
    if ps2_student_evals:
        for name, task in base_tasks.items():
            if not name.startswith(PS2_STUDENT_EVAL_DATASETS):
                continue
            tasks[name.replace("_ps1_", "_ps2_") + "_proj128"] = replace(
                task,
                patch_size=2,
                location_query_patch_size=1,
                eval_on_student_location_tokens=True,
                eval_student_dim=128,
            )
    evaluator.tasks = tasks
    evaluator.run_as_beaker_job = True
    evaluator.beaker_eval_module_path = module_path
    evaluator.beaker_eval_clusters = list(LOOP_EVAL_CLUSTERS)
    trainer_config.callbacks["wandb"].project = WANDB_PROJECT
    # Keep every permanent checkpoint, as the original runs did (138 kept): olmo-core
    # 2.6 defaults to the last 3, which deletes a checkpoint before its in-loop eval
    # job may have read it.
    trainer_config.callbacks["checkpointer"].max_checkpoints = None
    return trainer_config

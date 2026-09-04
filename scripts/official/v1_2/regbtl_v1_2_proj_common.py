"""Shared pieces for the detached 768->128 register-projection ("student") runs.

Motivation (2026-07-30 embedding-evals synthesis): the d768 register runs beat every
d128 arm by ~6 mIoU on the frozen ps=1 PASTIS probes (0.6447 vs 0.5800 at ws16
S1+S2; Tessera v2 large = 0.5938) while d128 matches d768 on the center-pixel
classification tasks -- i.e. training the narrow width NATIVELY under the pretext
loss is the bottleneck, not the 128-dim budget itself. Tessera v2's playbook (and
SEED/CompRess) says the fix is distillation from a wide teacher. These runs amortize
that distillation into pretraining: the d768 bottleneck trains exactly as before
(the teacher), and a DETACHED low-dim student is trained alongside it against the
improving teacher. Stop-gradient means the encoder run is unchanged -- variant
``sup768`` is encoder-identical to ``regbtl_v1_2_gdyn_d768_regsup_w0p1_newsampling_
psuniform`` -- and every checkpoint ships a 768d head plus a 128d student whose
first 64 dims are a self-sufficient Matryoshka prefix, all evaluated side by side
in-loop.

Two student architectures (``register_projection_type``):

* ``lin`` -- per-cell ``Linear(768, 128)`` on the detached registers: is the
  teacher's information linearly readable per cell at the low width?
* ``pcv`` -- a second wideread Perceiver bottleneck at d128 re-reading the detached
  final-layer tokens (the deployed d128 architecture, trained by distillation
  instead of the pretext loss): was the d128 architecture ever the problem, or only
  its native training signal?

Three supervision placements (``supervision_source``), crossing the student axis
for 6 runs total:

* ``sup768`` (``"registers"``) -- heads on the d768 registers only (the current
  regsup recipe); the student trains purely by distillation.
* ``supboth`` (``"both"``)     -- separate heads on both widths.
* ``sup128`` (``"projection"``) -- heads on the detached student only: the encoder
  gets NO supervision gradient, isolating whether regsup helps by shaping the
  encoder or by shaping the output space.

The student is always trained with the distillation terms (cosine via a learned
back-projection + Gram/relational matching, ``LatentMIMTrainModule`` defaults).

Recipe otherwise matches the d768 w0p1 newsampling psuniform run: wideread regbtl at
register_dim 768, regsup base_weight 0.1, decorrelated sampler at uniform patch
sizes. The architecture is baked into each run script (the in-loop Beaker eval jobs
rebuild the model from the launching module's ``build_model_config``).
"""

import logging
from dataclasses import dataclass, replace

from olmo_core.optim import OptimGroupOverride
from olmo_core.optim.scheduler import ConstantWithWarmup, Scheduler
from olmo_core.train.common import Duration
from regbtl_v1_2_common import (
    ENCODER_SIZE_NAME,
    FIFTY_CITIES_LOOP_EVAL_TASKS,
    LOOP_EVAL_CLUSTERS,
    PASTIS_EMBEDDING_LOOP_EVAL_TASKS,
)
from regbtl_v1_2_faster_common import build_wideread_regbtl_model_config
from regbtl_v1_2_regsup_common import add_register_supervision

from olmoearth_pretrain.internal.all_evals import (
    AEF_SUPPLEMENTAL_YEAR_ALIGNED,
)
from olmoearth_pretrain.internal.all_evals import (
    EMBEDDING_EVAL_TASKS as _EMBEDDING_EVAL_TASKS,
)
from olmoearth_pretrain.internal.experiment import CommonComponents
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

# The teacher (primary bottleneck) width and the student (shipped embedding) widths:
# the student runs at 128 and its first 64 dims are trained as a self-sufficient
# Matryoshka prefix (own back-projection / Gram term / supervision head), so one
# stored artifact serves both widths by truncation (Tessera-v2 per-prefix heads).
REGISTER_DIM = 768
PROJECTION_DIMS = [128, 64]
# Hidden width of the 2-layer back-projection heads, for the arms that use one
# (``register_back_projection_hidden``). SimReg's ``(m, 2m, d)`` rule read at
# ``m = max(PROJECTION_DIMS)``, then held FIXED across prefixes so that prefix
# width is the only thing varying between Matryoshka heads. The heads are
# training-only, so this is free at serving time.
BACK_PROJECTION_HIDDEN = 256
# w0p1 (base 0.1) everywhere heads attach: best for the smaller register dims, and
# d768 has only w0.01/w1 completed evals, so this doubles as the missing d768 w0p1
# point. The sup768 variant is encoder-identical to the in-flight
# regbtl_v1_2_gdyn_d768_regsup_w0p1_newsampling_psuniform run.
SUPERVISION_BASE_WEIGHT_W0P1 = 0.1
# w1 (base 1.0): under the new sampler the w1 teacher leads w0p1 by ~1.4-1.9 mIoU on
# the ps=1 PASTIS exports at 520k+ (0.6400 vs 0.6210 S2, 0.6429 vs 0.6285 S1+S2) --
# and the teacher's ceiling is what the student inherits. The w1 sup768 arm is
# encoder-identical to the in-flight regbtl_v1_2_gdyn_d768_regsup_w1_newsampling_
# psuniform run.
SUPERVISION_BASE_WEIGHT_W1 = 1.0

# Param-group tag for the detached student, so one scheduler override covers it and
# its LR is logged as ``optim/LR (student)``.
STUDENT_LR_GROUP = "student"
# Every parameter of the detached student: the projection itself plus its per-prefix
# back-projections. The projection lives under DIFFERENT attribute names per
# architecture -- ``register_projection`` for linear, ``register_projection_student``
# for perceiver -- so the glob must cover both without a trailing dot, since
# OptimConfig.build_groups is strict and raises when a pattern matches no parameter
# (a per-architecture pattern list would fail on whichever variant is not in use).
STUDENT_PARAM_GLOBS = [
    "*register_projection*",
    "*register_back_projections*",
]

# --- the ``small`` backbone arms -------------------------------------------------
# The v1.2 size sweep's ViT-Small (384-d, depth 12, 6 heads) is very strong relative
# to base for its cost, so the w1 pcv students are re-run on it. The register width
# drops with the backbone (768 -> 384): wideread ties the bottleneck's ATTENTION to
# the encoder width, so d768 storage on a 384-d encoder would be wider than anything
# it reads. The student stays at [128, 64] -- the shipped widths don't change.
SMALL_SIZE_NAME = "small_shallow_decoder"
SMALL_REGISTER_DIM = 384
# ``small.py``'s LR from the size-sweep best-LR search: the small encoder trains at
# 2e-4, not the base recipe's 1e-4 that the regbtl chain otherwise inherits.
SMALL_LEARNING_RATE = 0.0002

# The PASTIS ws16/ps1 embedding evals duplicated onto the projected head at each
# Matryoshka width: the ``_proj{d}`` tasks probe (a prefix of)
# ``projected_registers`` instead of the register grid, so every in-loop eval step
# scores the same checkpoint at all three widths (768 / 128 / 64).
PASTIS_PROJ_EMBEDDING_LOOP_EVAL_TASKS = {
    f"{name}_proj{dim}": replace(
        task, eval_on_projected_registers=True, eval_projection_dim=dim
    )
    for name, task in PASTIS_EMBEDDING_LOOP_EVAL_TASKS.items()
    for dim in PROJECTION_DIMS
}


@dataclass
class StudentArm:
    """One student-side variation on the ``pcv_sup768_w1`` baseline.

    Every arm is a d768 teacher + a single detached perceiver student at
    [128, 64], identical to the in-flight
    ``regbtl_v1_2_gdyn_d768_proj128pcv_sup768_w1_newsamp_psuniform`` run except for
    the one field it changes -- that run IS the shared baseline, so no arm re-runs
    it. Each arm is its own training run: the students are cheap but the encoder is
    not, so hosting them together would have meant one process, and that turned out
    to need five perceiver students in memory at once (it OOM'd) plus a halved
    microbatch that would have moved the Gram statistics out of line with the
    baseline.

    Attributes:
        slug: Goes into the run and script names.
        constant_lr: Hold the student's LR flat after the encoder's warmup instead
            of inheriting its cosine decay to 0.1x.
        supervision_scale: Attach supervision heads to the student at this fraction
            of the register head's weight (needs supervision_source="both").
        gram_weight: Weight of the flat (mostly cross-scene) relational term.
    """

    slug: str
    constant_lr: bool = False
    supervision_scale: float | None = None
    gram_weight: float = 1.0


#: The student arms. Rationale for each is in its run script's docstring.
STUDENT_ARMS = {
    arm.slug: arm
    for arm in [
        StudentArm(slug="flatlr", constant_lr=True),
        StudentArm(slug="supstu0p1", supervision_scale=0.1),
        StudentArm(slug="supstu0p01", supervision_scale=0.01),
    ]
}


def build_arm_model_config(
    common: CommonComponents, arm: StudentArm, **kwargs: object
) -> LatentMIMConfig:
    """Baseline pcv student config with the arm's supervision placement applied."""
    return build_proj_model_config(
        common,
        base_weight=SUPERVISION_BASE_WEIGHT_W1,
        projection_type="perceiver",
        # Supervision arms need heads on BOTH widths so the student's can be scaled
        # independently; every other arm is the sup768 recipe (registers only).
        supervision_source="both" if arm.supervision_scale is not None else "registers",
        projection_supervision_weight_scale=arm.supervision_scale,
        **kwargs,  # type: ignore[arg-type]
    )


def apply_arm(
    config: LatentMIMTrainModuleConfig, arm: StudentArm
) -> LatentMIMTrainModuleConfig:
    """Apply the arm's distillation weights and LR schedule, in place."""
    config.projection_distill_gram_weight = arm.gram_weight
    if arm.constant_lr:
        add_student_lr_group(config, scheduler=student_constant_scheduler(config))
    return config


def student_constant_scheduler(
    train_module_config: LatentMIMTrainModuleConfig,
) -> ConstantWithWarmup:
    """Constant student LR that MIRRORS the encoder's warmup, then stops decaying.

    Warmup is copied from the shared scheduler rather than restated, so the two can
    never drift apart. Mirroring it matters: during warmup the teacher's own LR is
    still ramping and its representation moves fastest, so a student pinned at full
    LR from step 0 would chase its most unstable target at its largest step. After
    warmup the schedules separate -- the encoder decays toward its ``alpha_f`` floor
    while the student holds, which is the contrast the flat-LR arm tests.
    """
    shared = train_module_config.scheduler
    return ConstantWithWarmup(
        warmup=getattr(shared, "warmup", None),
        warmup_steps=getattr(shared, "warmup_steps", None),
        warmup_fraction=getattr(shared, "warmup_fraction", None),
        warmup_min_lr=getattr(shared, "warmup_min_lr", 0.0),
        units=getattr(shared, "units", "steps"),
    )


def add_student_lr_group(
    train_module_config: LatentMIMTrainModuleConfig,
    *,
    lr: float | None = None,
    scheduler: Scheduler | None = None,
) -> LatentMIMTrainModuleConfig:
    """Put the detached student in its own param group, with its own LR/schedule.

    The student otherwise sits in the encoder's group and inherits its
    ``CosWithWarmup(alpha_f=0.1)``, which cuts its LR 10x over a run. That is right
    for an encoder converging on a stationary objective and wrong for a head chasing
    a teacher that is still improving at the end of training -- the in-flight runs
    show ``projection/distill_cosine_d128`` RISING ~17x while that decay is applied.

    Args:
        train_module_config: The train module config to modify in place.
        lr: Student peak LR. None keeps the optimizer's (i.e. the encoder's).
        scheduler: Schedule for the student group; see
            :func:`student_constant_scheduler`. None keeps the shared one.

    Returns:
        The modified config.
    """
    opts: dict[str, float | str] = {"group_name": STUDENT_LR_GROUP}
    if lr is not None:
        opts["lr"] = lr
    train_module_config.optim_config.group_overrides = [
        *(train_module_config.optim_config.group_overrides or []),
        OptimGroupOverride(params=list(STUDENT_PARAM_GLOBS), opts=opts),
    ]
    if scheduler is not None:
        train_module_config.scheduler_overrides = {
            **(train_module_config.scheduler_overrides or {}),
            STUDENT_LR_GROUP: scheduler,
        }
    return train_module_config


def build_proj_model_config(
    common: CommonComponents,
    *,
    projection_type: str,
    supervision_source: str,
    base_weight: float = SUPERVISION_BASE_WEIGHT_W0P1,
    register_dim: int = REGISTER_DIM,
    size_name: str = ENCODER_SIZE_NAME,
    projection_supervision_weight_scale: float | None = None,
) -> LatentMIMConfig:
    """Wideread regbtl + regsup + a detached [128, 64] Matryoshka student.

    Args:
        common: The common experiment components.
        projection_type: ``"linear"`` or ``"perceiver"`` (the student architecture).
        supervision_source: ``"registers"`` (sup768), ``"both"`` (supboth) or
            ``"projection"`` (sup128) -- where the supervision heads attach.
        base_weight: Supervision base weight (w0p1 = 0.1 default; w1 = 1.0).
        register_dim: Teacher (primary bottleneck) width; 768 on the base backbone,
            :data:`SMALL_REGISTER_DIM` on the small one.
        size_name: Encoder/decoder size preset (base by default).
        projection_supervision_weight_scale: Scales the student's supervision heads
            relative to the register head's ``base_weight`` -- with w1, 0.1 is the
            w0p1 arm and 0.01 the w0p01 arm. Requires a supervision_source that
            builds projection heads ("both" / "projection").
    """
    config = build_wideread_regbtl_model_config(
        common,
        latent_self_attn=True,
        register_dim=register_dim,
        size_name=size_name,
    )
    config = add_register_supervision(
        config,
        include_latlon=False,
        base_weight=base_weight,
    )
    config.encoder_config.register_projection_dims = list(PROJECTION_DIMS)
    config.encoder_config.register_projection_type = projection_type
    config.supervision_source = supervision_source
    config.projection_supervision_weight_scale = projection_supervision_weight_scale
    return config


def apply_small_learning_rate(
    config: LatentMIMTrainModuleConfig,
) -> LatentMIMTrainModuleConfig:
    """Swap the base recipe's LR for the small preset's, in place.

    Everything else about the optimizer (fused AdamW, weight decay, warmup schedule,
    grad clipping) is left as the faster recipe sets it -- only the LR is a function
    of backbone size, and ``small.py`` sets it from the size sweep's best-LR search.
    """
    config.optim_config.lr = SMALL_LEARNING_RATE
    return config


def add_proj_loop_eval_beaker_job(
    trainer_config,
    module_path: str,
    *,
    embedding_eval_interval_steps: int | None = None,
):
    """In-loop evals on BOTH heads, routed through Beaker.

    The standard fifty_cities + PASTIS embedding evals probe the d768 registers as
    usual; the ``_proj128`` / ``_proj64`` duplicates probe the detached student (full
    width / 64d Matryoshka prefix), so the distillation quality is tracked per
    checkpoint at every shipped width without a separate eval launch.

    The PASTIS embedding tasks (base + projected) are placed FIRST in the task
    order: eval jobs at urgent priority can be preempted mid-run, and the last
    tasks in a job are the ones that systematically lose their metrics -- these
    six are the program's primary readout, so they run before the catalog tasks.

    ``embedding_eval_interval_steps`` overrides the 20k-step default on the
    embedding tasks (base + projected + fifty_cities). The proj runs' 14-task
    eval jobs run LONGER than 20k training steps, so consecutive jobs overlap
    while sharing one resumed W&B run, and the overlapping writer's rows are
    silently dropped (observed as missing S1+S2 projected metrics). 40k gives
    the job time to finish before the next one spawns. Must be a multiple of
    the checkpointer save_interval (5000).
    """
    embedding_tasks = {
        **PASTIS_EMBEDDING_LOOP_EVAL_TASKS,
        **PASTIS_PROJ_EMBEDDING_LOOP_EVAL_TASKS,
        **FIFTY_CITIES_LOOP_EVAL_TASKS,
    }
    if embedding_eval_interval_steps is not None:
        embedding_tasks = {
            name: replace(
                task, eval_interval=Duration.steps(embedding_eval_interval_steps)
            )
            for name, task in embedding_tasks.items()
        }
    evaluator = trainer_config.callbacks["downstream_evaluator"]
    evaluator.tasks = {
        **embedding_tasks,
        **evaluator.tasks,
    }
    evaluator.run_as_beaker_job = True
    evaluator.beaker_eval_module_path = module_path
    evaluator.beaker_eval_clusters = list(LOOP_EVAL_CLUSTERS)
    return trainer_config


# --- year-aligned AEF-trial in-loop evals, both heads -----------------------------
# The six early-read probes: the PASTIS S1+S2 bridge task shared with the older
# runs' in-loop sets, plus the year-aligned S1+S2+Landsat PASTIS / ethiopia / descals
# probes. The two _knn tasks carry AEF's balanced-trial protocol automatically
# (_aef_ps1_task attaches a BalancedTrialConfig whenever eval_mode is KNN), which is
# where the aeftrial_* metrics come from. Names are looked up in the canonical
# registry so a typo raises KeyError at import instead of silently dropping a task.
_PROJ_EARLYREAD_LOOP_EVAL_NAMES = (
    "pastis_ws16_ps1_sentinel1_sentinel2_pretrain_export",
    "pastis_year_aligned_ws16_ps1_sentinel1_sentinel2_landsat",
    "ethiopia_crops_year_aligned_ws16_ps1_sentinel1_sentinel2_landsat",
    "ethiopia_crops_year_aligned_ws16_ps1_sentinel1_sentinel2_landsat_knn",
    "descals_year_aligned_ws16_ps1_sentinel1_sentinel2_landsat",
    "descals_year_aligned_ws16_ps1_sentinel1_sentinel2_landsat_knn",
)


def set_proj_earlyread_loop_evals(
    trainer_config,
    module_path: str,
    *,
    interval_steps: int = 40000,
    projection_dim: int = 128,
):
    """REPLACE the eval set with the early-read probes on BOTH heads, via Beaker.

    The six tasks above probe the d768 teacher registers; their ``_proj{d}``
    duplicates probe the detached student (``eval_on_projected_registers``), so the
    shipped d128 embedding gets the same year-aligned LP/KNN/aeftrial readout
    in-loop. Like ``set_earlyread_loop_evals`` this DISCARDS the shared catalog --
    these runs are judged on the embedding product, and the catalog evals would
    inflate the eval job's runtime measuring axes the experiment does not turn on.

    ``interval_steps`` defaults to 40k, not the earlyread 20k: with the projected
    duplicates this is a 12-task job, and the proj runs' 14-task jobs are the ones
    that overflowed a 20k window and silently dropped their tail metrics (see
    ``add_proj_loop_eval_beaker_job``). Must be a multiple of the checkpointer
    save_interval (5000).
    """
    base_tasks = {
        name: replace(
            _EMBEDDING_EVAL_TASKS[name],
            eval_interval=Duration.steps(interval_steps),
        )
        for name in _PROJ_EARLYREAD_LOOP_EVAL_NAMES
    }
    proj_tasks = {
        f"{name}_proj{projection_dim}": replace(
            task,
            eval_on_projected_registers=True,
            eval_projection_dim=projection_dim,
        )
        for name, task in base_tasks.items()
    }
    evaluator = trainer_config.callbacks["downstream_evaluator"]
    # Student tasks FIRST: eval jobs at urgent can be preempted mid-run and the
    # tail tasks are the ones that lose their metrics -- the shipped d128 head
    # is the primary readout, so a clipped job should cost teacher cells, not
    # student cells (lesson from the 2026-08-18 baseline sweep, which ran
    # teacher-first and made the deliverable wait).
    evaluator.tasks = {**proj_tasks, **base_tasks}
    evaluator.run_as_beaker_job = True
    evaluator.beaker_eval_module_path = module_path
    evaluator.beaker_eval_clusters = list(LOOP_EVAL_CLUSTERS)
    return trainer_config


# --- AEF-trial + PASTIS in-loop evals, student widths only ------------------------
# The eight year-aligned AEF datasets on S1+S2+Landsat (kNN twins, which carry AEF's
# balanced-trial protocol automatically -- _aef_ps1_task attaches a
# BalancedTrialConfig whenever eval_mode is KNN, and that is where the aeftrial_*
# metrics come from) plus the year-aligned PASTIS task on the same input. Names are
# looked up in the canonical registry so a typo raises KeyError at import instead of
# silently dropping a task.
_AEFTRIAL_LOOP_EVAL_NAMES = tuple(
    f"{dataset}_ws16_ps1_sentinel1_sentinel2_landsat_knn"
    for dataset in AEF_SUPPLEMENTAL_YEAR_ALIGNED
) + ("pastis_year_aligned_ws16_ps1_sentinel1_sentinel2_landsat",)


def set_proj_aeftrial_loop_evals(
    trainer_config,
    module_path: str,
    *,
    interval_steps: int = 80000,
    projection_dims: tuple[int, ...] = (128, 64),
):
    """REPLACE the eval set with the AEF trials + PASTIS on the STUDENT widths.

    Nine tasks -- the eight AEF-supplemental datasets' kNN twins (which report
    ``aeftrial_{ridge,knn5,knn20}``) and PASTIS -- all on unmasked S1+S2+Landsat,
    duplicated at each entry of ``projection_dims``. Every task probes the detached
    student (``eval_on_projected_registers``); the d768 teacher is deliberately NOT
    scored here, since these runs are judged on the shipped embedding and the
    teacher rows would add half again to an already long job.

    Task ORDER is d128 before d64, PASTIS before the trials within each width. Eval
    jobs at urgent priority get preempted mid-run and the trailing tasks are the ones
    that systematically lose their metrics, so the order is the priority order: the
    shipped width first, and the segmentation readout -- the cell the width question
    hurts most -- ahead of the classification block.

    ``interval_steps`` defaults to **80k**, double the proj runs' 40k. This is an
    18-task job over ~159k windows per width at three modality passes each; the
    12-task early-read job already needed 40k to finish before its successor spawned,
    and consecutive jobs sharing one resumed W&B run silently drop the overlapping
    writer's rows. Must be a multiple of the checkpointer save_interval (5000).
    """
    base_tasks = {
        name: replace(
            _EMBEDDING_EVAL_TASKS[name], eval_interval=Duration.steps(interval_steps)
        )
        for name in _AEFTRIAL_LOOP_EVAL_NAMES
    }
    # Ordered dict comprehension: widths outer, PASTIS-first inner.
    ordered = sorted(base_tasks, key=lambda n: (not n.startswith("pastis"), n))
    tasks = {
        f"{name}_proj{dim}": replace(
            base_tasks[name],
            eval_on_projected_registers=True,
            eval_projection_dim=dim,
        )
        for dim in projection_dims
        for name in ordered
    }
    evaluator = trainer_config.callbacks["downstream_evaluator"]
    evaluator.tasks = tasks
    evaluator.run_as_beaker_job = True
    evaluator.beaker_eval_module_path = module_path
    evaluator.beaker_eval_clusters = list(LOOP_EVAL_CLUSTERS)
    return trainer_config

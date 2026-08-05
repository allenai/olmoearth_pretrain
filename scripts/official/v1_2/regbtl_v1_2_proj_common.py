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
        gram_within_weight: Weight of the within-scene (block-diagonal) term.
    """

    slug: str
    constant_lr: bool = False
    supervision_scale: float | None = None
    gram_weight: float = 1.0
    gram_within_weight: float = 0.0


#: The five arms. Rationale for each is in its run script's docstring.
STUDENT_ARMS = {
    arm.slug: arm
    for arm in [
        StudentArm(slug="flatlr", constant_lr=True),
        StudentArm(slug="supstu0p1", supervision_scale=0.1),
        StudentArm(slug="supstu0p01", supervision_scale=0.01),
        StudentArm(slug="gramwithin", gram_weight=0.5, gram_within_weight=0.5),
        StudentArm(slug="gramonly", gram_weight=0.0, gram_within_weight=1.0),
        # Both spatial remedies at once. The single-arm runs test two independent
        # routes to the same deficiency -- the objective applies almost no pressure
        # to discriminate CELLS within a scene, since the cosine term is satisfiable
        # by the scene-mean direction and ~98% of the flat Gram's pairs are
        # cross-scene. gramonly supplies that pressure relationally (every pair
        # within one scene); supstu0p1 supplies it directly (each cell must predict
        # the map modalities at its own location). Whether they compose, or one
        # subsumes the other, is not answerable from the single-knob arms.
        StudentArm(
            slug="gramonly_supstu0p1",
            supervision_scale=0.1,
            gram_weight=0.0,
            gram_within_weight=1.0,
        ),
        StudentArm(
            slug="gramwithin_supstu0p1",
            supervision_scale=0.1,
            gram_weight=0.5,
            gram_within_weight=0.5,
        ),
        # The same two combinations on a flat student LR. Only arms whose slug ends
        # in ``_flatlr`` hold the LR constant after warmup; every other arm inherits
        # the encoder's CosWithWarmup(alpha_f=0.1) and its 10x decay, which is the
        # baseline's schedule. Run as pairs so the schedule stays attributable
        # rather than folded into the combination.
        StudentArm(
            slug="gramonly_supstu0p1_flatlr",
            constant_lr=True,
            supervision_scale=0.1,
            gram_weight=0.0,
            gram_within_weight=1.0,
        ),
        StudentArm(
            slug="gramwithin_supstu0p1_flatlr",
            constant_lr=True,
            supervision_scale=0.1,
            gram_weight=0.5,
            gram_within_weight=0.5,
        ),
        # The same four combinations at a TENTH the student supervision weight.
        # Supervision trades early speed for late stability -- supboth_w1 (1.0x)
        # sits ~6 mIoU behind at 40k and only crosses sup768 at 400k, once sup768
        # has degraded. 0.1x reproduces that shape scaled down (-1.3 at 40k, -0.6
        # by 120k). 0.01x asks whether the protection survives at a dose small
        # enough to cost nothing early, or whether it is simply too weak to act.
        StudentArm(
            slug="gramonly_supstu0p01",
            supervision_scale=0.01,
            gram_weight=0.0,
            gram_within_weight=1.0,
        ),
        StudentArm(
            slug="gramwithin_supstu0p01",
            supervision_scale=0.01,
            gram_weight=0.5,
            gram_within_weight=0.5,
        ),
        StudentArm(
            slug="gramonly_supstu0p01_flatlr",
            constant_lr=True,
            supervision_scale=0.01,
            gram_weight=0.0,
            gram_within_weight=1.0,
        ),
        StudentArm(
            slug="gramwithin_supstu0p01_flatlr",
            constant_lr=True,
            supervision_scale=0.01,
            gram_weight=0.5,
            gram_within_weight=0.5,
        ),
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
    config.projection_distill_gram_within_weight = arm.gram_within_weight
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
    include_ndvi: bool = False,
    temporal_anchor: str | None = None,
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
        include_ndvi: Add the time-conditioned NDVI supervision arm (requires the
            ndvi extra-decode dataset/dataloader/train-module builders from
            ``regbtl_v1_2_regsup_common`` in the run script).
        projection_supervision_weight_scale: Scales the student's supervision heads
            relative to the register head's ``base_weight`` -- with w1, 0.1 is the
            w0p1 arm and 0.01 the w0p01 arm. Requires a supervision_source that
            builds projection heads ("both" / "projection").
        temporal_anchor: If set (``"year_start"``), the register READ becomes
            temporally anchored (tanchor). NOTE: the perceiver student mirrors the
            primary's ``register_temporal_anchor``, so a tanchor arm changes both
            the teacher's and the pcv student's reads at once.
    """
    config = build_wideread_regbtl_model_config(
        common,
        latent_self_attn=True,
        register_dim=register_dim,
        size_name=size_name,
    )
    if temporal_anchor is not None:
        config.encoder_config.register_temporal_anchor = temporal_anchor
    config = add_register_supervision(
        config,
        include_latlon=False,
        include_ndvi=include_ndvi,
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

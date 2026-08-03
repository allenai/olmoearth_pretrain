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
from dataclasses import replace

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
from olmoearth_pretrain.nn.flexi_vit import RegisterStudentSpec
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

# Param-group tag shared by every student group, so one scheduler override covers
# them all and their LR is logged as ``optim/LR (students)``.
STUDENT_LR_GROUP = "students"

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


def build_multi_student_model_config(
    common: CommonComponents,
    students: list[RegisterStudentSpec],
    *,
    base_weight: float = SUPERVISION_BASE_WEIGHT_W1,
    supervision_weight_scales: dict[str, float] | None = None,
    register_dim: int = REGISTER_DIM,
    size_name: str = ENCODER_SIZE_NAME,
    include_ndvi: bool = False,
    temporal_anchor: str | None = None,
) -> LatentMIMConfig:
    """Wideread regbtl + regsup + SEVERAL detached students on one encoder.

    Students are detached from the encoder and from each other, so this is the
    single-teacher form of a student-side sweep: N arms share one encoder
    trajectory, one data order and one seed, and the only thing that differs
    between them is the arm. The register (teacher) supervision head is always on;
    ``supervision_weight_scales`` optionally adds a per-student head at a scale of
    the teacher's base weight, which is how a supervision-weight sweep becomes arms
    of one run rather than one pretraining run per weight.

    Args:
        common: The common experiment components.
        students: The arms, one :class:`RegisterStudentSpec` each. Names become the
            metric namespace (``projection/<name>/...``), the eval task selector and
            the optimizer parameter glob, so keep them short and stable.
        base_weight: Register-supervision base weight (w1 = 1.0 by default: the
            teacher's ceiling is what every student inherits).
        supervision_weight_scales: Per-student supervision head weight, as a SCALE
            on ``base_weight`` -- with the w1 default, 0.1 is the w0p1 arm and 0.01
            the w0p01 arm. Students omitted (or given 0.0) get no supervision head,
            which is the ``sup768`` recipe the in-flight runs use.
        register_dim: Teacher (primary bottleneck) width.
        size_name: Encoder/decoder size preset.
        include_ndvi: Add the time-conditioned NDVI supervision arm.
        temporal_anchor: If set, the register read becomes temporally anchored.
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
    config.encoder_config.register_students = students
    config.supervision_source = "registers"
    if supervision_weight_scales:
        config.projection_supervision_weight_scales = dict(supervision_weight_scales)
    return config


def student_constant_scheduler(
    train_module_config: LatentMIMTrainModuleConfig,
) -> ConstantWithWarmup:
    """Constant student LR that MIRRORS the encoder's warmup, then stops decaying.

    Warmup is copied from the shared scheduler rather than restated, so the two can
    never drift apart. Mirroring it matters: during warmup the teacher's own LR is
    still ramping and its representation is moving fastest, so a student pinned at
    full LR from step 0 would be chasing its most unstable target at its largest
    step. After warmup the schedules separate -- the encoder decays toward its
    ``alpha_f`` floor, the student holds -- which is the whole contrast being tested.
    """
    shared = train_module_config.scheduler
    return ConstantWithWarmup(
        warmup=getattr(shared, "warmup", None),
        warmup_steps=getattr(shared, "warmup_steps", None),
        warmup_fraction=getattr(shared, "warmup_fraction", None),
        warmup_min_lr=getattr(shared, "warmup_min_lr", 0.0),
        units=getattr(shared, "units", "steps"),
    )


def add_student_lr_groups(
    train_module_config: LatentMIMTrainModuleConfig,
    student_lrs: dict[str, float],
    *,
    group_name: str = STUDENT_LR_GROUP,
    scheduler: Scheduler | None = None,
) -> LatentMIMTrainModuleConfig:
    """Give named students their own LR (and optionally their own schedule shape).

    The encoder's ``CosWithWarmup(alpha_f=0.1)`` decays the LR 10x over a run, which
    the students inherit only because they sit in the same param group. That is the
    wrong default for them: they chase a teacher that is still improving at the end
    of training, so their step size shrinks while their target keeps moving. Passing
    a ``scheduler`` here decouples the schedule SHAPE -- see
    :func:`student_constant_scheduler`, which keeps the encoder's warmup and drops
    only the decay. ``student_lrs`` decouples the peak.

    Args:
        train_module_config: The train module config to modify in place.
        student_lrs: Peak LR per student name.
        group_name: Tag shared by the student groups, used to look up ``scheduler``.
            Also the label their LR is logged under (``optim/LR (<group_name>)``).
        scheduler: Schedule for the student groups. None keeps the shared one.

    Returns:
        The modified config.
    """
    overrides = list(train_module_config.optim_config.group_overrides or [])
    for name, lr in student_lrs.items():
        overrides.append(
            OptimGroupOverride(
                params=[f"*register_students.{name}.*"],
                opts={"lr": lr, "group_name": group_name},
            )
        )
    train_module_config.optim_config.group_overrides = overrides
    if scheduler is not None:
        train_module_config.scheduler_overrides = {
            **(train_module_config.scheduler_overrides or {}),
            group_name: scheduler,
        }
    return train_module_config


def build_multi_student_eval_tasks(
    student_names: list[str],
    dims: list[int] | None = None,
    task_names: list[str] | None = None,
) -> dict:
    """PASTIS embedding tasks duplicated per (student, Matryoshka width).

    The d768 teacher tasks are unchanged; each student adds
    ``<task>_<student>_proj{dim}`` so one eval job scores every arm against the same
    checkpoint. Task count is ``len(students) * len(dims) * len(task_names)``, so a
    wide sweep must narrow ``dims`` and/or ``task_names``: the note on
    :func:`add_proj_loop_eval_beaker_job` explains why long eval jobs silently drop
    their tail metrics. Both default to everything.
    """
    tasks = {}
    base_tasks = {
        name: task
        for name, task in PASTIS_EMBEDDING_LOOP_EVAL_TASKS.items()
        if task_names is None or name in task_names
    }
    if task_names is not None and len(base_tasks) != len(task_names):
        raise ValueError(
            f"unknown PASTIS embedding task(s) {sorted(set(task_names) - set(base_tasks))}; "
            f"available: {sorted(PASTIS_EMBEDDING_LOOP_EVAL_TASKS)}"
        )
    for student in student_names:
        for name, task in base_tasks.items():
            for dim in dims if dims is not None else PROJECTION_DIMS:
                tasks[f"{name}_{student}_proj{dim}"] = replace(
                    task,
                    eval_on_projected_registers=True,
                    eval_projection_dim=dim,
                    eval_projection_student=student,
                )
    return tasks


def add_multi_student_loop_eval_beaker_job(
    trainer_config,
    module_path: str,
    student_names: list[str],
    *,
    dims: list[int] | None = None,
    task_names: list[str] | None = None,
    embedding_eval_interval_steps: int | None = None,
):
    """In-loop evals on the d768 teacher head and on every student, via Beaker.

    The multi-student analogue of :func:`add_proj_loop_eval_beaker_job`. Since one
    job now scores every arm, the task list grows with the sweep -- keep ``dims``
    narrow and the interval long enough that consecutive jobs cannot overlap (see
    that function's note on dropped tail metrics).
    """
    embedding_tasks = {
        **PASTIS_EMBEDDING_LOOP_EVAL_TASKS,
        **build_multi_student_eval_tasks(student_names, dims, task_names),
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
    evaluator.tasks = {**embedding_tasks, **evaluator.tasks}
    evaluator.run_as_beaker_job = True
    evaluator.beaker_eval_module_path = module_path
    evaluator.beaker_eval_clusters = list(LOOP_EVAL_CLUSTERS)
    return trainer_config


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

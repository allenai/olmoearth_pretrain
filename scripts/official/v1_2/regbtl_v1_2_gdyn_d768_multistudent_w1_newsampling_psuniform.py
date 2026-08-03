"""d768 teacher + five detached PERCEIVER students: LR schedule, supervision weight.

Why one run instead of five. The students are detached from the encoder AND from
each other, so N students leave the encoder's trajectory bit-identical and cost only
their own readouts. Five separate runs would each get their own encoder trajectory,
and the projected PASTIS probes move +/-1 mIoU between adjacent checkpoints of the
SAME run -- variance the size of every effect below. Here all five arms see one
teacher, one data order, one seed, and (since the Gram subsample is drawn once per
microbatch and shared) one set of relational pairs. The only difference between two
arms is the arm.

Every arm is a PERCEIVER student. A per-cell Linear and a wideread Perceiver
bottleneck are different enough layers that their optimal LR need not be related, so
a linear arm would not stand in for a perceiver one -- and the perceiver student IS
the deployed d128 architecture, which is what any of these knobs would change.

--- (1) LR schedule ------------------------------------------------------------

On the in-flight ``proj128{pcv,lin}_sup768_w1`` runs the students plateau near 54
mIoU while the teacher climbs past 63, and ``projection/distill_cosine_d128`` RISES
monotonically (0.004 at 26k -> 0.072 at 356k, ~17x): the students fall further
behind a target that is still improving. Meanwhile the shared
``CosWithWarmup(alpha_f=0.1)`` cuts their LR 10x over the run, purely because they
sit in the encoder's param group and inherit its schedule. Decaying to a floor is
right for an encoder converging on a stationary objective and wrong for a head
chasing a live teacher.

* ``pcv_base`` -- 1e-4 on the shared cosine decay. Reproduces the in-flight arm, so
  the schedule contrast is measured under THIS run's teacher rather than across
  runs (three encoder-identical runs in this project read 63.31 / 62.83 / 62.94 at
  step 320k, so cross-run noise alone is ~0.5 mIoU -- the size of the effects here).
* ``pcv_flat`` -- 1e-4 with the SAME warmup, then held constant
  (:func:`student_constant_scheduler`). Warmup is mirrored rather than dropped: while
  the encoder's LR is still ramping the teacher representation moves fastest, and a
  student at full LR from step 0 would chase its most unstable target at its largest
  step. After warmup the two separate, which is the contrast being tested.

The distillation WEIGHTS are deliberately not swept: AdamW is invariant to a global
rescaling of the loss, and the students' parameters see gradient from the
distillation terms alone, so scaling both is a no-op -- only their ratio, or the LR,
does anything.

--- (2) Supervision weight on the student head ---------------------------------

Adding a supervision head to the student at the teacher's own weight cost 2-5 mIoU
on the single-student runs (``supboth_w1`` vs ``sup768_w1``: -2.1 to -6.4 for pcv),
while 0.1x was roughly neutral. The optimum over {0, 0.1, 1.0} sat at ZERO -- but
those points came from runs whose teacher weight moved with the student's, so the
tail was never cleanly measured. Here the teacher stays at w1 and only the student's
head weight varies. Both sit on the CONSTANT schedule, so ``pcv_flat`` -- not
``pcv_base`` -- is their zero point and the head weight is the single difference:

* ``pcv_sup_w0p1``  -- scale 0.1.
* ``pcv_sup_w0p01`` -- scale 0.01. The untested tail.
* the zero point is ``pcv_base`` (no head), so no extra arm is needed for it.

No scale-1.0 arm: ``supboth_w1`` already measured it and it is clearly worse, so
spending a perceiver student to re-derive that is waste. The cost is that this run
has no internal calibration point for the supervision mechanism -- if both arms come
back neutral, confirm the heads are actually attached via the
``supervision_projection_<arm>/d{128,64}/<modality>`` losses (they are only logged
when a head exists) rather than assuming the weight simply did not matter.

--- (3) Within-scene Gram ------------------------------------------------------

* ``pcv_gram_within`` -- splits the relational term 50/50 between the existing flat
  Gram and the block-diagonal one. The flat matrix is built over the flattened
  ``[B * N]`` grid, so at a microbatch of 64 only ~1/64 of its 4.2M pairs relate two
  cells of the SAME scene; the block-diagonal form gives 100% within-scene pairs at
  the same pair count, since m blocks of k cells cost O(m*k^2) against O((m*k)^2).
  Segmentation probes discriminate within a scene, so this asks whether the pairs
  that metric depends on were simply too rare to matter. Also on the constant
  schedule, against ``pcv_flat``.

--- How the arms chain --------------------------------------------------------

Every comparison is ONE hop from a control that shares its regime:

    pcv_base --(schedule)--> pcv_flat --(sup 0.1)-----> pcv_sup_w0p1
                                      --(sup 0.01)----> pcv_sup_w0p01
                                      --(gram mix)----> pcv_gram_within

--- What this run cannot settle ------------------------------------------------

If every arm plateaus together, the limit is the 128-dim budget or the moving
teacher itself, and the next experiment is offline distillation from a FROZEN
finished teacher -- not another schedule. That control is also far cheaper (no
encoder backward), so a flat result here should redirect rather than escalate.

--- Recipe ---------------------------------------------------------------------

Matches ``regbtl_v1_2_gdyn_d768_proj128pcv_sup768_w1_newsampling_psuniform``:
wideread regbtl d768, regsup base_weight 1.0 (w1) on the d768 registers, 1fwd +
fused AdamW, decorrelated sampler at uniform patch sizes. Encoder-identical to that
run and to ``regbtl_v1_2_gdyn_d768_regsup_w1_newsamp_psuniform``, so the d768
register evals double as a sanity anchor against both.

Deliberately NOT on the tanchor / ndvi recipe: those sweeps are still early (one 40k
eval point), and tanchor in particular re-anchors the perceiver STUDENT's read as
well as the teacher's, which would confound every arm here. Once they resolve,
``build_multi_student_model_config`` takes ``temporal_anchor`` / ``include_ndvi``
and this file becomes a one-line rebase onto the winner.
"""

import logging

from base import build_trainer_config as _base_build_trainer_config
from regbtl_v1_2_faster_common import build_faster_train_module_config
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_common_components,
    build_dataset_config,
    build_visualize_config,
)
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_dataloader_config as _base_build_dataloader_config,
)
from regbtl_v1_2_newsampling_common import (
    apply_microbatch,
    apply_new_sampling,
    apply_uniform_patch_sizes,
)
from regbtl_v1_2_proj_common import (
    PROJECTION_DIMS,
    SUPERVISION_BASE_WEIGHT_W1,
    add_multi_student_loop_eval_beaker_job,
    add_student_lr_groups,
    build_multi_student_model_config,
    student_constant_scheduler,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.flexi_vit import RegisterStudentSpec
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_gdyn_d768_multistudent_w1_newsampling_psuniform.py"
)

# The encoder's own LR, i.e. what the in-flight single-student arms give their
# students by default (they simply share the encoder's param group). Kept as the
# peak everywhere: the schedule arm changes the SHAPE, not the height, so the two
# are not confounded.
BASE_LR = 1e-4


def _pcv(name: str) -> RegisterStudentSpec:
    """One perceiver-student arm at the shipped [128, 64] widths."""
    return RegisterStudentSpec(
        name=name, projection_type="perceiver", dims=list(PROJECTION_DIMS)
    )


# Names are the metric namespace (``projection/<name>/...``), the eval task selector
# and the optimizer parameter glob, so they must stay stable across resumes.
STUDENTS = [
    _pcv("pcv_base"),
    _pcv("pcv_flat"),
    _pcv("pcv_sup_w0p1"),
    _pcv("pcv_sup_w0p01"),
    _pcv("pcv_gram_within"),
]

# Every arm EXCEPT pcv_base runs on the constant (post-warmup) schedule. pcv_base
# alone stays in the encoder's group on the shared CosWithWarmup -- it is the anchor
# to the in-flight run and the control for the schedule contrast. The peak LR is
# identical everywhere, so these groups exist solely to carry the schedule.
#
# Why the supervision and gram arms sit on CONSTANT rather than on pcv_base's decay:
# both knobs act by perturbing what the student optimizes, so their effect is
# proportional to how much the student can still move. Under the decay the student's
# LR is down 10x by the end of the run, which compresses any such effect toward zero
# -- measuring there risks a false "this weight is neutral" that is really "the
# student was already frozen". It also matches where these results would be USED: if
# flat wins, the shipped recipe is flat, and a knob tuned under decay need not
# transfer to it. The cost is that if flat LOSES, these three arms were measured in a
# regime we would not ship; that is the accepted risk, taken because the rising
# distillation loss is direct evidence that the decay is the wrong schedule here.
STUDENT_LRS = {
    "pcv_flat": BASE_LR,
    "pcv_sup_w0p1": BASE_LR,
    "pcv_sup_w0p01": BASE_LR,
    "pcv_gram_within": BASE_LR,
}

# Supervision head weight as a scale on the w1 register head. Arms not listed get no
# head at all (the sup768 recipe), which is the zero point of this sweep.
SUPERVISION_WEIGHT_SCALES = {
    "pcv_sup_w0p1": 0.1,
    "pcv_sup_w0p01": 0.01,
}

# pcv_gram_within splits the relational budget evenly between the flat (cross-scene)
# and block-diagonal (within-scene) Gram terms, holding the TOTAL relational weight
# at the 1.0 every other arm uses -- so this arm changes the mix, not the amount.
DISTILL_OVERRIDES = {
    "pcv_gram_within": {"gram_weight": 0.5, "gram_within_weight": 0.5},
}

# Five arms at both shipped widths on the S2-only PASTIS probe = 10 eval tasks, the
# size known to fit a 40k cadence (the 14-task proj jobs already ran long enough that
# a 20k cadence made consecutive jobs overlap and silently drop their tail metrics).
# d64 is worth carrying now that every arm is a perceiver: pcv's measured edge over
# lin was AT d64 (+1.2 to +2.2 mIoU), so that is where a student-side fix is most
# likely to show. The d768 teacher still gets both PASTIS tasks, unchanged.
STUDENT_EVAL_TASKS = ["pastis_ws16_ps1_sentinel2_pretrain_export"]
STUDENT_EVAL_DIMS = [128, 64]


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d768 teacher + the eight detached students, w1 supervision on the registers."""
    return build_multi_student_model_config(
        common,
        STUDENTS,
        base_weight=SUPERVISION_BASE_WEIGHT_W1,
        supervision_weight_scales=SUPERVISION_WEIGHT_SCALES,
    )


def build_dataloader_config(common: CommonComponents):
    """Newsampling dataloader at uniform patch sizes."""
    return apply_uniform_patch_sizes(
        apply_new_sampling(_base_build_dataloader_config(common))
    )


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """1fwd + fused AdamW at the newsampling microbatch, plus the student LR groups."""
    config = apply_microbatch(build_faster_train_module_config(common))
    config.projection_distill_overrides = DISTILL_OVERRIDES
    # The constant schedule copies its warmup from the shared one, so the arms only
    # diverge once the encoder starts decaying.
    return add_student_lr_groups(
        config, STUDENT_LRS, scheduler=student_constant_scheduler(config)
    )


def build_trainer_config(common: CommonComponents):
    """Base trainer + in-loop evals on the d768 head and on every student."""
    return add_multi_student_loop_eval_beaker_job(
        _base_build_trainer_config(common),
        MODULE_PATH,
        [spec.name for spec in STUDENTS],
        dims=STUDENT_EVAL_DIMS,
        task_names=STUDENT_EVAL_TASKS,
        embedding_eval_interval_steps=40000,
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

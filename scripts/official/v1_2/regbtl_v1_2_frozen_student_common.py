"""Shared pieces for the FROZEN-TEACHER student continuations.

Motivation (2026-08-07). ``lin_sup768_w1`` finished its 300 epochs while still
improving: at 665k its d128 student was climbing at +0.71 mIoU/100k on the
in-loop ps=1 PASTIS probe while its d768 teacher climbed at only +0.30. Decompose
that and roughly 62% of the student's gain is RETENTION improving (0.44/100k),
not the teacher getting better (0.30 x 0.894 = 0.27/100k). Retention headroom
alone is enough to win: the student sits at 89.4% of its teacher on the aligned
sweep (0.5812 / 0.6503), and 90.8% of the SAME teacher would be 0.5905 -- past
native d128's 0.5853. 90.4-91.2% is already achieved by ``small_pcv_supboth_w1``,
so this is not an extrapolation into unobserved territory.

So: stop training the teacher, keep training the student. Freezing also makes the
distillation target STATIONARY, which is the one change the distillation
literature most consistently supports (Anil et al. 2018 find stale teachers are
harmless; RCO / ESKD argue a fixed anchor beats a moving converged one). It is
also the cleanest test of the artifact hypothesis: ``distill_cosine_d128`` rose
~8x across the parent run because the target kept moving, and against a frozen
teacher it should fall from step one.

For the ``lin`` student this is a small, nearly-convex fitting problem -- a
Linear(768, 128) plus a Linear(128, 768) back-projection. That it has NOT
converged after 665k steps is itself evidence the only thing preventing
convergence is target motion.

How the freeze works, with no new model code. ``lin_sup768_w1`` carries a single
AdamW group (verified: no ``group_overrides``), so setting ``optim_config.lr = 0``
stops every parameter except the ones we then break out into the student group.
AdamW's decoupled weight decay is also scaled by lr, so nothing decays either.
Frozen: encoder, primary d768 bottleneck (the teacher), decoder, d768 supervision
heads. Moving: ``*register_projection*`` and ``*register_back_projections*``.

Not frozen, deliberately: masking/augmentation still vary per batch, so the
teacher's OUTPUTS differ batch to batch even though its function is fixed. That
is target noise, not target drift -- student and teacher read the same forward
pass, which is exactly the "consistent views" condition Beyer et al. (2022)
identify as necessary. EMA is a non-issue here: the parent runs at
``ema_decay = (1.0, 1.0)``, i.e. the target encoder never updates.

The pretext loss still computes and backprops into frozen parameters, wasting
roughly 3-5x the compute a real freeze would need. At ~8 GPU-days per arm that is
not worth new code to avoid; if these arms pay off, cache
(encoder tokens, teacher registers) and fit the student offline instead.

VERIFYING THE FREEZE. Two signals, both visible within an hour:
  * the in-loop d768 PASTIS eval must be EXACTLY FLAT -- it is a deterministic
    function of frozen weights, so any movement past eval noise means something
    is still training;
  * ``projection/distill_cosine_d128`` must turn DOWNWARD, against the ~8x rise
    the parent run showed.
"""

import logging

from olmo_core.optim.scheduler import ConstantWithWarmup
from regbtl_v1_2_proj_common import add_student_lr_group

from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

# The checkpoint both arms continue from: lin_sup768_w1 at the end of its 300
# epochs. Its final permanent checkpoint; the same one the 20260806 embedding
# sweep scored at 0.5812 on aligned PASTIS S2.
PARENT_RUN = "regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsamp_psuniform"
PARENT_CHECKPOINT = (
    f"/weka/dfive-default/olmoearth_pretrain/checkpoints/gabrielt/"
    f"{PARENT_RUN}/step667200"
)

# Warmup for the re-warmed arm. Short on purpose: the student is resuming from a
# trained state against a target that is not moving, so there is nothing to
# stabilise against -- this only avoids the first-step shock of jumping 1e-5 ->
# 1e-4 with fresh Adam moments.
REWARM_WARMUP_STEPS = 2000


def freeze_all_but_student(
    config: LatentMIMTrainModuleConfig,
    *,
    student_lr: float,
    warmup_steps: int,
) -> LatentMIMTrainModuleConfig:
    """Zero the shared LR and give the student its own group and schedule.

    Args:
        config: Train module config to modify in place.
        student_lr: Peak LR for the student group.
        warmup_steps: Linear warmup for the student group. Must be >= 1:
            ``ConstantWithWarmup`` divides by it (0 raises ZeroDivisionError) and
            rejects None outright. Pass 1 for "no warmup" -- that puts step 0 at
            lr 0 and every step after at full LR, which is what the floor arm
            wants since it is already sitting at this LR.

    Returns:
        The modified config.
    """
    if warmup_steps < 1:
        raise ValueError(
            f"warmup_steps must be >= 1 (ConstantWithWarmup divides by it), got "
            f"{warmup_steps}; pass 1 for no warmup"
        )
    config.optim_config.lr = 0.0
    return add_student_lr_group(
        config,
        lr=student_lr,
        # Constant, not cosine: with a stationary target the student is trying to
        # CONVERGE, and a decay schedule over an extension of unknown sufficient
        # length would just re-impose the LR floor this run exists to escape.
        scheduler=ConstantWithWarmup(warmup_steps=warmup_steps, units="steps"),
    )

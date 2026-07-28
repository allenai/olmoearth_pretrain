"""Shared config builders for open-set supervised *mid-training*.

Instead of training the open-set probe jointly with SSL from scratch (the
``open_set_{only,osm}_d*`` runs, which slowed SSL learning: the probe's gradient
inflated the total grad norm under a fixed clip, and the random probe injected
noise from step 0), these runs start from a finished register-bottleneck
pretraining checkpoint and add the probe in two in-run stages:

1. steps < ``FREEZE_BACKBONE_UNTIL_STEP``: everything loaded from the checkpoint
   is frozen and only the (freshly initialized) open-set probe trains -- a linear
   probe fit against a fixed encoder;
2. afterwards the whole model unfreezes and trains at a low backbone LR with the
   SSL loss plus the (w0.1) supervised loss (LP-then-FT, Kumar et al. 2022).

The model, dataset (full osm_sampling + open-set concat), dataloader, masking and
sup-loss weight are identical to ``open_set_base``; only the initialization,
freeze schedule, LRs and duration differ.

The runs are initialized via ``trainer.load_path`` from checkpoints whose model
lacks the probe parameters, so they must be launched with
``OE_LOAD_SKIP_MISMATCHED_KEYS=1`` in the environment (propagated to Beaker by
``internal/common.py``): the probe tensors are dropped from the load plan and
keep their fresh initialization, and the optimizer starts fresh.
"""

import logging

from olmo_core.optim import OptimGroupOverride
from olmo_core.train.common import Duration
from open_set_base import (
    build_train_module_config as _build_open_set_train_module_config,
)
from open_set_base import build_trainer_config as _build_open_set_trainer_config

from olmoearth_pretrain.internal.experiment import CommonComponents
from olmoearth_pretrain.train.train_module.open_set_latentmim import (
    OpenSetLatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

WANDB_PROJECT = "2026_07_28_open_set_mid"

# Finished wideread + regsup (w0p1) pretraining checkpoints (667,200 steps = 300
# epochs), verified state-dict-compatible with the open-set model on this branch
# (identical keys/shapes except the 4 fresh open_set_probe tensors).
LOAD_PATHS = {
    768: "/weka/dfive-default/olmoearth_pretrain/checkpoints/gabrielt/regbtl_v1_2_gdyn_d768_regsup_w0p1/step667200",
    128: "/weka/dfive-default/olmoearth_pretrain/checkpoints/gabrielt/regbtl_v1_2_gdyn_d128_wideread_regsup_w0p1/step667200",
}

# Steps 0..N: only the open-set probe trains (backbone frozen); then full model.
FREEZE_BACKBONE_UNTIL_STEP = 40_000
# Total mid-training duration. The concat dataset is ~2.55M samples at batch 512
# (~4,980 steps/epoch), so this is ~28 epochs / ~100k unfrozen steps.
TOTAL_STEPS = 140_000
# Low LR for the pretrained backbone after unfreezing (pretraining used 1e-4 from
# scratch; the checkpoint is already converged, we only want a gentle nudge).
BACKBONE_LR = 3e-5
# High LR for the probe's own param group: linear probes against a frozen encoder
# converge far faster at ~1e-3, and a converged probe is a cleaner target for the
# encoder when it unfreezes.
PROBE_LR = 1e-3


def build_mid_train_module_config(
    common: CommonComponents,
) -> OpenSetLatentMIMTrainModuleConfig:
    """Open-set train module with the freeze schedule and two-tier LR."""
    config = _build_open_set_train_module_config(common)
    config.freeze_backbone_until_step = FREEZE_BACKBONE_UNTIL_STEP
    config.optim_config.lr = BACKBONE_LR
    # The probe gets its own param group at a higher LR; the (single, cosine)
    # scheduler scales each group from its own base LR. During the frozen stage
    # only this group receives gradients anyway.
    config.optim_config.group_overrides = [
        OptimGroupOverride(params=["open_set_probe.*"], opts={"lr": PROBE_LR})
    ]
    return config


def build_mid_trainer_config(
    common: CommonComponents, module_path: str, register_dim: int
):
    """Trainer config: init from the pretrained checkpoint, steps-based duration.

    ``load_path`` only applies on the first start; preemption restarts resume
    from the run's own save folder (with trainer + optimizer state) as usual.
    """
    trainer_config = _build_open_set_trainer_config(common, module_path)
    trainer_config.max_duration = Duration.steps(TOTAL_STEPS)
    trainer_config.load_path = LOAD_PATHS[register_dim]
    trainer_config.load_trainer_state = False
    trainer_config.load_optim_state = False
    trainer_config.callbacks["wandb"].project = WANDB_PROJECT
    return trainer_config

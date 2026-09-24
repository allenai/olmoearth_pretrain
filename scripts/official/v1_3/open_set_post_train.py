r"""Launch script: post-train the v1.3 base checkpoint with the open-set probe.

Instead of training the open-set probe jointly with SSL from scratch
(``open_set_only.py`` / ``open_set_osm.py``), this starts from the finished v1.3
base checkpoint and adds the probe in ONE run with two in-run phases (see
``OpenSetLatentMIMTrainModule``):

1. steps ``< FREEZE_BACKBONE_UNTIL_STEP``: everything loaded from the checkpoint is
   frozen and only the freshly initialized open-set probe trains -- a linear probe
   fit against the fixed d768 register grid. Only the encoder runs (no decoder /
   target / map / student losses), so these steps are cheap.
2. afterwards the backbone unfreezes at a low LR and the probe is FROZEN: the model
   trains on the full v1.3 objective (latent-MIM + map supervision + student
   distillation) plus the supervised loss through the fixed probe, on the
   osm_sampling + open-set concatenation.

Initialization. The release checkpoint was written by the ``gabi/perceiver``
training branch under legacy parameter names, so it cannot be loaded through
``trainer.load_path``. Convert it ONCE to the current layout, then point
``V13_BASE_WEIGHTS`` at the resulting ``weights.pth``::

    python scripts/official/v1_3/convert_legacy_checkpoint.py \\
        {V13_BASE_CHECKPOINT} {V13_BASE_CONVERTED_DIR}

The train module loads those weights at construction (``init_weights_path``),
letting only the ``open_set_probe.*`` tensors keep their fresh initialization
(``init_weights_allow_missing``). Preemption restarts resume from the run's own
save folder (with trainer + optimizer state) as usual and override the init.

Usage (from the repo root)::

    python scripts/official/v1_3/open_set_post_train.py launch open_set_post_train \\
        ai2/jupiter --launch.num_gpus=8
"""

import logging

from base import build_visualize_config
from olmo_core.optim import OptimGroupOverride
from olmo_core.train.common import Duration
from open_set_base import (
    build_common_components,
    build_dataloader_config,
    build_model_config,
    build_osm_plus_open_set_dataset_config,
)
from open_set_base import build_train_module_config as _build_open_set_train_module
from open_set_base import build_trainer_config as _build_open_set_trainer_config

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.train.train_module.open_set_latentmim import (
    OpenSetLatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

# Path (relative to the repo root) used by the in-loop Beaker eval jobs to rebuild
# this exact model config when loading a checkpoint.
MODULE_PATH = "scripts/official/v1_3/open_set_post_train.py"

# The v1.3 release run (see v1_3/README.md), at the end of its 300 epochs; the same
# checkpoint shipped as OlmoEarth-v1_3-Base.
V13_BASE_CHECKPOINT = (
    "/weka/dfive-default/olmoearth_pretrain/checkpoints/gabrielt/"
    "regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsamp_psuniform_stunorm_mlpgram1/"
    "step667200"
)
# Output of convert_legacy_checkpoint.py for the checkpoint above (config.json +
# weights.pth under the current parameter names).
V13_BASE_CONVERTED_DIR = V13_BASE_CHECKPOINT + "_converted"
V13_BASE_WEIGHTS = V13_BASE_CONVERTED_DIR + "/weights.pth"

# Steps 0..N: only the open-set probe trains (backbone frozen); then the backbone
# trains against the frozen probe.
FREEZE_BACKBONE_UNTIL_STEP = 40_000
# Total post-training duration. The concat dataset is ~2.59M samples (1,138,828
# osm_sampling + 1,448,494 open-set) at batch 512 (~5,050 steps/epoch), so this is
# ~28 epochs / ~100k unfrozen steps.
TOTAL_STEPS = 140_000
# Low LR for the pretrained backbone after unfreezing (pretraining used 1e-4 from
# scratch; the checkpoint is converged, we only want a gentle nudge).
BACKBONE_LR = 3e-5
# High LR for the probe's own param group: linear probes against a frozen encoder
# converge far faster at ~1e-3, and a converged probe is a cleaner target for the
# backbone when it unfreezes.
PROBE_LR = 1e-3


def build_train_module_config(
    common: CommonComponents,
) -> OpenSetLatentMIMTrainModuleConfig:
    """Open-set train module with the freeze schedule, two-tier LR and init weights."""
    config = _build_open_set_train_module(common)
    config.freeze_backbone_until_step = FREEZE_BACKBONE_UNTIL_STEP
    config.freeze_probe_after_unfreeze = True
    config.optim_config.lr = BACKBONE_LR
    # The probe gets its own param group at a higher LR; the (single, cosine)
    # scheduler scales each group from its own base LR. During the frozen phase only
    # this group receives gradients anyway; after it, only the backbone group does.
    config.optim_config.group_overrides = [
        OptimGroupOverride(params=["open_set_probe.*"], opts={"lr": PROBE_LR})
    ]
    config.init_weights_path = V13_BASE_WEIGHTS
    config.init_weights_allow_missing = ["open_set_probe."]
    return config


def build_dataset_config(common: CommonComponents):
    """osm_sampling + open-set concatenation (full v1.3 objective + probe)."""
    return build_osm_plus_open_set_dataset_config(common)


def build_trainer_config(common: CommonComponents):
    """Trainer with a steps-based duration and the in-loop evals pointed here."""
    trainer_config = _build_open_set_trainer_config(common, MODULE_PATH)
    trainer_config.max_duration = Duration.steps(TOTAL_STEPS)
    return trainer_config


if __name__ == "__main__":
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )

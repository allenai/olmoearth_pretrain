"""Arm J: ``control_pix512.py`` with a JEPA target (EMA encoder, final-layer tokens).

The earlier "EMA" ablation exited every token at layer 0 (an EMA of the patch embedding,
not JEPA), and the full-depth one ran its target encoder in train mode (drop-path 0.1
and band dropout on the targets). Here the target is a full EMA copy of the encoder,
decay 0.996 -> 1.0 over training, sensor tokens exit after all 12 layers, the
decode-only maps keep layer-0 targets, and the target encoder runs in eval mode.
Pixel targets need the projection-only target, so this builds on the patch-target
control ``v1_3_rc_ld1_pix512``.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    build_common_components,
    build_dataset_config,
    build_visualize_config,
)
from common import build_dataloader_config as _build_dataloader_config  # noqa: E402
from common import build_model_config as _build_model_config  # noqa: E402
from common import build_train_module_config as _build_train_module_config  # noqa: E402
from common import build_trainer_config as _build_trainer_config  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/claude_findings/arm_jepa.py"
WANDB_PROJECT = "20261013_claude_findings"
ENCODER_DEPTH = 12
EMA_DECAY = (0.996, 1.0)


def build_model_config(common: CommonComponents):
    """The control model with a full encoder copy as the target."""
    return _build_model_config(common, projection_only_target=False)


def build_dataloader_config(common: CommonComponents):
    """The control dataloader."""
    return _build_dataloader_config(common)


def build_train_module_config(common: CommonComponents):
    """The control train module with EMA final-layer targets."""
    config = _build_train_module_config(common)
    sensors = {"sentinel2_l2a", "sentinel1", "landsat"}
    config.token_exit_cfg = {
        modality: (ENCODER_DEPTH if modality in sensors else 0)
        for modality in config.token_exit_cfg
    }
    config.ema_decay = EMA_DECAY
    config.target_encoder_eval_mode = True
    return config


def build_trainer_config(common: CommonComponents):
    """The control's in-loop evals, re-importing THIS module."""
    return _build_trainer_config(common, MODULE_PATH, wandb_project=WANDB_PROJECT)


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

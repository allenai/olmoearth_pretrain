"""Arm E: ``control_pixtgt.py`` + augmentation + centered targets + WorldCover fix.

Arms A and C together, plus WorldCover class values that match its normalization
(and nodata ignored) in the supervision head.
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

MODULE_PATH = "scripts/official/v1_3/claude_findings/arm_combined.py"
WANDB_PROJECT = "20261013_claude_findings"


def build_model_config(common: CommonComponents):
    """The control model with the WorldCover class fix."""
    return _build_model_config(common, worldcover_fix=True)


def build_dataloader_config(common: CommonComponents):
    """The control dataloader with flip/rotate augmentation."""
    return _build_dataloader_config(common, augment=True)


def build_train_module_config(common: CommonComponents):
    """The control train module with centered targets."""
    config = _build_train_module_config(common, pixel_targets=True, center_targets=True)
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

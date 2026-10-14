"""Control: ``v1_3_rc_ld1_pix512`` rebuilt on the merged (main-API) code.

``control_pixtgt.py`` with patch-resolution MIM targets (the projection of the whole
token footprint) instead of one drawn pixel per cell. This is the base for the JEPA
arm, since pixel targets require the projection-only target encoder.

Trained originally at 9f5a25b02 as ``v1_3_rc_ld1_pix512`` (W&B project
``20260921_perceiver_shapes``), 300 epochs = 667,200 steps.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_model_config,
    build_train_module_config,
    build_visualize_config,
)
from common import build_trainer_config as _build_trainer_config  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/claude_findings/control_pix512.py"


def build_trainer_config(common: CommonComponents):
    """The rc_ld1 in-loop evals, re-importing THIS module."""
    return _build_trainer_config(common, MODULE_PATH)


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

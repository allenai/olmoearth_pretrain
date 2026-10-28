"""High-res target arm N5: the linear per-cell head + bilinear resize for the 2.5 m targets.

See ``hires_targets_common.py``. Trained as ``v1_3_naip_n5_hires``.
"""

import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hires_targets_common  # noqa: E402
import naip_common  # noqa: E402
from base import build_visualize_config  # noqa: E402

from olmoearth_pretrain.internal.experiment import main  # noqa: E402

MODULE_PATH = "scripts/official/v1_3/naip/n5_hires_targets.py"

build_model_config = partial(
    hires_targets_common.build_model_config, hires_decoder="linear"
)


def build_trainer_config(common):
    """The control's evals, re-importing THIS module."""
    return naip_common.build_trainer_config(common, MODULE_PATH)


def run() -> None:
    """Run the experiment."""
    main(
        common_components_builder=hires_targets_common.build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=hires_targets_common.build_train_module_config,
        dataset_config_builder=hires_targets_common.build_dataset_config,
        dataloader_config_builder=hires_targets_common.build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )


if __name__ == "__main__":
    run()

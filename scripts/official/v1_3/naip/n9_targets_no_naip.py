"""Arm N9: N6 without NAIP -- the v1.3 control plus the Open Buildings and VIIRS targets.

N6's heads and targets (``UpsampleHead`` for 2.5 m Open Buildings, per-pixel VIIRS L1)
on the control's encoded modalities: naip_10 is neither loaded, encoded nor decoded.
N9 vs N6 isolates NAIP given the targets; N9 vs the control isolates the targets.

See ``hires_targets_common.py``. Trained as ``v1_3_naip_n9_targets_nonaip``.
"""

import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hires_targets_common  # noqa: E402
import naip_common  # noqa: E402
from base import build_visualize_config  # noqa: E402

from olmoearth_pretrain.internal.experiment import main  # noqa: E402

# Module-level builders: checkpoint_sweep_evals.py (the in-loop eval jobs) imports this
# module and looks these up by name, falling back to defaults that do not match.
build_common_components = hires_targets_common.build_common_components_no_naip
build_train_module_config = hires_targets_common.build_train_module_config
build_dataloader_config = hires_targets_common.build_dataloader_config
build_dataset_config = hires_targets_common.build_dataset_config

MODULE_PATH = "scripts/official/v1_3/naip/n9_targets_no_naip.py"

build_model_config = partial(
    hires_targets_common.build_model_config, hires_decoder="upsample", naip=False
)


def build_trainer_config(common):
    """The control's evals, re-importing THIS module."""
    return naip_common.build_trainer_config(common, MODULE_PATH)


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

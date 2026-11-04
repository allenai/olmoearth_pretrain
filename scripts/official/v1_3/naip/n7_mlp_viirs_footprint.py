"""Arm N7: N6 + MLP supervision heads + the VIIRS head scored on footprint means.

Two changes on top of N6 (upsample decoder for Open Buildings), combined in one arm
by choice, so N7 vs N6 measures them together:

* MLP heads: every per-cell linear supervision head (WorldCover, CDL, WorldCereal,
  OSM, SRTM, canopy height, VIIRS) becomes Linear -> LayerNorm -> ReLU -> Linear
  (hidden 256, as the mlpgram1 back-projection head), so the targets need not be
  linearly readable from the tokens (cf. SimCLR's nonlinear projection head).
* VIIRS footprint loss: VIIRS pixels are ~460 m, stored nearest-resampled onto the
  10 m grid, so a per-pixel L1 asks the head to paint flat ~46 x 46 px blocks whose
  edges no input can show. The head's 10 m predictions are instead averaged over
  each VIIRS footprint in radiance space and that mean is scored
  (``footprint_loss="log1p"``).

See ``hires_targets_common.py``. Trained as ``v1_3_naip_n7_mlp_viirsfp``.
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
build_common_components = hires_targets_common.build_common_components
build_train_module_config = hires_targets_common.build_train_module_config
build_dataloader_config = hires_targets_common.build_dataloader_config
build_dataset_config = hires_targets_common.build_dataset_config

MODULE_PATH = "scripts/official/v1_3/naip/n7_mlp_viirs_footprint.py"

build_model_config = partial(
    hires_targets_common.build_model_config,
    hires_decoder="upsample",
    viirs_footprint_loss="log1p",
    mlp_heads=True,
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

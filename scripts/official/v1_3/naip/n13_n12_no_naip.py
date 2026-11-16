"""Arm N13: N12 without NAIP -- the base of the architecture arms (N14, N15).

N12 (N7 + linear patch projection) on the control's encoded modalities: naip_10 is
neither loaded, encoded nor decoded, as in N9. So N13 is the v1.3 control plus the Open
Buildings and VIIRS targets (upsample decoder for Open Buildings, VIIRS scored on
footprint means), MLP supervision heads and a linear patch projection, with the
sub-patch offset loss on (``naip_common``). N13 vs N12 isolates NAIP as an input.

The encoder inputs are Sentinel-2, Sentinel-1 and Landsat only, the set a global
embedding run can feed everywhere.

See ``hires_targets_common.py``. Trained as ``v1_3_naip_n13_n12_nonaip``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hires_targets_common  # noqa: E402
import naip_common  # noqa: E402
from base import build_visualize_config  # noqa: E402

from olmoearth_pretrain.data.constants import Modality  # noqa: E402
from olmoearth_pretrain.internal.experiment import main  # noqa: E402

# Module-level builders: checkpoint_sweep_evals.py (the in-loop eval jobs) imports this
# module and looks these up by name, falling back to defaults that do not match.
build_common_components = hires_targets_common.build_common_components_no_naip
build_train_module_config = hires_targets_common.build_train_module_config
build_dataloader_config = hires_targets_common.build_dataloader_config
build_dataset_config = hires_targets_common.build_dataset_config

MODULE_PATH = "scripts/official/v1_3/naip/n13_n12_no_naip.py"

# The modalities the encoder ever sees (the rest of its supported set is decode-only).
ENCODER_INPUT_MODALITIES = [
    Modality.SENTINEL2_L2A.name,
    Modality.SENTINEL1.name,
    Modality.LANDSAT.name,
]


def build_model_config(common):
    """N12's model without naip_10."""
    config = hires_targets_common.build_model_config(
        common,
        hires_decoder="upsample",
        viirs_footprint_loss="log1p",
        mlp_heads=True,
        naip=False,
    )
    config.encoder_config.patch_embed_hidden_sizes = None
    return config


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

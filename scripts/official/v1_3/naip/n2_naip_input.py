"""NAIP arm N2: ``naip_10`` as an ordinary masked encoder input and MIM target.

The v1.3 control (``ablations/rc_ld1_pixtgt_pix512.py``) trained from scratch on
the naip_10 sibling H5 with ``naip_10`` added to the encoder and decoder. naip_10
tokens sit on the S2 token grid (one token = ``4p x 4p`` NAIP pixels at patch size
``p``) and are masked with the rest (random_time_with_decode). With pixel targets the
masked naip_10 token's target is the projection of the 4 x 4 NAIP block under the
cell's drawn S2 pixel. This repeats the 2025-06 "NAIP as input" attempt on the v1.3
recipe; it also provides the checkpoint for the naip_10-slot high-res evals.

W&B project ``2026_10_23_naip_v13``; trained as ``v1_3_naip_n2_input``. See
``naip_common.py``.
"""

import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import naip_common  # noqa: E402
from base import build_visualize_config  # noqa: E402

from olmoearth_pretrain.data.constants import Modality  # noqa: E402
from olmoearth_pretrain.internal.experiment import main  # noqa: E402

MODULE_PATH = "scripts/official/v1_3/naip/n2_naip_input.py"

EXTRA_MODALITIES = [Modality.NAIP_10.name]
ENCODED_MODALITIES = [
    Modality.SENTINEL2_L2A.name,
    Modality.SENTINEL1.name,
    Modality.LANDSAT.name,
    Modality.NAIP_10.name,
    *naip_common.CONTROL_MODALITIES[3:],
]

build_common_components = partial(naip_common.build_common_components, EXTRA_MODALITIES)


def build_model_config(common):
    """The control's model with naip_10 encoded and decoded."""
    return naip_common.build_model_config(common, ENCODED_MODALITIES)


def build_train_module_config(common):
    """The control's train module (pixel targets; naip_10 is masked like S2)."""
    return naip_common.build_train_module_config(common, [])


def build_dataloader_config(common):
    """The control's dataloader (naip_10 counts toward the token budget)."""
    return naip_common.build_dataloader_config(common, [])


def build_trainer_config(common):
    """The control's evals, re-importing THIS module."""
    return naip_common.build_trainer_config(common, MODULE_PATH)


def run() -> None:
    """Run the experiment."""
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=naip_common.build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )


if __name__ == "__main__":
    run()

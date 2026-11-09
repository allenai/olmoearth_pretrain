"""Arm N10: the v1.3 control with MLP supervision heads and nothing else.

The control (``ablations/rc_ld1_pixtgt_pix512.py``, ``v1_3_rc_ld1_pixtgt_pix512``) on
its own H5 and modalities, with every per-cell linear supervision head (WorldCover,
CDL, WorldCereal, OSM, SRTM, canopy height) turned into N7's 2-layer MLP
(``decoder="mlp"``). No NAIP, Open Buildings or VIIRS, so N10 vs the control isolates
the MLP heads; N7 vs N8 measures them on top of the NAIP + hi-res-target arms.
Runs launched after commit 9c99bb01f also get the sub-patch offset loss
(``naip_common.SUBPATCH_OFFSET_LOSS_WEIGHT``); the first N10 run did not.

W&B project ``2026_10_23_naip_v13``; trained as ``v1_3_naip_n10_control_mlp``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ablations"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import naip_common  # noqa: E402
from base import (  # noqa: E402
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_visualize_config,
)
from rc_ld1_pix512 import build_model_config as _control_model_config  # noqa: E402

from olmoearth_pretrain.internal.experiment import main  # noqa: E402

MODULE_PATH = "scripts/official/v1_3/naip/n10_control_mlp_heads.py"


def build_train_module_config(common):
    """The control's train module plus the sub-patch offset loss (no extra modalities)."""
    return naip_common.build_train_module_config(common, [])


def build_model_config(common):
    """The control's model with every linear supervision head an MLP."""
    config = _control_model_config(common)
    assert config.supervision_head_config is not None
    for cfg in config.supervision_head_config.modality_configs.values():
        if cfg.decoder == "linear":
            cfg.decoder = "mlp"
    return config


def build_trainer_config(common):
    """The control's evals, re-importing THIS module, in our W&B project."""
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

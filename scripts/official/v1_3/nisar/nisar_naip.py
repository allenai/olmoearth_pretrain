"""NISAR + NAIP run A: all-token budget (5120 / 380, as tb5120).

See ``nisar_naip_common.py``. No predicted-token cap.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "claude_findings"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import nisar_naip_common  # noqa: E402
from common import build_model_config, build_visualize_config  # noqa: E402

from olmoearth_pretrain.internal.experiment import main  # noqa: E402

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/nisar/nisar_naip.py"
# checkpoint_sweep_evals.py (in-loop evals) looks up a module-level build_common_components.
build_common_components = nisar_naip_common.build_common_components


def build_train_module_config(common):
    """The v1.3 train module with the new masking."""
    return nisar_naip_common.build_train_module_config(common)


def build_dataloader_config(common):
    """tb5120's dataloader with the new masking."""
    return nisar_naip_common.build_dataloader_config(common)


def build_trainer_config(common):
    """The NISAR arms' trainer, re-importing THIS module."""
    return nisar_naip_common.build_trainer_config(common, MODULE_PATH)


def run() -> None:
    """Run the experiment."""
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=nisar_naip_common.build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )


if __name__ == "__main__":
    run()

"""NISAR + NAIP run C with one NISAR slot (card t-grl7q6, favyen 2026-10-31).

Run C (``nisar_naip_c.py``) with the ``nisar_asc`` + ``nisar_desc`` training modalities
replaced by a single ``nisar`` modality. ``nisar`` is built at load time from the stored
passes (``OlmoEarthDataset.read_h5_file``): for each month, one of the passes present,
uniformly at random when both are (fresh per sample load), as Sentinel-1's monthly
layers take whichever orbit is available. Its month mask is the union of the passes'.
Normalized with the passes' pooled computed stats (``norm_configs/computed.json``).

Everything else (data mix, shapes, encoder/decode budgets, masking, ``MAX_NUM_VISIBLE``,
loss, steps, in-loop evals) is run C's. With one NISAR slot a fully present sample has
5 encode-decode band sets instead of 6.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "claude_findings"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import nisar_naip_c  # noqa: E402
import nisar_naip_common  # noqa: E402
from common import build_model_config, build_visualize_config  # noqa: E402
from nisar_common import NISAR_MODALITIES  # noqa: E402

from olmoearth_pretrain.data.constants import Modality  # noqa: E402
from olmoearth_pretrain.internal.experiment import (  # noqa: E402
    CommonComponents,
    SubCmd,
    main,
)

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/nisar/nisar_naip_c_one_slot.py"


def build_common_components(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """Run C's modalities with nisar_asc + nisar_desc replaced by one ``nisar`` slot."""
    config = nisar_naip_common.build_common_components(
        script, cmd, run_name, cluster, overrides
    )
    modalities = list(config.training_modalities)
    first = modalities.index(NISAR_MODALITIES[0])
    modalities = [m for m in modalities if m not in NISAR_MODALITIES]
    modalities.insert(first, Modality.NISAR.name)
    config.training_modalities = modalities
    return config


def build_trainer_config(common):
    """Run C's trainer, re-importing THIS module for the in-loop evals."""
    return nisar_naip_common.build_trainer_config(common, MODULE_PATH)


def run() -> None:
    """Run the experiment."""
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=nisar_naip_c.build_train_module_config,
        dataset_config_builder=nisar_naip_common.build_dataset_config,
        dataloader_config_builder=nisar_naip_c.build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )


if __name__ == "__main__":
    run()

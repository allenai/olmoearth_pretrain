"""NISAR + NAIP run B (2026-10-28 redesign): at most 3-4 input modalities per sample, v1.3 budget.

Run A (``nisar_naip.py``) with a different shape/modality sampler:

* every rank batch draws ``k ~ U{3, 4}``; each sample keeps a uniform random subset of at
  most ``k`` of its present encode-decode modalities (S2, S1, Landsat, NISAR asc/desc,
  NAIP; decode-only maps are always kept) and the rest are dropped as if absent;
* the crop is sized for ``k`` modalities under the v1.3 control budget:
  ``k * hw^2 * t <= 3072`` (floor 228), i.e. the control's shapes at ``k = 3``;
* masking as in run A: ``e ~ U{1..n-1}`` of the ``n`` kept band sets are visible (so 1-3
  visible, 1-3 hidden) and the visible band sets' non-encoded tokens are predicted.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "claude_findings"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import nisar_naip_common  # noqa: E402
from common import MIN_TOKENS_PER_INSTANCE, TOKEN_BUDGET  # noqa: E402
from common import build_model_config, build_visualize_config  # noqa: E402

from olmoearth_pretrain.internal.experiment import main  # noqa: E402

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/nisar/nisar_naip_cap4.py"
NUM_KEPT_MODALITIES_CHOICES = [3, 4]
build_common_components = nisar_naip_common.build_common_components


def build_train_module_config(common):
    """The v1.3 train module with the new masking."""
    return nisar_naip_common.build_train_module_config(common)


def build_dataloader_config(common):
    """v1.3 budget (3072 / 228) over k ~ U{3, 4} kept modalities per rank batch."""
    config = nisar_naip_common.build_dataloader_config(common)
    config.token_budget = TOKEN_BUDGET
    config.min_tokens_per_instance = MIN_TOKENS_PER_INSTANCE
    config.num_kept_modalities_choices = NUM_KEPT_MODALITIES_CHOICES
    return config


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

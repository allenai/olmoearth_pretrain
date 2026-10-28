"""NISAR + NAIP run B (2026-10-28 redesign): at most 3-4 input modalities per sample, v1.3 budget.

Run A (``nisar_naip.py``) with a different shape/modality sampler:

* every rank batch draws ``k ~ U{3, 4}``; each sample keeps a uniform random subset of at
  most ``k`` of its present encode-decode modalities (S2, S1, Landsat, NISAR asc/desc,
  NAIP; decode-only maps are always kept) and the rest are dropped as if absent;
* the crop is sized for ``k`` modalities under the v1.3 control budget. The control's
  dataloader counts default band sets (S2 3 + S1 1 + Landsat 2 = 6 per ``hw^2 * t``), so
  its 3072 / 228 means ``hw^2 * t <= 512`` (>= 38), i.e. 1536 / 114 tokens as the model
  tokenizes them (one band set each). Here ``k * hw^2 * t <= 1536`` (>= 114): the
  control's shapes at ``k = 3`` and the control's token count at ``k = 4``;
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
# The v1.3 budget in model tokens (the control charges 6 default band sets for its 3
# one-band-set modalities).
CONTROL_DEFAULT_BANDSETS = 6
CONTROL_MODALITIES = 3
KEPT_TOKEN_BUDGET = TOKEN_BUDGET * CONTROL_MODALITIES // CONTROL_DEFAULT_BANDSETS  # 1536
KEPT_MIN_TOKENS = (
    MIN_TOKENS_PER_INSTANCE * CONTROL_MODALITIES // CONTROL_DEFAULT_BANDSETS
)  # 114
build_common_components = nisar_naip_common.build_common_components


def build_train_module_config(common):
    """The v1.3 train module with the new masking."""
    return nisar_naip_common.build_train_module_config(common)


def build_dataloader_config(common):
    """The v1.3 budget (1536 / 114 model tokens) over k ~ U{3, 4} kept modalities."""
    config = nisar_naip_common.build_dataloader_config(common)
    config.token_budget = KEPT_TOKEN_BUDGET
    config.min_tokens_per_instance = KEPT_MIN_TOKENS
    config.num_kept_modalities_choices = NUM_KEPT_MODALITIES_CHOICES
    # Band sets as the model tokenizes them (S2 / Landsat as one band set each).
    config.tokenization_config = common.tokenization_config
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

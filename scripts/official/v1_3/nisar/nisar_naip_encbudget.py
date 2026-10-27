"""NISAR + NAIP run B: budget the online-encoder tokens, not all tokens.

Run A (``nisar_naip.py``) with a different shape sampler:

* every rank batch draws a visible band-set count ``e ~ U{1..5}`` (6 encode-decode
  band sets) and its (grid, t) satisfies ``e * hw^2 * ceil(t / 2) <= 1280``, so fewer
  visible modalities give larger crops (e=2: hw^2*t <= 1280; e=3: <= 853; the control's
  all-token budget allows 1024 at its largest);
* the masking uses that ``e`` for every sample with more than ``e`` band sets; a sample
  with ``n <= e`` draws its own ``e ~ U{1..n-1}`` (uniform per sample, below budget);
* the encoder floor is the control's 228 scaled the same way (228 * 1280 / 3072 = 95);
* at most 2560 predicted tokens per sample (run A's maximum: half of 5120); a random
  subset is kept beyond that.
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

MODULE_PATH = "scripts/official/v1_3/nisar/nisar_naip_encbudget.py"
ENCODER_TOKEN_BUDGET = 1280
MIN_ENCODER_TOKENS = 95
MAX_DECODE_TOKENS = 2560
build_common_components = nisar_naip_common.build_common_components


def build_train_module_config(common):
    """The v1.3 train module with the new masking and the predicted-token cap."""
    return nisar_naip_common.build_train_module_config(
        common, max_decode_tokens=MAX_DECODE_TOKENS
    )


def build_dataloader_config(common):
    """Shapes from the online-encoder budget; one visible count per rank batch."""
    config = nisar_naip_common.build_dataloader_config(
        common, max_decode_tokens=MAX_DECODE_TOKENS
    )
    config.encoder_token_budget = ENCODER_TOKEN_BUDGET
    config.min_encoder_tokens = MIN_ENCODER_TOKENS
    config.max_num_visible = nisar_naip_common.NUM_ENCODE_DECODE_BANDSETS - 1
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

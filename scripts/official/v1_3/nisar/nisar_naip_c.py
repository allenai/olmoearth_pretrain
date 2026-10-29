"""NISAR + NAIP run C (2026-10-28): encoder budget + decode budget, v1.3 speed.

Run A (``nisar_naip.py``) with a different shape sampler and masking:

* every rank batch draws ``e ~ U{1..4}`` visible band sets (a sample with ``n <= e``
  present encode-decode band sets draws its own ``e ~ U{1..n-1}``); all other present
  encode-decode band sets are hidden;
* the crop is sized by the encoder budget ``e * hw^2 * t / 2 <= 512`` (>= 38): at
  ``e = 2`` exactly the v1.3 control's shapes (``hw^2 * t`` in [38, 512]), larger crops
  at ``e = 1`` (up to 1024) and smaller ones at ``e = 3, 4``; the grid list is v1.3's;
* half of each visible band set's tokens are encoded and the rest predicted; every
  hidden band-set token is a decode candidate; at most ``MAX_DECODE_TOKENS`` predicted
  encode-decode tokens per sample (uniform subset, the rest target-only), and every
  decode-only map token is predicted on top, as in v1.3;
* the patch-discrimination loss truncates its score matrices to the decoder tokens
  (exact; avoids the ``T x T`` cost of the larger crops);
* every rank draws the same shape per step (``sync_shapes_across_ranks``), so a step
  does not wait on whichever rank drew the most expensive shape;
* naip_10's patch embedding recomputes in backward (exact): at ``e = 1`` and
  ``hw = 32`` its per-pixel features are 4 GiB per activation at microbatch 32, which
  ran out of memory at step 862 of the first launch.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "claude_findings"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import nisar_naip_common  # noqa: E402
from common import build_model_config as build_base_model_config  # noqa: E402
from common import build_visualize_config  # noqa: E402

from olmoearth_pretrain.internal.experiment import main  # noqa: E402

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/nisar/nisar_naip_c.py"
# The v1.3 control's largest encoder count: 2 of 3 band sets visible, half of hw^2 t <= 512.
ENCODER_TOKEN_BUDGET = 512
MIN_ENCODER_TOKENS = 38  # = the control's hw^2 t >= 38 at e = 2
MAX_NUM_VISIBLE = 4
# Predicted encode-decode tokens per sample (maps excluded).
MAX_DECODE_TOKENS = 1024
build_common_components = nisar_naip_common.build_common_components


def _set_masking(masking_config) -> None:
    nisar_naip_common.set_masking(masking_config, MAX_DECODE_TOKENS)
    masking_config.strategy_config["decode_cap_skips_decode_only"] = True


def build_model_config(common):
    """The v1.3 model, with naip_10's patch embedding checkpointed."""
    config = build_base_model_config(common)
    config.encoder_config.checkpoint_patch_embed_modalities = ["naip_10"]
    return config


def build_train_module_config(common):
    """The v1.3 train module, new masking, decode cap, truncated patch-disc loss."""
    config = nisar_naip_common.build_train_module_config(common)
    _set_masking(config.masking_config)
    config.loss_config.loss_config["truncate_to_decoder"] = True
    return config


def build_dataloader_config(common):
    """Shapes from the encoder budget (t / 2, no ceil), one visible count per rank batch."""
    config = nisar_naip_common.build_dataloader_config(common)
    _set_masking(config.masking_config)
    config.encoder_token_budget = ENCODER_TOKEN_BUDGET
    config.min_encoder_tokens = MIN_ENCODER_TOKENS
    config.max_num_visible = MAX_NUM_VISIBLE
    config.encoder_budget_half_t = True
    config.sync_shapes_across_ranks = True
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

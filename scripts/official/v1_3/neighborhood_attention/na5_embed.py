"""Run 2: an embedding-focused 5x5 neighborhood-attention run (from scratch).

Built on ``na5.py`` (the v1.3 control + 5x5 NA in the encoder, pooler and decoder),
with the data shaped like production embedding inference:

* patch size 2 always, S2 + S1 + Landsat, exactly 12 timesteps;
* square inputs of 16 / 24 / 32 px (8 / 12 / 16 patches), equal weights, one size per
  step on every rank;
* masking grows with the input so the step costs about 1.9x a v1.3 step: v1.3's
  ``random_time_with_decode`` (2 of 3 band sets encode-only) with the within-band-set
  visible share and the decode share scheduled by size (``MASK_SCHEDULES``; v1.3 is
  50% / 50% at every size);
* a 1024-query location budget, so one location query per pixel is possible at every
  size (production embeds at location-query patch size 1);
* smaller rank microbatches for the big sizes (gradient accumulation; same 512-instance
  global batch).

48 px was dropped (favyen, 2026-11-05): with 2:2:2:1 a step measured ~2.7x v1.3, mostly
the target encoder on all 20,736 tokens of a 48 px input, which masking cannot reduce.

``NA_EMBED_SCHEDULE`` (env, default "A") picks the schedule; it only affects the data,
so the in-loop eval jobs (which rebuild the model from this module) are unaffected.
Card t-ktw5up.
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "claude_findings"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import (  # noqa: E402
    build_common_components,
    build_dataset_config,
    build_visualize_config,
)
from common import build_dataloader_config as _build_dataloader_config  # noqa: E402
from common import build_trainer_config as _build_trainer_config  # noqa: E402
from na5 import WANDB_PROJECT, build_model_config  # noqa: E402
from na5 import build_train_module_config as _build_train_module_config  # noqa: E402
from v1_2.base import ONLY_DECODE_MODALITIES  # noqa: E402

from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig  # noqa: E402
from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.train.masking import MaskingConfig  # noqa: E402
from olmoearth_pretrain.train.train_module.latent_mim import (  # noqa: E402
    LatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/neighborhood_attention/na5_embed.py"

PATCH_SIZE = 2
NUM_TIMESTEPS = 12
# Grid sides in patches (16 / 24 / 32 px at ps2) and their sampling weights.
SAMPLED_HW_P_LIST = [8, 12, 16]
SAMPLED_HW_P_WEIGHTS = [1.0, 1.0, 1.0]
# Crop budget: 3 band sets x 16^2 x 12 = the largest shape, so every shape keeps all 12
# timesteps (the dataset crops spatially only when a budget is set).
TOKEN_BUDGET = 3 * 16 * 16 * NUM_TIMESTEPS
MAX_LOCATION_QUERIES = 32 * 32
# [hw, visible share of an encode band set, decoded share of a decode band set].
MASK_SCHEDULES = {
    # ~1.9x a v1.3 step (measured per-size costs): 50 / 65 / 75 % masked. Chosen.
    "A": [[8, 0.50, 0.50], [12, 0.35, 0.40], [16, 0.25, 0.30]],
    # ~2.0x: 25 / 60 / 80 % masked.
    "B": [[8, 0.75, 0.50], [12, 0.40, 0.40], [16, 0.20, 0.25]],
}
MASK_SCHEDULE = os.environ.get("NA_EMBED_SCHEDULE", "A")
# Rank microbatch (instances) per grid size, for memory; the global batch stays 512.
RANK_MICROBATCH_SIZE_BY_HW = [[8, 64], [12, 32], [16, 32]]


def _masking_config(common: CommonComponents) -> MaskingConfig:
    return MaskingConfig(
        strategy_config={
            "type": "size_scheduled_random_time_with_decode",
            "schedule": MASK_SCHEDULES[MASK_SCHEDULE],
            "random_ratio": 0.5,
            "bandset_encode_ratio": 0.5,
            "only_decode_modalities": ONLY_DECODE_MODALITIES,
        },
        tokenization_config=common.tokenization_config,
    )


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """Fixed ps2, 12 timesteps, 16-32 px inputs, size-scheduled masking."""
    config = _build_dataloader_config(common)
    config.fixed_patch_size = PATCH_SIZE
    # One shape per step on every rank: a step costs the slowest rank's batch, so
    # independent draws would make most steps cost the largest size.
    config.sync_shape_across_ranks = True
    config.sampled_hw_p_list = list(SAMPLED_HW_P_LIST)
    config.sampled_hw_p_weights = list(SAMPLED_HW_P_WEIGHTS)
    config.fixed_timesteps = NUM_TIMESTEPS
    config.token_budget = TOKEN_BUDGET
    config.min_tokens_per_instance = 0
    config.max_location_queries = MAX_LOCATION_QUERIES
    config.masking_config = _masking_config(common)
    return config


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """na5's train module with size-dependent microbatches and the scheduled masking."""
    config = _build_train_module_config(common)
    config.masking_config = _masking_config(common)
    config.rank_microbatch_size_by_hw = [list(x) for x in RANK_MICROBATCH_SIZE_BY_HW]
    return config


def build_trainer_config(common: CommonComponents):
    """The control's in-loop evals, re-importing THIS module."""
    trainer_config = _build_trainer_config(common, MODULE_PATH)
    trainer_config.callbacks["wandb"].project = WANDB_PROJECT
    return trainer_config


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

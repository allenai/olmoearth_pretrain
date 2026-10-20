"""Config D' : config D with a narrow front end at v1.3's real-token budget.

Same model as D (``allcap_slot_n4d192``: 4 x d192 within-cell narrow stage, <= 24
slots per cell, 12 x d768 trunk), except:

- **Narrow front end** (``SlotMergeConfig.narrow_front_end``): the patch embedding and
  composite encodings produce d192 tokens directly and there is no encoder-width skip,
  so everything before the merge runs at the narrow width.
- **v1.3's real-token budget:** ``TOKEN_BUDGET`` (1536 real tokens, what v1.3's 3072
  band-set budget allows), so D' draws the same grids and timestep counts as v1.3.
- **Microbatch 64** (v1.3's; D: 32).
- **Compiled mixed-RoPE rotation** (``encodings.use_compiled_mixed_rope``): -19% step
  time on D (1-GPU profile, 2026-10-19).

W&B project ``20261016_allcap_slot_narrow``; trained as ``v1_3_allcap_slot_n4d192_fe``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from allcap_narrow import (  # noqa: E402
    build_allcap_trainer_config,
    build_common_components,
    build_dataset_config,
    build_narrow_model_config,
)
from allcap_narrow import build_dataloader_config as _allcap_dataloader  # noqa: E402
from allcap_narrow import (  # noqa: E402
    build_train_module_config as _allcap_train_module,
)
from base import build_visualize_config  # noqa: E402

from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig  # noqa: E402
from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.encodings import use_compiled_mixed_rope  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402

MODULE_PATH = "scripts/official/v1_3/ablations/allcap_slot_n4d192_fe.py"
NARROW_DEPTH = 4
NARROW_DIM = 192
# Real input tokens per sample, matched to v1.3. v1.3's loader budgets 3072 in default
# band sets (no tokenization_config: S2 = 3, Landsat = 2, S1 = 1, so 6 per cell and
# timestep) while its model makes 3 tokens there (S2 and Landsat are one band set
# each), i.e. <= 1536 real tokens and grids <= 22 cells. This loader counts real tokens,
# so 3072 here drew up to 2x v1.3's tokens and 32x32 grids that OOMed at mb64
# (2026-10-20).
TOKEN_BUDGET = 1536
# v1.3's microbatch: the whole rank batch in one pass. Fits at 1536 tokens (peak 71 GiB
# on the heaviest shape, 1-GPU profile 2026-10-20); D's 32 was for its 2x budget.
RANK_MICROBATCH_SIZE = 64


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """D's model with the narrow front end and compiled RoPE."""
    use_compiled_mixed_rope(True)
    config = build_narrow_model_config(common, NARROW_DEPTH, NARROW_DIM)
    slot_merge = config.encoder_config.slot_merge_config
    assert slot_merge is not None
    slot_merge.narrow_front_end = True
    return config


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """D's sampler at v1.3's real-token budget."""
    config = _allcap_dataloader(common)
    config.token_budget = TOKEN_BUDGET
    return config


def build_train_module_config(common: CommonComponents):
    """D's train module at v1.3's microbatch."""
    config = _allcap_train_module(common)
    config.rank_microbatch_size = RANK_MICROBATCH_SIZE
    return config


def build_trainer_config(common: CommonComponents):
    """The allcap eval set, re-importing THIS module."""
    return build_allcap_trainer_config(common, MODULE_PATH)


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

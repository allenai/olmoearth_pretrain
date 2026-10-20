"""Config D' : config D with a narrow front end and a budget on merged tokens.

Same model as D (``allcap_slot_n4d192``: 4 x d192 within-cell narrow stage, <= 24
slots per cell, 12 x d768 trunk), except:

- **Narrow front end** (``SlotMergeConfig.narrow_front_end``): the patch embedding and
  composite encodings produce d192 tokens directly and there is no encoder-width skip,
  so everything before the merge runs at the narrow width.
- **v1.3's raw-token budget:** ``INPUT_TOKEN_BUDGET`` (1536 real tokens, what v1.3's
  3072 band-set budget allows) caps the raw input tokens and the shape sampler, so D'
  draws the same grids and timestep counts as v1.3.
- **Compiled mixed-RoPE rotation** (``encodings.use_compiled_mixed_rope``): -19% step
  time on D (1-GPU profile, 2026-10-19).
- **Recomputed MLP activations** (``attention.use_recompute_mlp_activation``): the GELU
  output is recomputed in backward instead of stored.
- **Decoder cap 1100** tokens per sample (D: 2048), about v1.3's mean.

W&B project ``20261016_allcap_slot_narrow``; trained as ``v1_3_allcap_slot_n4d192_fe``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from allcap_narrow import (  # noqa: E402
    SLOT_CAP,
    SLOT_MIN,
    SLOT_RATIO,
    build_allcap_trainer_config,
    build_common_components,
    build_dataset_config,
    build_narrow_model_config,
    build_train_module_config,
)
from allcap_narrow import build_dataloader_config as _allcap_dataloader  # noqa: E402
from base import build_visualize_config  # noqa: E402

from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig  # noqa: E402
from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.attention import use_recompute_mlp_activation  # noqa: E402
from olmoearth_pretrain.nn.encodings import use_compiled_mixed_rope  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402

MODULE_PATH = "scripts/official/v1_3/ablations/allcap_slot_n4d192_fe.py"
NARROW_DEPTH = 4
NARROW_DIM = 192
# Raw input tokens per sample, matched to v1.3. v1.3's loader budgets 3072 in default
# band sets (no tokenization_config: S2 = 3, Landsat = 2, S1 = 1, so 6 per cell and
# timestep) while its model makes 3 tokens there (S2 and Landsat are one band set
# each), i.e. <= 1536 real tokens and grids <= 22 cells. This loader counts real tokens,
# so 3072 here drew up to 2x v1.3's tokens and 32x32 grids that OOMed at mb64
# (2026-10-20). Merged tokens <= raw tokens, so the merged budget no longer binds.
INPUT_TOKEN_BUDGET = 1536
# Decoder tokens per sample: ~v1.3's mean (~1.1k) rather than its p90 (D's 2048). The
# loss's per-sample score matrices scale with its square; at 2048, mb64 still OOMed on
# rare heavy batches (2026-10-20, step ~206).
MAX_DECODE_TOKENS = 1100


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """D's model with the narrow front end; compiled RoPE and recomputed MLP GELUs."""
    use_compiled_mixed_rope(True)
    # The MLP's GELU output was the 2nd-largest activation (6.7 GiB of ~43 at mb32,
    # 2026-10-20 snapshot); recompute it in backward instead (same values).
    use_recompute_mlp_activation(True)
    config = build_narrow_model_config(common, NARROW_DEPTH, NARROW_DIM)
    slot_merge = config.encoder_config.slot_merge_config
    assert slot_merge is not None
    slot_merge.narrow_front_end = True
    return config


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """D's sampler, budgeting the merged tokens and capping the input tokens."""
    config = _allcap_dataloader(common)
    config.merged_slots = [SLOT_RATIO, SLOT_MIN, SLOT_CAP]
    # The shape sampler (grid, timesteps) reads token_budget at the real-token rates,
    # so it gets the raw cap too.
    config.token_budget = INPUT_TOKEN_BUDGET
    config.input_token_budget = INPUT_TOKEN_BUDGET
    config.max_decode_tokens = MAX_DECODE_TOKENS
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

"""Config D' : config D with a narrow front end and a budget on merged tokens.

Same model as D (``allcap_slot_n4d192``: 4 x d192 within-cell narrow stage, <= 24
slots per cell, 12 x d768 trunk), except:

- **Narrow front end** (``SlotMergeConfig.narrow_front_end``): the patch embedding and
  composite encodings produce d192 tokens directly and there is no encoder-width skip,
  so everything before the merge runs at the narrow width.
- **Budget on the merged tokens:** ``TOKEN_BUDGET`` (v1.3's 3072) caps the tokens the
  encoder-width trunk sees after the per-cell merge, and ``INPUT_TOKEN_BUDGET`` caps the
  raw input tokens separately. Training cells then hold up to a year of captures, the
  regime an embedding run sees (with D's input budget, ~99.5% of training cells held
  <= 32 tokens).
- **Compiled mixed-RoPE rotation** (``encodings.use_compiled_mixed_rope``): -19% step
  time on D (1-GPU profile, 2026-10-19).

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
from olmoearth_pretrain.nn.encodings import use_compiled_mixed_rope  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402

MODULE_PATH = "scripts/official/v1_3/ablations/allcap_slot_n4d192_fe.py"
NARROW_DEPTH = 4
NARROW_DIM = 192
# Raw input tokens per sample; the trunk still sees <= allcap_narrow.TOKEN_BUDGET
# merged tokens. Set from the 1-GPU memory/speed sweep before training.
INPUT_TOKEN_BUDGET = 12288


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """D's model with the narrow front end; compiled RoPE for this process."""
    use_compiled_mixed_rope(True)
    config = build_narrow_model_config(common, NARROW_DEPTH, NARROW_DIM)
    slot_merge = config.encoder_config.slot_merge_config
    assert slot_merge is not None
    slot_merge.narrow_front_end = True
    return config


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """D's sampler, budgeting the merged tokens and capping the input tokens."""
    config = _allcap_dataloader(common)
    config.merged_slots = [SLOT_RATIO, SLOT_MIN, SLOT_CAP]
    config.input_token_budget = INPUT_TOKEN_BUDGET
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

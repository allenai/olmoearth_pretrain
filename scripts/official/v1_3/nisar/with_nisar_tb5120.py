"""NISAR from-scratch arm with the token budget scaled for the two extra modalities.

``with_nisar.py`` keeps the control's per-instance token budget (3072) and floor (228).
The shape sampler charges every encoded spacetime band set against that budget, so
adding ``nisar_asc`` + ``nisar_desc`` (1 band set each) to S2 + S1 + Landsat (1 each)
raises the cost of a (grid, t) shape from 3*hw^2*t to 5*hw^2*t, and with_nisar draws
smaller grids / fewer timesteps than the control.

Here the budget and floor are scaled by 5/3 (3072 -> 5120, 228 -> 380). The sampler's
max_t/min_t windows are then exactly those of the control (see
OlmoEarthDataLoader's max_t_for/min_t_for), so S2/S1/Landsat see the same shapes as
in no_nisar and NISAR tokens come on top. MISSING tokens (NISAR covers ~3-4 of 12
months, half the samples) are removed before the encoder, so the extra compute is
only the NISAR months actually present.

Memory: the patch-discrimination loss builds a (batch, T, T) score matrix over each
sample's decoder tokens, so 5/3 more tokens per sample is ~2.8x more memory there. At
rank_microbatch_size 64 (no accumulation: 512 / 8 GPUs) the first launch OOMed at
step ~9 (72 GiB allocated). This arm uses rank microbatch 32 (2 microbatches per step).
The global batch is unchanged and the loss is per sample and divided by the number of
microbatches (pred2unit off), so the optimization is the same.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "claude_findings"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import MIN_TOKENS_PER_INSTANCE, TOKEN_BUDGET  # noqa: E402
from common import build_dataloader_config as _build_dataloader_config  # noqa: E402
from nisar_common import (  # noqa: E402
    build_common_components_fn,
    build_dataset_config,
)
from nisar_common import build_trainer_config as _build_trainer_config  # noqa: E402
from with_nisar import (  # noqa: E402
    WITH_NISAR,
    build_model_config,
    build_visualize_config,
)
from with_nisar import build_train_module_config as _build_train_module_config  # noqa: E402

from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig  # noqa: E402
from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.train.train_module.latent_mim import (  # noqa: E402
    LatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/nisar/with_nisar_tb5120.py"
# Encoded spacetime band sets: control S2 + S1 + Landsat = 3; + NISAR asc/desc = 5.
CONTROL_BANDSETS = 3
NISAR_BANDSETS = 5
assert (TOKEN_BUDGET * NISAR_BANDSETS) % CONTROL_BANDSETS == 0
assert (MIN_TOKENS_PER_INSTANCE * NISAR_BANDSETS) % CONTROL_BANDSETS == 0
SCALED_TOKEN_BUDGET = TOKEN_BUDGET * NISAR_BANDSETS // CONTROL_BANDSETS  # 5120
SCALED_MIN_TOKENS = MIN_TOKENS_PER_INSTANCE * NISAR_BANDSETS // CONTROL_BANDSETS  # 380
RANK_MICROBATCH_SIZE = 32


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """The control dataloader with budget and floor scaled by 5/3."""
    config = _build_dataloader_config(common)
    config.token_budget = SCALED_TOKEN_BUDGET
    config.min_tokens_per_instance = SCALED_MIN_TOKENS
    return config


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """with_nisar's train module with rank microbatch 32 (see module docstring)."""
    config = _build_train_module_config(common)
    config.rank_microbatch_size = RANK_MICROBATCH_SIZE
    return config


def build_trainer_config(common: CommonComponents):
    """The v1.3 control in-loop evals, re-importing THIS module."""
    return _build_trainer_config(common, MODULE_PATH)


def run() -> None:
    """Run the experiment."""
    main(
        common_components_builder=build_common_components_fn(WITH_NISAR),
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )


if __name__ == "__main__":
    run()

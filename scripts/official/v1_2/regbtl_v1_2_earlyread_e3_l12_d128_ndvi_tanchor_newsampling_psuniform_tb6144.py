"""EARLY READ + DOUBLED TOKEN BUDGET: 3-layer trunk, 12 bottleneck blocks, budget 6144.

``regbtl_v1_2_earlyread_e3_l12_d128_ndvi_tanchor_newsampling_psuniform`` with
``token_budget`` raised from 3072 to 6144 (``apply_shape_sweep``, ``temporal_bias`` held
at the recipe's 2.75 so the budget is the only axis that moves). Its control is that
3072 arm, so the budget effect is isolated; the depth split is controlled by the same
arm against the frontier.

WHY THIS ARM EXISTS: it is the configuration where doubling the data per instance is
free. Measured encoder MACs at each budget's largest full-year shape, relative to the
cand_ndvi frontier at 3072:

    budget  shape        base     e3_l12    e3_l35
      3072  hw=9,t=12    1.00x     0.37x     0.57x
      6144  hw=13,t=12   2.92x     0.99x     1.47x
      9216  hw=16,t=12   5.68x     1.85x     2.64x

e3_l12 at 6144 costs what the frontier costs at 3072. The early read is what makes the
budget increase affordable at all -- the base at 6144 is 2.92x.

WHAT THE BUDGET BUYS: ``max_sequence_length=12`` caps t, and at 3072 that cap is already
saturated for hw<=9. So this does not lift the ceiling; it raises the grid size at which
a full-year sequence is reachable, hw<=9 -> hw<=13. At 3072 most of the sampled grid
distribution is timestep-starved (hw=12 -> t<=7, hw=16 -> t<=4, hw>=24 -> t=1); at 6144
that becomes hw=13 -> t=12, hw=16 -> t=8, hw=32 -> t=2. The gain is concentrated in the
hw=10..16 band, which is where the frozen ws16 ps=1 eval shape sits.

TWO CAVEATS ON THE COST NUMBERS. They are encoder-only: more tokens means more mask
tokens for the depth-4 decoder, which the early read does not touch, so the real step
ratio sits above 0.99x. And they are computed at the LARGEST shape each budget allows --
``MIN_TOKENS_PER_INSTANCE`` (228) does not scale with the budget, so raising it widens
the spread of per-instance cost rather than doubling the mean, making these an upper
bound on the average step rather than the expected value.

MEMORY, NOT COMPUTE, IS THE LIKELY CONSTRAINT: ``rank_microbatch_size`` is dropped to 32
here (the recipe's own note records micro 32 @ 6144 OOMing *before* the broadcastable
key-mask fix). Microbatch affects only memory -- not tokens/step, not the loss, not the
LR schedule. This arm should have real headroom regardless: a read block stores ~3*N*d of
activations against a trunk layer's ~12-15*N*d, so 3+12 carries roughly half the base's
activation memory. If it runs comfortably at 32, 64 is worth trying.

Duration is untouched: ``epochs(300)`` is a fixed instance count, so this runs the same
662,700 steps as every other arm and differs only in tokens per step.
"""

import logging

from base import build_trainer_config as _base_build_trainer_config
from regbtl_v1_2_earlyread_common import (
    build_budget_dataloader_config,
    build_budget_train_module_config,
    build_earlyread_model_config,
    set_earlyread_loop_evals,
)
from regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform import (
    build_dataset_config,
)
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_common_components,
    build_visualize_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)

TRUNK_DEPTH = 3
LATENT_DEPTH = 12
TOKEN_BUDGET = 6144
RANK_MICROBATCH_SIZE = 32
MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_earlyread_e3_l12_d128_ndvi_tanchor_newsampling_psuniform_tb6144.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Identical to the 3072 arm: 3-layer trunk + 12 interleaved read/latent blocks."""
    return build_earlyread_model_config(
        common, trunk_depth=TRUNK_DEPTH, latent_depth=LATENT_DEPTH
    )


def build_dataloader_config(common: CommonComponents):
    """The ndvi newsampling/psuniform dataloader at token_budget 6144."""
    return build_budget_dataloader_config(common, TOKEN_BUDGET)


def build_train_module_config(common: CommonComponents):
    """ndvi-aware faster train module at the halved microbatch the larger budget needs."""
    return build_budget_train_module_config(common, RANK_MICROBATCH_SIZE)


def build_trainer_config(common: CommonComponents):
    """Base trainer + ONLY the S1+S2+Landsat embedding evals (pastis/ethiopia/descals)."""
    return set_earlyread_loop_evals(_base_build_trainer_config(common), MODULE_PATH)


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

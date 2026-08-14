"""EARLY READ + DOUBLED TOKEN BUDGET: 3-layer trunk, 35 bottleneck blocks, budget 6144.

``regbtl_v1_2_earlyread_e3_l35_d128_ndvi_tanchor_newsampling_psuniform`` at
``token_budget=6144``. The param-matched partner to
``..._e3_l12_..._tb6144``: together the four trunk-3 arms form a 2x2 over
{fewer blocks, param-matched} x {budget 3072, budget 6144}, each cell controlled on the
other axis.

    budget  shape        base     e3_l12    e3_l35
      3072  hw=9,t=12    1.00x     0.37x     0.57x
      6144  hw=13,t=12   2.92x     0.99x     1.47x

At 1.47x the frontier's current step this is the most expensive arm in the sweep, and it
is the one to watch for the two failure modes the cheap arm does not have.

MEMORY IS THE REAL RISK HERE, not compute. A read block stores ~3*N*d of activations (its
input norm plus a k and a v projection over the FULL token array) against a trunk layer's
~12-15*N*d, so 3+35 carries roughly 12 "trunk-layer units" against the base's 13 -- i.e.
almost no saving, unlike 3+12 at roughly half. Doubling N on top of that is the tightest
memory configuration in the sweep. ``rank_microbatch_size`` is 32 here (as in the e3_l12
tb6144 arm); if it OOMs, drop to 16 -- microbatch affects only memory, not tokens/step,
the loss, or the LR schedule.

That same accounting is why the K/V hoist matters most for this arm: ~73% of a read
block's cost is re-projecting the whole token array into K/V, once per read, 35 times.
Computing it once and sharing across reads would cut both compute and activation memory
sharply -- at the price of ``pdproj``'s per-read lens, which was worth ~1.3 pts at 4
reads. Not attempted here; noted as the obvious follow-up if this arm is memory-bound.

Caveats on the cost numbers are the same as the e3_l12 tb6144 arm's: encoder-only (the
depth-4 decoder also grows with token count), and computed at the largest shape each
budget allows, which is an upper bound on the average step because
``MIN_TOKENS_PER_INSTANCE`` (228) does not scale with the budget.
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
LATENT_DEPTH = 35
TOKEN_BUDGET = 6144
RANK_MICROBATCH_SIZE = 32
MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_earlyread_e3_l35_d128_ndvi_tanchor_newsampling_psuniform_tb6144.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Identical to the 3072 arm: 3-layer trunk + 35 interleaved read/latent blocks."""
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

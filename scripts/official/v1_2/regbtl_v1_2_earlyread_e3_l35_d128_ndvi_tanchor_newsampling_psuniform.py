"""EARLY READ, param-matched: 3-layer trunk + 35 bottleneck blocks.

The capacity control for ``regbtl_v1_2_earlyread_e3_l12_d128_ndvi_tanchor_newsampling_psuniform``.
Same base, same 3-layer trunk, but ``register_latent_depth`` raised from 12 to 35 so the
encoder+bottleneck parameter count matches the 12+4 A/B partner. If the plain e3_l12 arm
loses to the base, this separates "reading at layer 3 costs us something" from "we shipped
a 0.61x-sized model".

MEASURED on one full S1+S2+Landsat window at ws16 / ps=1 (9216 patch tokens, 256
registers), counted with the FlopCounterMode setup in ``scripts/tools/20251111_flops.py``
so attention's QK^T / AV matmuls are included:

    arm            MACs      vs base   params    vs base
    base 12+4    2436.8 G     1.000x   121.58M    1.000x
    e3_l12        794.2 G     0.326x    74.10M    0.609x
    e3_l35 (this) 1134.9 G    0.466x   121.02M    0.995x

So this arm matches the base's capacity at 2.1x fewer MACs. Marginal costs behind those
numbers: one trunk layer = 195.7 G MACs / 7.09M params; one ``[read -> latent
self-attend]`` block = 14.8 G / 2.04M. A trunk layer is worth 13.2 bottleneck blocks in
MACs but only 3.5 in parameters, which is the whole reason the two matching criteria
diverge so hard -- FLOPs-matching at this trunk depth needs ``latent_depth=123`` and
overshoots parameters by 2.5x, so parameters are the criterion used here.

WHAT TO WATCH: 35 sequential blocks over a 256x128 register grid is far deeper than
anything in this program has trained. Two specific risks -- optimization (drop_path is
inherited at 0.1, which over 35 blocks is a much heavier stochastic-depth budget than over
12) and wall-clock (matched MACs never mean matched time when the ops get small; this arm
has ~3x the block count of the base at lower arithmetic intensity, so measure steps/sec
before reading any efficiency claim off the MAC column).
"""

import logging

from base import build_trainer_config as _base_build_trainer_config
from regbtl_v1_2_earlyread_common import (
    build_earlyread_model_config,
    set_earlyread_loop_evals,
)
from regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform import (
    build_dataloader_config,
    build_dataset_config,
    build_train_module_config,
)
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_common_components,
    build_visualize_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)

TRUNK_DEPTH = 3
# Solved for parameter parity with the 12+4 base: 121.02M vs 121.58M (0.995x).
LATENT_DEPTH = 35
MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_earlyread_e3_l35_d128_ndvi_tanchor_newsampling_psuniform.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """The d128 NDVI tanchor base at a 3-layer trunk + 35 interleaved read/latent blocks."""
    return build_earlyread_model_config(
        common, trunk_depth=TRUNK_DEPTH, latent_depth=LATENT_DEPTH
    )


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

"""EARLY READ, param-matched: 6-layer trunk + 25 bottleneck blocks.

The capacity control for ``regbtl_v1_2_earlyread_e6_l8_d128_ndvi_tanchor_newsampling_psuniform``,
exactly as ``..._e3_l35_...`` is for the 3-layer arm: same base, same 6-layer trunk,
``register_latent_depth`` raised from 8 to 25 so the encoder+bottleneck parameter count
matches the 12+4 A/B partner.

MEASURED on one full S1+S2+Landsat window at ws16 / ps=1 (9216 patch tokens, 256
registers), counted with the FlopCounterMode setup in ``scripts/tools/20251111_flops.py``:

    arm            MACs      vs base   params    vs base
    base 12+4    2436.8 G     1.000x   121.58M    1.000x
    e6_l8        1322.0 G     0.542x    87.20M    0.717x
    e6_l25 (this) 1573.8 G    0.646x   121.89M    1.003x

The four early-read arms form a 2x2: trunk depth (3 or 6) x matching (none, or parameters).
Read the param-matched pair against the base to ask whether reading early costs capability,
and the unmatched pair against its param-matched sibling to ask what the extra bottleneck
depth actually bought -- if e3_l12 ~ e3_l35, the depth is free and the cheap arm wins.
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

TRUNK_DEPTH = 6
# Solved for parameter parity with the 12+4 base: 121.89M vs 121.58M (1.003x).
LATENT_DEPTH = 25
MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_earlyread_e6_l25_d128_ndvi_tanchor_newsampling_psuniform.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """The d128 NDVI tanchor base at a 6-layer trunk + 25 interleaved read/latent blocks."""
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

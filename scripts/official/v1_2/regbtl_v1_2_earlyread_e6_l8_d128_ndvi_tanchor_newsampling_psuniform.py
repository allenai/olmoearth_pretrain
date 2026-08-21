"""EARLY READ, hedge arm: 6-layer trunk + 8 bottleneck blocks.

Same reallocation as ``regbtl_v1_2_earlyread_e3_l12_d128_ndvi_tanchor_newsampling_psuniform``
but at a milder split, on the same base
(``regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform``).

WHY IT EXISTS: TokenLearner's own depth sweep (arXiv:2106.11297) put the best
accuracy/compute points at 1/2 and 3/4 of network depth rather than 1/4, and the evidence
that layer 3 is a viable read source is d768 and single-seed. This arm catches the case
where the 3+12 split loses something 6+8 keeps.

Read the three points together: 12+4 (base), 6+8 (here), 3+12 (primary). If 6+8 lands
between the other two, the axis is real and monotone and the split is a compute/quality
dial. If 6+8 matches 3+12, take 3+12 -- it is ~1.6x cheaper again.

MEASURED on one full S1+S2+Landsat window at ws16 / ps=1, counted with the FlopCounterMode
setup in ``scripts/tools/20251111_flops.py``:

    arm            MACs      vs base   params    vs base
    base 12+4    2436.8 G     1.000x   121.58M    1.000x
    e6_l8 (this) 1322.0 G     0.542x    87.20M    0.717x
    e3_l12        794.2 G     0.326x    74.10M    0.609x

Like e3_l12 this is a smaller model as well as a cheaper one; ``..._e6_l25_...`` is its
param-matched control. See the primary arm's docstring for the per-block marginal costs.
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
LATENT_DEPTH = 8
MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_earlyread_e6_l8_d128_ndvi_tanchor_newsampling_psuniform.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """The d128 NDVI tanchor base at a 6-layer trunk + 8 interleaved read/latent blocks."""
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

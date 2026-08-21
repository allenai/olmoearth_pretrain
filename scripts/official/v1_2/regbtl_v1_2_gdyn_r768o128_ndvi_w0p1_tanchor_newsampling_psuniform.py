"""cand_ndvi WITH A WIDE BOTTLENECK: 12-layer trunk + 4 blocks at 768, output 128.

``regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform``
(cand_ndvi) with exactly one thing changed: the register bottleneck runs at
``register_dim=768`` and a single in-graph ``Linear(768, 128)`` projects its output, so
the decoder, the supervision heads and the frozen evals still consume a 128-wide grid.

The trunk stays at 12 layers and the bottleneck at 4 read/latent pairs -- this is NOT an
early-read arm. It is the control the whole ``r768o128`` family was missing: it isolates
the width-plus-output-projection change against the production frontier, so an early-read
arm's delta can be decomposed into "wider bottleneck" and "shallower trunk" instead of
being read as one lump.

WHY IT MATTERS EVEN IF THE EARLY-READ ARMS LOSE. The d768 register runs beat every d128
arm by ~6 mIoU on the frozen ps=1 PASTIS probes (0.6447 vs 0.5800 at ws16 S1+S2, Tessera
v2 large 0.5938), and ``regbtl_v1_2_proj_common`` concluded that training the narrow width
NATIVELY under the pretext loss was the bottleneck rather than the 128-dim budget. This
run tests that conclusion directly and cheaply: keep the frontier's shape, widen only the
internal stream, and ship 128 through a learned projection that the pretext loss trains --
rather than recovering 128 from a detached distillation student afterwards. If it beats
cand_ndvi, the width fix is available without touching anything else.

NOT shared K/V and NOT thinned LSA: those are separate axes, live in the early-read
family, and would confound this comparison. Everything except the bottleneck width and the
output projection is inherited from cand_ndvi's own builders.
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

# The cand_ndvi shape, restated so this file reads as a point on the same axis as the
# early-read arms rather than as a different recipe.
TRUNK_DEPTH = 12
LATENT_DEPTH = 4
REGISTER_DIM = 768
OUTPUT_DIM = 128
MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_gdyn_r768o128_ndvi_w0p1_tanchor_newsampling_psuniform.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """cand_ndvi's 12+4 shape with the bottleneck at 768, projected to 128 on output."""
    return build_earlyread_model_config(
        common,
        trunk_depth=TRUNK_DEPTH,
        latent_depth=LATENT_DEPTH,
        register_dim=REGISTER_DIM,
        output_dim=OUTPUT_DIM,
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

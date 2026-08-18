"""EARLY READ, primary arm: 3-layer trunk + 12 bottleneck blocks.

``regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform`` with the
encoder/bottleneck depth split inverted: 3 + 12 instead of 12 + 4. The bottleneck reads
the patch tokens after three encoder layers, and every subsequent block runs on the
compressed ``n_h*n_w`` register grid instead of the ``n_h*n_w * T * channel_groups`` token
set. Everything else -- d128 wideread, regsup + NDVI at w0p1, the anchored read, the
decorrelated sampler at uniform patch sizes, the NDVI extra-decode path -- is inherited
from that script's own builders, so the depth split is the ONLY difference.

A/B partners:
* ``regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform`` --
  the 12+4 base. One-variable comparison. Its in-loop evals are the shared catalog, not
  this arm's embedding set, so compare on the embedding tasks or re-run it under
  ``set_earlyread_loop_evals``.
* ``regbtl_v1_2_earlyread_e6_l8_d128_ndvi_tanchor_newsampling_psuniform`` -- the 6+8
  hedge, the midpoint of the same axis.

WHY 3 AND NOT 6: in ``2026_05_18_v2_knn_lp_evals``, at matched block count, four reads all
taken from encoder layer 12 and four reads spread over layers 3/6/9/12 score +0.11 pts
apart over 54 tasks. Layer 3 is already a viable read source. That evidence is d768 and
single-seed, which is what the 6+8 arm is for.

MEASURED on one full S1+S2+Landsat window at ws16 / ps=1 (9216 patch tokens, 256
registers), counted with the FlopCounterMode setup in ``scripts/tools/20251111_flops.py``
so attention's QK^T / AV matmuls are included:

    arm            MACs      vs base   params    vs base
    base 12+4    2436.8 G     1.000x   121.58M    1.000x
    e3_l12 (this) 794.2 G     0.326x    74.10M    0.609x

Marginal costs behind those numbers: one trunk layer = 195.7 G MACs / 7.09M params; one
``[read -> latent self-attend]`` block = 14.8 G / 2.04M. A trunk layer is worth 13.2
bottleneck blocks in MACs but only 3.5 in parameters.

So the 3.07x MAC saving comes with a 0.61x parameter count: this arm is a genuinely
smaller model, not a neutral reallocation, and "we shipped a smaller model" is the first
alternative explanation if it underperforms. ``..._e3_l35_...`` is the param-matched
control that separates the two. The MAC ratio IMPROVES if the savings are spent on more
timesteps or modalities -- the trunk carries a 2N^2d term and grows quadratically in the
token count, the reads only linearly.

WHAT TO WATCH:
* Per-read attention entropy. Twelve reads against one shallow source could collapse onto
  the same content; if entropy flattens after read ~4 the honest config is 6+8.
* The registers now carry essentially all of the representational burden -- the trunk
  barely digests anything before compression. At d128 that burden lands on a narrow
  storage stream, so a loss here may be a WIDTH interaction rather than a depth one; the
  d768 sibling is the disambiguating run if this arm underperforms.
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
LATENT_DEPTH = 12
MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_earlyread_e3_l12_d128_ndvi_tanchor_newsampling_psuniform.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """The d128 NDVI tanchor base at a 3-layer trunk + 12 interleaved read/latent blocks.

    Memory note for ``build_train_module_config`` (inherited unchanged, including the
    base's halved rank microbatch): this arm moves activation memory OFF the encoder pass
    -- 3 layers over the full token set instead of 12 -- and ONTO the register stream,
    which is 128-wide over n_h*n_w cells. It should have strictly more headroom than the
    base at the same microbatch.
    """
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

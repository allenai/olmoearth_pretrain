"""EARLY READ: 3-layer trunk + 24 bottleneck blocks.

Replaces ``regbtl_v1_2_earlyread_e3_l35_d128_ndvi_tanchor_newsampling_psuniform``, which OOM'd on every attempt.
That arm was solved for exact parameter parity with the 12+4 base (121.02M) and needed
35 blocks to get there; at micro 64 it exceeded 80 GB and a rank died at step 2,
after which the surviving ranks blocked in the gradient allreduce until the 30-minute
NCCL watchdog fired. Six failed attempts across six distinct nodes, all with the OOM
preceding the timeout by exactly 1800 s.

WHY 24 AND NOT 35: doubling the cheap arm's bottleneck depth (12 -> 24) keeps
the memory within budget at the SAME microbatch (a read block's saved activations are the
dominant scaling term, and 24 blocks is 0.69x the 35 that failed) while
closing most of the parameter gap: 98.58M, 0.81x the base, against the cheap arm's
0.61x.

WHAT IS TRADED: this is no longer an exact capacity control, so it cannot by itself
separate "reading early costs capability" from "we shipped a smaller model". What it gives
instead is a SLOPE -- e3_l12 and e3_l24 share a trunk and differ only in
bottleneck depth, so their delta measures what that depth buys directly. If they match,
bottleneck depth is not the binding axis and the remaining parameter gap does not matter.

MEASURED (encoder MACs; scripts/tools/20251111_flops.py's counter, so attention matmuls
are included), relative to the cand_ndvi frontier:

    arm       blocks    params   xbase   MACs@3072   MACs@ws16-eval
    base 12+4      4   121.58M   1.000x      1.00x           1.00x
    e3_l12        12    74.10M   0.609x      0.37x           0.33x
    e3_l24        24    98.58M   0.811x      0.47x           0.40x
    e6_l8          8    87.20M   0.717x      0.57x           0.54x
    e6_l16        16   103.53M   0.852x      0.64x           0.59x
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
LATENT_DEPTH = 24
MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_earlyread_e3_l24_d128_ndvi_tanchor_newsampling_psuniform.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """3-layer trunk + 24 interleaved read/latent blocks, anchored reads, regsup+NDVI."""
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

"""EARLY READ + SHARED K/V: 6-layer trunk + 8 bottleneck blocks.

``regbtl_v1_2_earlyread_e6_l8_d128_ndvi_tanchor_newsampling_psuniform`` with
``register_shared_read_kv=True``: the patch tokens are projected into keys and values
ONCE and every read block attends over that single copy, instead of each block
re-projecting the full token array.

WHY: the first pass of this sweep found a hard read-depth ceiling. Arms with 4, 8, 12 and
16 read blocks trained fine at micro 64; arms with 24, 25 and 35 OOM'd on every attempt
(six failures across six distinct nodes, each with an OutOfMemoryError preceding the NCCL
collective timeout by exactly 1800 s). The cause is that a read block is cheap in FLOPs --
only the register queries attend, ~7% of a trunk layer -- but expensive in MEMORY, because
its key side is the entire token array: input-norm output, k, RoPE-rotated k and v, four
token-array-sized tensors saved for backward, per block. Block count multiplied those.

Sharing the projection collapses all of that to one copy per forward, and removes ~73% of
each read block's FLOPs (the K/V projection dominates them).

WHAT IS TRADED: the read blocks now share key/value weights, so successive reads can no
longer view the source through different lenses -- only their queries differ. Under
``interleave`` that is where the depth diversity already lived (read N queries registers
refined N-1 times), but it is a model change, not a pure optimization. It also replaces
``per_depth_read_proj``, which is meaningless against a single shared source.

Verified numerically identical to the unshared bottleneck when the per-block k/v weights
are tied to block 0's (max abs difference 0.0), with no parameters left without gradients
-- an unused parameter here would stall DDP's allreduce exactly like the OOM did.

COMPARABILITY: all four arms of this sweep run with shared K/V, so the depth axis
(l8 vs its partner) is clean. They are NOT directly comparable to the earlier unshared
runs, nor to the cand_ndvi frontier, on architecture -- only on the eval product.
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
    "regbtl_v1_2_earlyread_e6_l8_d128_ndvi_tanchor_newsampling_psuniform_sharedkv.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """6-layer trunk + 8 interleaved read/latent blocks over one shared K/V."""
    return build_earlyread_model_config(
        common,
        trunk_depth=TRUNK_DEPTH,
        latent_depth=LATENT_DEPTH,
        shared_read_kv=True,
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

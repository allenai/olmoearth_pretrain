"""LSA 1:2 AT PARAM PARITY: 3-layer trunk, 5 reads, 3 LSA blocks (768 -> 128).

``regbtl_v1_2_earlyread_e3_l4_r768o128_ndvi_tanchor_newsampling_psuniform_sharedkv``
with ``register_latent_every_n=2`` -- one latent self-attention block per two reads (plus
one after the last, so the shipped grid is always mixed) -- and the read count raised from
4 to 5 so the parameter count matches.

WHAT THIS TESTS. Not speed. Holding parameters fixed, the LSA blocks removed are spent on
extra reads, so the block count lands within one of the 1:1 arm (11 vs 11)
and depth is what drives wall-clock here. This is a REALLOCATION at neutral cost and
neutral size: the same bottleneck budget spent on more information intake (reads) and less
lateral mixing (LSA).

    arm            reads  LSA  blocks   params    MACs
    e3_l4 (1:1)      4    4      11  102.87M  119.5 G
    e3_l5 (1:2)      5    3      11  101.69M  119.8 G

WHY NOT ZERO LSA. The lsa/nolsa ablation in 2026_07_02_perceiver_knn_lp_evals is decisive
on the noic arm: +8.38 pts across 76 tasks, 61/76 wins, with swings like lfmc_woody_3k
+140.8 and pastis_sentinel2 +35.6. Latent self-attention earns its place. What is untested
is the shape of the curve between 1:1 and 0:1 -- this bets only that the eighth block
contributes far less than the first, which is the usual diminishing-returns story, and it
keeps a block after the final read either way.

WHY NOT MORE LSA. Perceiver IO runs the opposite ratio (~6 self-attends per cross-attend),
because its cross-attention over a large input array is the expensive part. That reasoning
does not carry here: with shared K/V the reads are cheap, and reads and LSA blocks are
equally launch-bound, so the ratio is a modelling choice rather than a cost one.

IF SPEED IS THE GOAL INSTEAD: keep the read count at 4 and thin the LSA -- that gives
4 reads + 2 LSA and drops the block count outright, at fewer parameters.
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
LATENT_DEPTH = 5
LATENT_EVERY_N = 2
REGISTER_DIM = 768
OUTPUT_DIM = 128
MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_earlyread_e3_l5_lsa2_r768o128_ndvi_tanchor_newsampling_psuniform_sharedkv.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """3-layer trunk, 5 reads at 768 with one LSA per two, projected to 128."""
    return build_earlyread_model_config(
        common,
        trunk_depth=TRUNK_DEPTH,
        latent_depth=LATENT_DEPTH,
        shared_read_kv=True,
        register_dim=REGISTER_DIM,
        output_dim=OUTPUT_DIM,
        latent_every_n=LATENT_EVERY_N,
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

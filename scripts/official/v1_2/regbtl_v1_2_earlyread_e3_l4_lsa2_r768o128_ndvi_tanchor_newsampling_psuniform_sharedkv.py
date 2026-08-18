"""LSA 1:2 AT FIXED READ COUNT: 3-layer trunk, 4 reads, 2 LSA blocks (768 -> 128).

``regbtl_v1_2_earlyread_e3_l4_r768o128_ndvi_tanchor_newsampling_psuniform_sharedkv``
with ``register_latent_every_n=2`` and the read count UNCHANGED -- one latent
self-attention block per two reads, plus one after the last so the shipped grid is always
mixed. This is the SPEED variant of the LSA-ratio question: blocks and parameters both
fall.

    arm                 reads  LSA  blocks    params    MACs
    e3_l4      (1:1)       4    4      11  102.87M  119.5 G
    e3_l4_lsa2 (1:2)       4    2       9   88.70M  118.3 G

Its sibling ``e3_l5_lsa2`` answers the other half: it raises the read count until
parameters match the 1:1 arm, which puts the block count back where it started. Between
them the two isolate the two things thinning LSA can buy -- a smaller, shallower model
here, and a reallocation from lateral mixing to information intake there.

Read this arm against the 1:1 sibling for the cost question (does removing half the LSA
buy real wall-clock?) and for the quality floor (how much does the model lose?). The
lsa/nolsa ablation says LSA in total is worth +8.38 pts across 76 tasks on the noic arm,
so the floor is not obviously safe -- what is untested, and what this bets on, is that the
eighth block contributes far less than the first.
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
LATENT_DEPTH = 4
LATENT_EVERY_N = 2
REGISTER_DIM = 768
OUTPUT_DIM = 128
MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_earlyread_e3_l4_lsa2_r768o128_ndvi_tanchor_newsampling_psuniform_sharedkv.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """3-layer trunk, 4 reads at 768 with one LSA per two, projected to 128."""
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

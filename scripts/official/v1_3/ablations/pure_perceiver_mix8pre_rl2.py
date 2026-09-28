"""Token-mixing pure-Perceiver arm: 8 neighbourhood-mixing blocks (3x3 cells) FIRST, then 2 [read -> latent] pairs.

Layout ``MMMMMMMMRR`` (each ``R`` is a read plus its paired latent block): all the token
computation happens before the Perceiver, which keeps only 2 ``[read -> self-attend]``
pairs (the RC's latent depth was inert between 2 and 4 layers). About 90% of the
MACs on a 12-timestep S1+S2+L8 window go to the mixing (621 G at ws16 / ps1).

Question: with nearly all compute in the (temporal + local spatial) token mixing, is
2 layers of latent depth enough?

See ``pure_perceiver_mix.py`` for the design and the evals. W&B project
``20260921_perceiver_shapes``; trained as ``v1_3_vit0_mix8pre_rl2``.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import (  # noqa: E402
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_train_module_config,
    build_visualize_config,
)
from pure_perceiver_mix import (  # noqa: E402
    build_mix_model_config,
    build_mix_trainer_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/ablations/pure_perceiver_mix8pre_rl2.py"

# 8 mixing blocks, then 2 [read -> latent] pairs.
LAYOUT = "MMMMMMMMRR"


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """``trope_ld12`` with this arm's token-mixing layout."""
    return build_mix_model_config(common, layout=LAYOUT)


def build_trainer_config(common: CommonComponents):
    """Student evals + m-eurosat / pastis on the registers, re-importing THIS module."""
    return build_mix_trainer_config(common, MODULE_PATH)


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

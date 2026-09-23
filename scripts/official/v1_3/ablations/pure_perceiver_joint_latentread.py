"""Joint latent-token attention where the LATENTS read the whole sample.

In ``pure_perceiver_joint.py`` a latent at grid cell ``(i, j)`` sees every latent and
only the tokens of its own cell: at ws16/ps1 with 12 timesteps and three modalities
that is 256 latents + 36 same-location tokens, so all cross-pixel information reaches
a token in two hops through one summary vector per location (a per-pixel temporal
transformer coupled to a 256-token spatial one). The step-160k in-loop evals show
the cost of that factorisation: the joint arm beats the pure Perceiver on PASTIS and
most kNN tasks but loses the context-hungry ones (ethiopia -4.9, descals -2.7,
africa -1.8), exactly where the pure arm's one-hop reads over all several-thousand
tokens still help.

This arm opens the latent rows (``latent_reads_all=True``): a latent query attends to
every valid token and every latent, as the pure Perceiver's reads do, while token
queries keep the cell-local pattern. Extra cost is ``M x (N + M)`` score pairs per
block: about +5% MACs at the ws16/ps1 eval shape and ~+1% at training shapes, and
1.3 points more of the 128-block grid computed (8.99% vs 7.67% with the cell-sorted
layout, which is on by default).

Everything else is ``pure_perceiver_joint.py``: 12 joint blocks, no ViT, point
latents, d768 registers, supervision, the ``[128, 64]`` student, sampler, evals.

W&B project ``20260921_perceiver_shapes``; trained as ``v1_3_vit0_latentread_joint12``.
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
    set_student_loop_evals,
)
from pure_perceiver import WANDB_PROJECT  # noqa: E402
from pure_perceiver_joint import build_model_config as _joint_model_config  # noqa: E402
from v1_2.base import build_trainer_config as _v1_2_build_trainer_config  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.joint_latent import JointLatentConfig  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/ablations/pure_perceiver_joint_latentread.py"


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """The joint arm with latents that read every valid token."""
    config = _joint_model_config(common)
    perceiver = config.encoder_config.perceiver_config
    assert isinstance(perceiver, JointLatentConfig)
    perceiver.latent_reads_all = True
    return config


def build_trainer_config(common: CommonComponents):
    """Student in-loop evals rebuilt from THIS module, in the shapes project."""
    trainer_config = set_student_loop_evals(
        _v1_2_build_trainer_config(common), MODULE_PATH
    )
    trainer_config.callbacks["wandb"].project = WANDB_PROJECT
    return trainer_config


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

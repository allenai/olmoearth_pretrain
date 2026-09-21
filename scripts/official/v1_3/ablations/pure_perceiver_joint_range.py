"""Joint latent-token attention with INTERVAL latents (sinc-gated RoPE).

``pure_perceiver_joint.py`` anchors each latent at a single time, the mean of the
sample's visible tokens, so under 3D RoPE a latent's attention over time peaks at
the window centre and falls off toward the edges. This arm encodes every latent as
the whole window instead (``latent_time_range=True``): its rotation is the average of
the point rotations over the visible tokens' time range, which is the centre rotation
with each RoPE pair scaled by ``sinc(theta_t * width / 2)``. Pairs whose temporal
frequency completes a turn or more across the window are silenced for the latents,
coarse pairs pass, and the latent's temporal kernel becomes a soft box over its window
("am I in this latent's window") rather than a peak ("how close am I to its centre").
Tokens stay points and keep their fine temporal frequencies for token-token attention.
Integrated positional encoding (mip-NeRF) applied to RoPE; no new parameters.

Everything else is ``pure_perceiver_joint.py``: 12 joint blocks, no ViT, no reads,
d768 registers, supervision, the ``[128, 64]`` student, sampler, evals.

W&B project ``20260921_perceiver_shapes``; trained as ``v1_3_vit0_rangerope_joint12``.
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

MODULE_PATH = "scripts/official/v1_3/ablations/pure_perceiver_joint_range.py"


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """The joint arm with interval (sinc-gated) latents."""
    config = _joint_model_config(common)
    perceiver = config.encoder_config.perceiver_config
    assert isinstance(perceiver, JointLatentConfig)
    perceiver.latent_time_range = True
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

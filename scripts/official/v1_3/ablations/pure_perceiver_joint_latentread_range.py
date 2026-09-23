"""Latents read the whole sample AND are encoded as temporal intervals.

The union of ``pure_perceiver_joint_latentread.py`` (``latent_reads_all=True``: a
latent query attends to every valid token, restoring the pure Perceiver's one-hop
view of the sample while tokens keep cell-local attention) and
``pure_perceiver_joint_range.py`` (``latent_time_range=True``: each latent's RoPE is
sinc-gated over the sample's visible time range, so its temporal kernel is a soft box
over the window rather than a peak at the window centre). The two are complementary
here: once a latent reads tokens from every location and every timestep, a
point-in-time anchor makes it prefer tokens near the window centre for no reason,
whereas the interval anchor treats the whole window evenly. The paired point-anchor
arm is ``v1_3_vit0_latentread_joint12``; this pair is the range-vs-point ablation
under open latent rows.

Everything else is ``pure_perceiver_joint.py``: 12 joint blocks, no ViT, d768
registers, supervision, the ``[128, 64]`` student, sampler, evals.

W&B project ``20260921_perceiver_shapes``; trained as
``v1_3_vit0_latentread_rangerope_joint12``.
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

MODULE_PATH = "scripts/official/v1_3/ablations/pure_perceiver_joint_latentread_range.py"


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """The joint arm with open latent rows and interval (sinc-gated) latents."""
    config = _joint_model_config(common)
    perceiver = config.encoder_config.perceiver_config
    assert isinstance(perceiver, JointLatentConfig)
    perceiver.latent_reads_all = True
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

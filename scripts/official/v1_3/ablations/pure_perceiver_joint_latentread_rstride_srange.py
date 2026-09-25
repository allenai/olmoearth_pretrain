"""Random-stride joint latentread with footprint-aware latents (spatial interval RoPE).

``pure_perceiver_joint_latentread_rstride.py`` with ``latent_spatial_range=True``: every latent is encoded in RoPE as the
square of pixels it stands for (side = the stride the batch drew) instead of a point.
Its RoPE pairs are scaled by ``sinc(theta_row * side / 2) * sinc(theta_col * side / 2)``,
the rotation averaged over the footprint -- the spatial counterpart of the interval
latents -- so a latent can tell whether it summarises 1 pixel or a whole patch.
Tokens stay spatial points. Everything else (budget of 512 latents, sampler,
microbatch, evals at stride 1) is ``pure_perceiver_joint_latentread_rstride.py``, so the two differ by that one flag.

W&B project ``20260921_perceiver_shapes``; trained as ``v1_3_vit0_rstride_ps8_lb512_srange_latentread_joint12``.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import (  # noqa: E402
    build_common_components,
    build_dataset_config,
    build_visualize_config,
)
from pure_perceiver_joint_latentread_rstride import (  # noqa: E402
    build_dataloader_config,
    build_train_module_config,
)
from pure_perceiver_joint_latentread_rstride import (  # noqa: E402
    build_model_config as _base_arm_model_config,
)
from pure_perceiver_joint_latentread_rstride import (  # noqa: E402
    build_trainer_config as _rstride_trainer_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.joint_latent import JointLatentConfig  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402

logger = logging.getLogger(__name__)

MODULE_PATH = (
    "scripts/official/v1_3/ablations/pure_perceiver_joint_latentread_rstride_srange.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """``pure_perceiver_joint_latentread_rstride``'s model with footprint-aware (spatially gated) latents."""
    config = _base_arm_model_config(common)
    perceiver = config.encoder_config.perceiver_config
    assert isinstance(perceiver, JointLatentConfig) and perceiver.random_latent_stride
    perceiver.latent_spatial_range = True
    return config


def build_trainer_config(common: CommonComponents):
    """The random-stride arms' evals, re-importing THIS module."""
    return _rstride_trainer_config(common, module_path=MODULE_PATH)


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

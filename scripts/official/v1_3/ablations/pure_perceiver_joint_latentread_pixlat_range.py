"""Pixel-latent joint latentread with INTERVAL latents (sinc-gated RoPE over time).

``pure_perceiver_joint_latentread_pixlat.py`` with ``latent_time_range=True``: every
pixel latent is encoded as the sample's whole visible time range (midpoint anchor,
full width) rather than a point at the mean token time, as in
``pure_perceiver_joint_latentread_range.py``. Same shapes, sampler caps, microbatch
and evals (student at ps1 plus d128 at ps4) as the point-latent pixel arm, so the two
differ by that one flag.

W&B project ``20260921_perceiver_shapes``; trained as
``v1_3_vit0_pixlat_latentread_rangerope_joint12``.
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
from pure_perceiver_joint_latentread_pixlat import (  # noqa: E402
    build_dataloader_config,
    build_train_module_config,
)
from pure_perceiver_joint_latentread_pixlat import (  # noqa: E402
    build_model_config as _pixlat_model_config,
)
from pure_perceiver_joint_latentread_pixlat import (  # noqa: E402
    build_trainer_config as _pixlat_trainer_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.joint_latent import JointLatentConfig  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402

logger = logging.getLogger(__name__)

MODULE_PATH = (
    "scripts/official/v1_3/ablations/pure_perceiver_joint_latentread_pixlat_range.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """The pixel-latent latentread arm with interval latents."""
    config = _pixlat_model_config(common)
    perceiver = config.encoder_config.perceiver_config
    assert isinstance(perceiver, JointLatentConfig) and perceiver.pixel_latents
    perceiver.latent_time_range = True
    return config


def build_trainer_config(common: CommonComponents):
    """The pixel-latent arm's evals, re-importing THIS module."""
    return _pixlat_trainer_config(common, module_path=MODULE_PATH)


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

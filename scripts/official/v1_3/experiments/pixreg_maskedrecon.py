"""``pixreg_pixrecon`` with the reconstruction scored on ENCODER-MASKED timesteps only.

Identical to ``pixreg_pixrecon.py`` (pixel-resolution d128 registers, ps 1..4, hw_p <=
24, per-cell map supervision at base 0.1) except the two time-conditioned S2 L2A / S1
heads set ``masked_timesteps_only``: the MSE is taken only over the ``(pixel,
timestep)`` units whose input the online encoder did NOT see (mask ``DECODER`` /
``TARGET_ENCODER_ONLY``), at weight 0.1 each instead of 0.05 to compensate for the
roughly halved target count. The heads still predict every observed timestep; only
the loss mask changes.

WHY: ``pixreg_pixrecon`` masks its targets on MISSING_VALUE only, so with the 50%
encode ratio about half of the reconstruction targets are timesteps the register
reads saw directly. That half is a copy task -- it asks the 128-d register to store
raw reflectance it was just handed -- and the S2 recon loss plateaued at ~0.030 from
300k steps onward in the original run. Restricting the loss to hidden timesteps turns
the head into temporal inpainting from the visible observations, the same kind of
demand the masked-modelling objective makes, so the register has to hold a
predictive seasonal model of its pixel rather than a lookup table.

Everything else (sampler, microbatch, evals at 40k steps, W&B project) is imported
from ``pixreg_pixrecon``.
"""

import logging
import sys
from pathlib import Path

# Sibling arms import the pixreg_pixrecon builders from this directory and the
# release recipe from the directory above.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import (  # noqa: E402
    build_common_components,
    build_dataset_config,
    build_visualize_config,
)
from pixreg_pixrecon import (  # noqa: E402
    apply_pixel_reconstruction,
    build_dataloader_config,
    build_pixreg_model_config,
    build_train_module_config,
)
from pixreg_pixrecon import (
    build_trainer_config as _pixreg_build_trainer_config,  # noqa: E402
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/experiments/pixreg_maskedrecon.py"

# 2x pixreg_pixrecon's 0.05: the masked-only loss sees roughly half the targets.
MASKED_RECON_WEIGHT = 0.1


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d128 pixel registers + map supervision (w0.1) + masked-timestep S2/S1 recon."""
    config = build_pixreg_model_config(common)
    return apply_pixel_reconstruction(
        config, weight=MASKED_RECON_WEIGHT, masked_only=True
    )


def build_trainer_config(common: CommonComponents):
    """pixreg_pixrecon's trainer, with the eval job re-importing THIS module."""
    return _pixreg_build_trainer_config(common, module_path=MODULE_PATH)


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

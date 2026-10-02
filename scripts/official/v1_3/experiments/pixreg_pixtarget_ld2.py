"""``pixreg_pixtarget`` with a register latent depth of 2 instead of 4.

Two interleaved ``[read -> self-attn]`` blocks in the register bottleneck instead of
four; everything else is ``pixreg_pixtarget.py`` (pixel registers, subsampled
pixel-resolution MIM targets, sampler, evals, W&B project).
"""

import logging
import sys
from pathlib import Path

# Sibling arms import the pixreg builders from this directory and the release
# recipe from the directory above.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import (  # noqa: E402
    build_common_components,
    build_dataset_config,
    build_visualize_config,
)
from pixreg_pixrecon import build_dataloader_config  # noqa: E402
from pixreg_pixrecon import (
    build_trainer_config as _pixreg_build_trainer_config,  # noqa: E402
)
from pixreg_pixtarget import (  # noqa: E402
    build_pixtarget_model_config,
    build_train_module_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/experiments/pixreg_pixtarget_ld2.py"
LATENT_DEPTH = 2


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d128 pixel registers + map supervision (w0.1), latent depth 2."""
    return build_pixtarget_model_config(common, latent_depth=LATENT_DEPTH)


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

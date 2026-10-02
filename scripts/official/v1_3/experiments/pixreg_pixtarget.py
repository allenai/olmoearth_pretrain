"""``pixreg`` (the pixel-register control) with PIXEL-resolution MIM targets, subsampled.

Identical to ``pixreg.py`` (pixel-resolution d128 registers, ps 1..4, hw_p <= 24,
per-cell map supervision at base 0.1, no reconstruction heads) except the MIM
targets: every token cell draws ONE pixel uniformly at random, each masked token's
decoder query moves to that pixel's center (the decoder's only spatial signal is
2D RoPE), and its target is the frozen projection of that single pixel (the
projection-only target at ``patch_size=1``) instead of the whole patch. See
``olmoearth_pretrain/nn/pixel_targets.py``.

The decode query count is unchanged -- one per masked token, as in ``pixreg`` -- so
the decoder's cost is too: the targets move to latent resolution without the p**2
query blow-up of decoding every pixel. At ps=1 the cell is the pixel and nothing
changes.

WHY: in ``pixreg`` the register grid is at pixel resolution but the MIM decoder still
reads it at patch resolution (queries at patch coordinates, patch-level targets), so
the MIM loss never asks a register to differ from its neighbours inside a patch.
Pixel targets put that pressure on the MIM objective itself, with spatial context
from cross-attention over the whole register grid -- unlike ``pixreg_maskedrecon``,
which reconstructed each pixel from its own register through a small MLP.

Everything else (sampler, microbatch, evals at 40k steps, W&B project) is imported
from ``pixreg_pixrecon``. The latent depth sibling is ``pixreg_pixtarget_ld2.py``.
"""

import logging
import sys
from pathlib import Path

# Sibling arms import the pixreg_pixrecon builders from this directory and the
# release recipe from the directory above.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import (  # noqa: E402
    REGISTER_LATENT_DEPTH,
    build_common_components,
    build_dataset_config,
    build_visualize_config,
)
from pixreg_pixrecon import (  # noqa: E402
    build_dataloader_config,
    build_pixreg_model_config,
)
from pixreg_pixrecon import (
    build_train_module_config as _pixreg_build_train_module_config,  # noqa: E402
)
from pixreg_pixrecon import (
    build_trainer_config as _pixreg_build_trainer_config,  # noqa: E402
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402
from olmoearth_pretrain.train.train_module.latent_mim import (  # noqa: E402
    LatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/experiments/pixreg_pixtarget.py"


def build_pixtarget_model_config(
    common: CommonComponents, latent_depth: int = REGISTER_LATENT_DEPTH
) -> LatentMIMConfig:
    """``pixreg``'s model at the given register latent depth ([read -> self] blocks)."""
    config = build_pixreg_model_config(common)
    assert config.supervision_head_config is not None
    assert not any(
        cfg.time_conditioned
        for cfg in config.supervision_head_config.modality_configs.values()
    ), "the pixreg control must not carry reconstruction heads"
    assert config.projection_only_target, "pixel targets need the projection target"
    config.encoder_config.register_latent_depth = latent_depth
    return config


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d128 pixel registers + map supervision (w0.1), latent depth 4."""
    return build_pixtarget_model_config(common)


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """``pixreg``'s train module with pixel-resolution (subsampled) MIM targets."""
    config = _pixreg_build_train_module_config(common)
    config.pixel_targets = True
    return config


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

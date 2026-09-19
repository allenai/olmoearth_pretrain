"""Pixel-resolution registers WITHOUT the raw-band reconstruction heads (control).

``pixreg_pixrecon.py`` minus ``apply_pixel_reconstruction``: pixel-resolution d128
registers, ps 1..4, hw_p <= 24, per-cell map supervision at base weight 0.1, and
nothing else. The register grid gets its fine-grained pressure only from the
per-cell map heads and the MIM decoder (which cross-attends the pixel grid).

WHY: the pixel-branch program never produced an evaluated pixel-registers-only
control. Its ``regbtl_v1_2_..._ps14_pixreg`` run went NaN at step ~12k (the loss and
grad norm are NaN from 12,058 onward; the run kept stepping to 209k and its loop-eval
job failed), so every comparison so far has been pixel-grid + {pixrecon, conv branch,
thin conv, embed read} versus the PATCH-grid control. Those four arms are within
noise of each other, which suggests the grid itself is doing the work -- this arm
tests that directly and is what ``pixreg_pixrecon`` / ``pixreg_maskedrecon`` /
``pixreg_thinconv_pixrecon`` should be read against.

WATCH: given the old NaN, check the loss and ``optim/total grad norm`` over the first
~20k steps. The v1.3 stack (DDP + bf16 autocast, fused AdamW) differs from the one
that failed, and the current ``pixreg_pixrecon`` run is finite past 120k, but the
recon heads may have been what regularized it.

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

MODULE_PATH = "scripts/official/v1_3/experiments/pixreg.py"


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d128 pixel registers + map supervision (w0.1); no reconstruction heads."""
    config = build_pixreg_model_config(common)
    assert config.supervision_head_config is not None
    assert not any(
        cfg.time_conditioned
        for cfg in config.supervision_head_config.modality_configs.values()
    ), "the pixreg control must not carry reconstruction heads"
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

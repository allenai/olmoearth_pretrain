"""``pure_perceiver_pix_mix6c_rw6.py`` with own-cell mixing (a 1x1 window, radius 0).

Same layout (``MRWMRWMRWMRWMRWMR``), per-pixel latents, write-backs and evals; the only
change is that the mixing blocks attend within each token's own patch cell (all
timesteps and modalities) instead of a 5x5 neighbourhood of cells. Among the arms
without write-backs, the own-cell control (``mix6c_rl6``) is so far level with or ahead
of its 5x5 twin, so this checks whether spatial mixing matters once the tokens also
get global context from the write-backs. 823 G at ws16 / ps1 / T12 / S1+S2+L8, 72 G at
ps4 per pixel, 226 G on the 13-task protocol with patch-stride latents.

See ``pure_perceiver_mix.py`` for the design and the evals. W&B project
``20260921_perceiver_shapes``; trained as ``v1_3_vit0_pix512_mix6c_rw6``.
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

MODULE_PATH = "scripts/official/v1_3/ablations/pure_perceiver_pix_mix6c_rw6.py"

LAYOUT = "MRWMRWMRWMRWMRWMR"
# Per-pixel latents: random stride under this budget in training (uniform), stride 1 at eval.
MAX_LATENTS = 512
# Own-cell mixing (1x1 window).
RADIUS = 0


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """``trope_ld12`` with this arm's token-mixing layout."""
    return build_mix_model_config(
        common,
        layout=LAYOUT,
        radius=RADIUS,
        pixel_latents=True,
        max_latents=MAX_LATENTS,
    )


def build_trainer_config(common: CommonComponents):
    """Student evals + m-eurosat / pastis on the registers, re-importing THIS module."""
    return build_mix_trainer_config(common, MODULE_PATH, ps4_student_evals=True)


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

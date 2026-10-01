"""``pure_perceiver_pix_mix6d384_rw6.py``: the 256-dim token-stream arm widened to 384 dims, 5x5 window.

Identical to ``pure_perceiver_pix_mix6d256_rw6.py`` (layout ``MRWMRWMRWMRWMRWMR``,
per-pixel latents under a 512 budget, stride 1 at eval, same evals) except the token
stream: 384 dims with 6 x 64-dim heads (ViT-S / OlmoEarth v1.2 Small width) instead of
256 with 4. The latent stream stays 768. Estimated from the measured 256-d and 768-d
siblings (cost ~ fixed latent side + width^2 token side): ~280 G at ws16 / ps1 / T12 /
S1+S2+L8, ~37 G at ps4 per pixel, ~149 G on the 13-task protocol with patch-stride
latents. Purpose: locate the token-width knee between 256 (cheap) and 768.

W&B project ``20260921_perceiver_shapes``; trained as ``v1_3_vit0_pix512_mix6d384_rw6``.
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

MODULE_PATH = "scripts/official/v1_3/ablations/pure_perceiver_pix_mix6d384_rw6.py"

LAYOUT = "MRWMRWMRWMRWMRWMR"
# Per-pixel latents: random stride under this budget in training (uniform), stride 1 at eval.
MAX_LATENTS = 512
# 384-dim token stream, 6 x 64-dim heads; mixing window radius 2 (5x5).
MIX_DIM = 384
MIX_HEADS = 6
RADIUS = 2


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """``trope_ld12`` with this arm's token-mixing layout."""
    return build_mix_model_config(
        common,
        layout=LAYOUT,
        radius=RADIUS,
        mix_dim=MIX_DIM,
        mix_heads=MIX_HEADS,
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

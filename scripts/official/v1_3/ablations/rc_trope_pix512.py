"""``rc_pix512`` with TIME-AWARE reads (``read_time_rope``).

Everything is ``rc_pix512.py`` (the RC with per-pixel, random-stride Perceiver
latents under a 512 budget, one latent per pixel at eval) except the Perceiver's
reads. The RC reads rotate over ``(row, col)`` only, so a read's attention can depend
on a token's spatial offset but not on its timestep; time reaches the latents only
through what the 12 ViT blocks (3D RoPE) and the additive month embedding mixed into
the token content. With ``read_time_rope`` the reads use the encoder's mixed 3D RoPE
over ``(t, row, col)``: keys carry each token's calendar day and the latent queries
sit at the mean time of the sample's visible tokens, so a read sees each token's
offset within the window and heads can specialize to parts of the season. The latent
self-attention and the decoder stay 2D (the grid has no time axis).

The pure-Perceiver arms (``pure_perceiver*.py``) read this way; the RC never has.

W&B project ``20260921_perceiver_shapes``; trained as ``v1_3_rc_trope_pix512``.
Compare against ``rc_pix512``.
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
from pure_perceiver_mix import build_mix_trainer_config  # noqa: E402
from rc_pix512 import build_model_config as _rc_pix512_build_model_config  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.flexi_vit import PerceiverConfig  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/ablations/rc_trope_pix512.py"


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """rc_pix512's model with reads that rotate over the tokens' time too."""
    config = _rc_pix512_build_model_config(common)
    perceiver = config.encoder_config.perceiver_config
    assert isinstance(perceiver, PerceiverConfig)
    perceiver.read_time_rope = True
    return config


def build_trainer_config(common: CommonComponents):
    """rc_pix512's evals (incl. ps4 per pixel), re-importing THIS module."""
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

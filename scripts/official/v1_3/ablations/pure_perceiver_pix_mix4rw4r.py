"""Token-mixing Perceiver arm: 4 mixing blocks, read, write-back, 4 more mixing blocks, read, with per-pixel latents.

Layout ``MMMMRWMMMMR``. The same 8 mixing blocks, 2 reads and 1 write-back as ``pix512_mix8pre_rwr``, but the
write-back sits in the middle of the mixing: the tokens take in the first read's global
context, mix locally with it for 4 more blocks, and are read again. Per-pixel latents as
there (512 budget, uniform stride, stride 1 at eval).
Question: is global context during the mixing worth more than after it?

See ``pure_perceiver_mix.py`` for the design and the evals (plus the d128 student at
patch size 4). W&B project ``20260921_perceiver_shapes``; trained as ``v1_3_vit0_pix512_mix4rw4r``.
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

MODULE_PATH = "scripts/official/v1_3/ablations/pure_perceiver_pix_mix4rw4r.py"

LAYOUT = "MMMMRWMMMMR"
# Per-pixel latents: random stride under this budget in training (uniform), stride 1 at eval.
MAX_LATENTS = 512


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """``trope_ld12`` with this arm's token-mixing layout."""
    return build_mix_model_config(
        common, layout=LAYOUT, pixel_latents=True, max_latents=MAX_LATENTS
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

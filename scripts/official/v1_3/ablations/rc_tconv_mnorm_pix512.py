"""``rc_tconv_pix512`` with MASK-NORMALIZED (partial) depthwise convolutions.

Everything is ``rc_tconv_pix512.py`` except the conv steps of the pixel branch
(``pixel_branch_mask_normalized``): each depthwise convolution reads only the ONLINE
cells of its window and rescales by ``k**2 / (ONLINE cells in the window)`` before the
bias. The mask is held fixed through the stack (holes are never filled) and the frame
border's zero padding counts as missing, so border cells are renormalized too.

WHY: random-mode masking leaves token-shaped zero holes in the branch's frames that
inference never has, so the plain convs see systematically darker neighbourhoods in
training than at eval. With partial convolutions an ONLINE cell's output no longer
depends on how many of its neighbours were masked, which removes that train/inference
mismatch inside a frame (time-mode masking, which hides whole timesteps, is unaffected
either way). Compare against ``rc_tconv_pix512``.

W&B project ``20260921_perceiver_shapes``; trained as ``v1_3_rc_tconv_mnorm_pix512``.
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
from rc_tconv_pix512 import apply_thin_conv  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/ablations/rc_tconv_mnorm_pix512.py"


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """rc_tconv_pix512's model with mask-normalized depthwise convolutions."""
    return apply_thin_conv(_rc_pix512_build_model_config(common), mask_normalized=True)


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

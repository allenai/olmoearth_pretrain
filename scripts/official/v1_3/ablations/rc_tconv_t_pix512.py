"""``rc_tconv_st_pix512`` WITHOUT the spatial convolutions: time convolutions only.

Everything is ``rc_tconv_st_pix512.py`` (per-modality register init,
``pixel_branch_register_pool="modality_concat"``) except the mixing
(``pixel_branch_mixing="time"``): every one of the 4 ConvNeXt-style units is
``x += mlp(dw1d_t(norm(x)))``, a depthwise kernel-3 conv over the timesteps of each
``(modality, band set, cell)`` series and the pointwise MLP. With no spatial mixing
each ``s x s`` cell is processed on its own (a per-pixel temporal model at ``s = 1``);
spatial context comes only from the Perceiver.

WHY: against ``rc_tconv_st_pix512`` (same temporal convs and head) this isolates the
spatial convs: if the two match, the branch's spatial mixing is redundant with the
Perceiver's.

W&B project ``20260921_perceiver_shapes``; trained as ``v1_3_rc_tconv_t_pix512``.
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

MODULE_PATH = "scripts/official/v1_3/ablations/rc_tconv_t_pix512.py"


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """rc_tconv_pix512's model with time convs only and the per-modality init."""
    return apply_thin_conv(
        _rc_pix512_build_model_config(common),
        mixing="time",
        register_pool="modality_concat",
    )


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

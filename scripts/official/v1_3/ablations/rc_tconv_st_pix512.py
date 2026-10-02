"""``rc_tconv_pix512`` with SPACE + TIME convolutions and a per-modality register init.

Two changes to the pixel branch of ``rc_tconv_pix512.py``:

1. Mixing (``pixel_branch_mixing="space_time"``): every one of the 4 ConvNeXt-style
   units is ``x += mlp(dw1d_t(dw2d(norm(x))))``. The depthwise 3x3 spatial conv over
   each ``(timestep, band set)`` frame is followed by a depthwise kernel-3 conv over
   the timesteps of each ``(modality, band set, cell)`` series (zero-padded, weights
   shared across modalities). Still one MLP per unit, so only the mixing changes;
   nothing mixes band sets or modalities.
2. Register init (``pixel_branch_register_pool="modality_concat"``): per modality the
   ONLINE mean over ``(timestep, band set)``, LayerNorm, a per-modality
   ``Linear(128, 128) + GELU`` (zeroed where the modality has no ONLINE unit),
   concatenated over S2 / S1 / Landsat (384), then the zero-init
   ``Linear(384, register_dim)``. Step 0 is still exactly ``rc_pix512``.

Leakage guard as in ``rc_tconv_pix512``: masked cells are zeroed before the first unit,
so nothing propagated through space or time derives from masked values, and the
per-modality pools are ONLINE-only.

WHY: the ported branch only mixes within a frame and averages timesteps at the end, so
temporal structure (phenology, change) reaches the latent init only as a mean. A cheap
temporal conv lets each cell's features see its neighbouring timesteps; the per-modality
head keeps S1 / Landsat from being averaged into the S2 features. Compare against
``rc_tconv_pix512`` and ``rc_tconv_t_pix512`` (no spatial conv).

W&B project ``20260921_perceiver_shapes``; trained as ``v1_3_rc_tconv_st_pix512``.
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

MODULE_PATH = "scripts/official/v1_3/ablations/rc_tconv_st_pix512.py"


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """rc_tconv_pix512's model with space + time convs and the per-modality init."""
    return apply_thin_conv(
        _rc_pix512_build_model_config(common),
        mixing="space_time",
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

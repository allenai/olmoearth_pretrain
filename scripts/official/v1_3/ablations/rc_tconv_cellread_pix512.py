"""``rc_tconv_pix512`` + CELL READS + masked per-pixel reconstruction.

Everything is ``rc_tconv_pix512.py`` (the RC with per-pixel random-stride Perceiver
latents under a 512 budget, plus the thin conv branch initializing the latents) plus
``pixel_branch_cell_read``: the branch keeps its final features for every ONLINE
``(timestep, band set, modality)`` unit of each ``s x s`` cell, and after each of the
4 global reads every latent cross-attends the units of ITS OWN cell, so the stack is
``[global read -> cell read -> self-attend] x 4``. See ``nn/pixel_branch.py``.

One query per cell over at most ``T * modalities`` (36 at 12 timesteps of S2 / S1 /
Landsat) keys: two dense einsums on ``[B, cells, units]``, no attention kernel and no
long masked sequence. The keys are a shared ``LayerNorm + Linear(128, 256)`` of the
branch features (which carry the branch's additive timestep encoding) plus a learned
per-modality embedding; 4 heads at width 128, a per-read query ``Linear(768, 128)``
and a zero-init output ``Linear(128, 768)``, so step 0 is exactly ``rc_tconv_pix512``.
Non-ONLINE units are never keys and a cell with no ONLINE unit is not updated.

WHY: the thin conv branch hands each latent its pixel's features averaged over time,
and the RC's reads rotate over ``(row, col)`` only, so a latent sees its pixel's
phenology only as a mean or as whatever the trunk tokens kept. ``pixreg_thinconv`` was
good on the fine-grained WorldCover benchmark and worse on crop type; the cell reads
let every latent pick out timesteps of its own pixel, conditioned on the spatial
context it has gathered so far.

RECONSTRUCTION: also the masked-timestep S2 L2A / S1 reconstruction heads of
``rc_pix512_pixrecon.py`` (``pixreg_maskedrecon``'s heads at weight 1.0 each), which
ask each latent to predict its pixel's hidden timesteps -- the temporal signal the cell
reads are meant to carry.

W&B project ``20260921_perceiver_shapes``; trained as ``v1_3_rc_tconv_cellread_pix512``.
Compare against ``rc_pix512_pixrecon`` (reconstruction without the branch) and
``rc_tconv_pix512`` (the branch without cell reads or reconstruction).
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
from rc_pix512_pixrecon import apply_masked_pixel_reconstruction  # noqa: E402
from rc_tconv_pix512 import (  # noqa: E402
    build_model_config as _rc_tconv_build_model_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.flexi_vit import PerceiverConfig  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/ablations/rc_tconv_cellread_pix512.py"

CELL_READ_HEADS = 4


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """rc_tconv_pix512's model + cell reads + the masked reconstruction heads."""
    config = _rc_tconv_build_model_config(common)
    perceiver = config.encoder_config.perceiver_config
    assert isinstance(perceiver, PerceiverConfig)
    perceiver.pixel_branch_cell_read = True
    perceiver.pixel_branch_cell_read_heads = CELL_READ_HEADS
    return apply_masked_pixel_reconstruction(config)


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

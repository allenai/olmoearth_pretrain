"""The v1.3 RC (``rc_ld1_pixtgt_pix512``) with Gaussian attention windows instead of RoPE.

Every attention in the model predicts, per query and head, a Gaussian over the key
coordinates -- a centre offset and a full (tilted) precision -- and scores keys by
bounded cosine content times that Gaussian (``olmoearth_pretrain/nn/gaussian_attention.py``):

- encoder self-attention: ``(t, row, col)`` windows (was 3D RoPE-Mixed);
- Perceiver read: ``(t, row, col)`` windows, latent queries at the visible tokens'
  mean time (was time-blind 2D RoPE), so each latent can choose which dates to read;
- Perceiver latent self-attention: ``(row, col)`` windows (was 2D RoPE);
- decoder cross-attention: ``(row, col)`` windows (was 2D RoPE).

Everything else -- data, masking, pixel targets, losses, optimizer, schedule, evals --
is ``rc_ld1_pixtgt_pix512.py``, trained as ``v1_3_rc_ld1_pixtgt_pix512`` at 9f5a25b02
(W&B ``20260921_perceiver_shapes``, the project this run logs to as well).
"""

import logging
import sys
from pathlib import Path

V1_3 = Path(__file__).resolve().parents[2] / "official" / "v1_3"
sys.path.insert(0, str(V1_3 / "ablations"))
sys.path.insert(0, str(V1_3))

from base import (  # noqa: E402
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_visualize_config,
)
from pure_perceiver_mix import build_mix_trainer_config  # noqa: E402
from rc_ld1_pix512 import build_model_config as _rc_ld1_model_config  # noqa: E402
from rc_pixtgt_pix512 import build_train_module_config  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.encodings import PositionEncoding  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/vnext/gaussian_attention/rc_ld1_pixtgt_pix512_gauss.py"


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """The RC's model with Gaussian windows in every attention."""
    config = _rc_ld1_model_config(common)
    config.encoder_config.position_encoding = PositionEncoding.GAUSSIAN_3D
    perceiver = config.encoder_config.perceiver_config
    assert perceiver is not None
    # Time-aware reads: the latent queries predict where in (t, row, col) to read.
    perceiver.read_time_rope = True
    config.decoder_config.position_encoding = PositionEncoding.GAUSSIAN_2D
    return config


def build_trainer_config(common: CommonComponents):
    """The RC's evals (ps1 + ps4 + ps2 student), re-importing THIS module."""
    return build_mix_trainer_config(
        common, MODULE_PATH, ps4_student_evals=True, ps2_student_evals=True
    )


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

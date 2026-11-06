"""The released v1.3 Base recipe at width 1024 (12 encoder layers, as in Base).

``rc_ld1_pixtgt_pix512.py`` with the encoder, decoder and register grid widened
768 -> 1024 and 16 heads (head dim 64, as in Base). Depths stay 12 (encoder) and 4
(decoder). Everything else is the release config; see ``rc_ld1_pixtgt_pix512_sized.py``.

W&B project ``20260921_perceiver_shapes``; trained as ``v1_3_d1024_rc_ld1_pixtgt_pix512``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# build_common_components / build_train_module_config are re-exported for the in-loop
# eval jobs, which import this script and look them up by name.
from rc_ld1_pixtgt_pix512_sized import (  # noqa: E402, F401
    build_common_components,
    build_sized_model_config,
    build_sized_trainer_config,
    build_train_module_config,
    run_sized,
)

from olmoearth_pretrain.internal.experiment import CommonComponents  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402

MODULE_PATH = "scripts/official/v1_3/ablations/rc_ld1_pixtgt_pix512_d1024.py"


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """The release model at width 1024, 12 encoder layers, 16 heads, registers 1024."""
    return build_sized_model_config(
        common,
        width=1024,
        encoder_depth=12,
        num_heads=16,
        decoder_depth=4,
        register_dim=1024,
    )


def build_trainer_config(common: CommonComponents):
    """The release run's trainer and loop evals, rebuilt from THIS module."""
    return build_sized_trainer_config(common, MODULE_PATH)


if __name__ == "__main__":
    run_sized(build_model_config, build_trainer_config)

"""The released v1.3 Base recipe at Large size (1024 wide, 24 encoder layers).

``rc_ld1_pixtgt_pix512.py`` with the ``large_shallow_decoder`` sizes of v1 / v1.2 Large:
encoder 1024 / 24 layers / 16 heads, decoder 1024 / 4 layers / 16 heads, and the register
grid at 1024. Everything else is the release config; see
``rc_ld1_pixtgt_pix512_sized.py``.

W&B project ``20260921_perceiver_shapes``; trained as ``v1_3_large_rc_ld1_pixtgt_pix512``.
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
from olmoearth_pretrain.internal.utils import MODEL_SIZE_ARGS  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402

MODULE_PATH = "scripts/official/v1_3/ablations/rc_ld1_pixtgt_pix512_large.py"
SIZE = MODEL_SIZE_ARGS["large_shallow_decoder"]


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """The release model at the Large preset, registers at the encoder width."""
    assert SIZE["encoder_num_heads"] == SIZE["decoder_num_heads"]
    assert SIZE["encoder_embedding_size"] == SIZE["decoder_embedding_size"]
    return build_sized_model_config(
        common,
        width=SIZE["encoder_embedding_size"],
        encoder_depth=SIZE["encoder_depth"],
        num_heads=SIZE["encoder_num_heads"],
        decoder_depth=SIZE["decoder_depth"],
        register_dim=SIZE["encoder_embedding_size"],
    )


def build_trainer_config(common: CommonComponents):
    """The release run's trainer and loop evals, rebuilt from THIS module."""
    return build_sized_trainer_config(common, MODULE_PATH)


if __name__ == "__main__":
    run_sized(build_model_config, build_trainer_config)

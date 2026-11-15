"""The released v1.3 Base recipe plus Strata's EMA-teacher invariance on the register grid.

``rc_ld1_pixtgt_pix512.py`` plus an EMA copy of the encoder (incl. the Perceiver,
momentum 0.996 -> 1.0 linear) that encodes the same sample with every encoder token
visible, at the student's drawn latent stride; the student's register grid from the
masked view, through a predictor MLP, regresses it with ``1 - cos`` (stop-grad), weight
1.0 (``olmoearth_pretrain.nn.strata_aux``). The MIM targets stay the frozen projection;
everything else is the release config.

W&B project ``20260921_perceiver_shapes``; trained as ``v1_3_emainv``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# build_common_components / build_train_module_config are re-exported for the in-loop
# eval jobs, which import this script and look them up by name.
from rc_ld1_pixtgt_pix512 import (  # noqa: E402
    build_model_config as build_release_model_config,
)
from rc_ld1_pixtgt_pix512_sized import (  # noqa: E402, F401
    build_common_components,
    build_sized_trainer_config,
    build_train_module_config,
    run_sized,
)

from olmoearth_pretrain.internal.experiment import CommonComponents  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402
from olmoearth_pretrain.nn.strata_aux import EmaInvarianceConfig  # noqa: E402

MODULE_PATH = "scripts/official/v1_3/ablations/rc_ld1_pixtgt_pix512_emainv.py"


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """The release model plus the emainv auxiliary loss."""
    config = build_release_model_config(common)
    config.ema_invariance_config = EmaInvarianceConfig()
    return config


def build_trainer_config(common: CommonComponents):
    """The release run's trainer and loop evals, rebuilt from THIS module."""
    return build_sized_trainer_config(common, MODULE_PATH)


if __name__ == "__main__":
    run_sized(build_model_config, build_trainer_config)

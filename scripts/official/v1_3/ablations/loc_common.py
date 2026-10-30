"""v1.3 release recipe (``rc_ld1_pixtgt_pix512``) + a frozen location-encoder input.

favyen 2026-10-30 (card t-i9widi): exactly the released v1.3 run
(``v1_3_rc_ld1_pixtgt_pix512``, trained at 9f5a25b02) with one change: the sample's
window-centre (lat, lon) goes through a FROZEN pretrained location encoder, and the
vector is added to every encoder token via LayerNorm + a zero-initialized Linear (so the
model starts as the release recipe). Half of the training samples (``location_dropout``
0.5) get a learned "no location" vector instead, which is also what inputs without
coordinates (all in-loop evals) use.

Shared by the two arm scripts, which set ``LOC_ENCODER`` and ``MODULE_PATH`` here
before training starts (the in-loop eval jobs re-import the arm script, so each arm is a
file rather than an environment switch):

* ``rc_ld1_pixtgt_pix512_loc_tte.py`` -- TTE "Tessellating The Earth" (ECCV 2026,
  MVRL/TTE), 512-d;
* ``rc_ld1_pixtgt_pix512_loc_satclip.py`` -- SatCLIP ResNet50-L40, 256-d (the encoder
  Strata's nd_loc used).

latlon plumbing: "latlon" (stored in every h5 sample, the 128x4 parent-tile centre) is
appended to the dataset's training modalities and to the masking strategy's
only-decode modalities. Neither the encoder nor the decoder supports it, so it is never
tokenized or used as a target; it only rides along in the batch, where the encoder's
``LocationConditioning`` reads ``batch.latlon``. Everything else (model, optimizer,
schedule, sampler, evals) is imported unchanged from the release script.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import (  # noqa: E402
    build_common_components,
    build_visualize_config,
)
from base import (  # noqa: E402
    build_dataloader_config as _base_build_dataloader_config,
)
from base import (  # noqa: E402
    build_dataset_config as _base_build_dataset_config,
)
from pure_perceiver_mix import build_mix_trainer_config  # noqa: E402
from rc_ld1_pix512 import build_model_config as _release_model_config  # noqa: E402
from rc_pixtgt_pix512 import (  # noqa: E402
    build_train_module_config as _release_train_module_config,
)

from olmoearth_pretrain.data.constants import Modality  # noqa: E402
from olmoearth_pretrain.internal.experiment import CommonComponents  # noqa: E402
from olmoearth_pretrain.train.olmo_core_patches import (  # noqa: E402
    drop_async_checkpoint_duration_metric,
)

logger = logging.getLogger(__name__)

# Re-exported for the arm scripts (the in-loop eval jobs read them off the arm module).
__all__ = [
    "build_common_components",
    "build_dataloader_config",
    "build_dataset_config",
    "build_model_config",
    "build_train_module_config",
    "build_trainer_config",
    "build_visualize_config",
]

# Set by the arm script.
MODULE_PATH = ""
LOC_ENCODER = ""
LOC_WEIGHTS = {
    "tte": "/weka/dfive-default/agent/favyen/2026-10-30-favyen-location-encoder/weights/tte.pt",
    "satclip_l40": "/weka/dfive-default/agent/favyen/2026-10-30-favyen-location-encoder/weights/satclip_l40.pt",
}
LOC_DROPOUT = 0.5

# The async-save duration metric is recorded at a rank-dependent step and hangs the gloo
# bookkeeping all-reduce (several v1.3-recipe runs hit it); see olmo_core_patches.
drop_async_checkpoint_duration_metric()


def _add_latlon_ride_along(masking_config) -> None:  # noqa: ANN001
    only_decode = list(masking_config.strategy_config["only_decode_modalities"])
    if Modality.LATLON.name not in only_decode:
        only_decode.append(Modality.LATLON.name)
    masking_config.strategy_config["only_decode_modalities"] = only_decode


def build_model_config(common: CommonComponents):  # noqa: ANN201
    """The release model with frozen-location-encoder conditioning in the encoder."""
    assert LOC_ENCODER in LOC_WEIGHTS, (
        f"arm script must set LOC_ENCODER ({LOC_ENCODER!r})"
    )
    config = _release_model_config(common)
    config.encoder_config.location_encoder = LOC_ENCODER
    config.encoder_config.location_encoder_weights = LOC_WEIGHTS[LOC_ENCODER]
    config.encoder_config.location_dropout = LOC_DROPOUT
    assert Modality.LATLON.name not in config.encoder_config.supported_modality_names
    assert Modality.LATLON.name not in config.decoder_config.supported_modality_names
    return config


def build_dataset_config(common: CommonComponents):  # noqa: ANN201
    """The release dataset, additionally loading latlon from the h5 files."""
    config = _base_build_dataset_config(common)
    config.training_modalities = [*config.training_modalities, Modality.LATLON.name]
    return config


def build_dataloader_config(common: CommonComponents):  # noqa: ANN201
    """The release dataloader; its masking marks latlon decode-only (inert)."""
    config = _base_build_dataloader_config(common)
    _add_latlon_ride_along(config.masking_config)
    return config


def build_train_module_config(common: CommonComponents):  # noqa: ANN201
    """The release train module; its masking marks latlon decode-only (inert)."""
    config = _release_train_module_config(common)
    _add_latlon_ride_along(config.masking_config)
    return config


def build_trainer_config(common: CommonComponents):  # noqa: ANN201
    """The release evals, re-importing the arm script for the in-loop eval jobs."""
    assert MODULE_PATH, "arm script must set MODULE_PATH"
    return build_mix_trainer_config(
        common, MODULE_PATH, ps4_student_evals=True, ps2_student_evals=True
    )

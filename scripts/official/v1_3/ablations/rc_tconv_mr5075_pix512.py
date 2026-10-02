"""``rc_tconv_pix512`` with the within-band-set encode ratio drawn from U[0.5, 0.75].

Everything is ``rc_tconv_pix512.py`` except the masking. ``random_time_with_decode``
applies ``encode_ratio = 0.5`` twice: to split each instance's band sets into encoded and
decoded ones, and then, inside the encoded band sets, to choose the encoded tokens
(random mode) or present timesteps (time mode). Here the band-set split is unchanged
and the second ratio is drawn per instance, ``r ~ U[0.5, 0.75]``
(``within_bandset_encode_ratio_range``), so the encoder -- and the pixel branch, which
only sees ONLINE pixels -- gets denser inputs, closer to the unmasked inputs at
inference. Decoded band sets still contribute nothing to the branch's latent init.

In time mode the timesteps not encoded are the decode band sets' targets, so a larger
``r`` also leaves those band sets fewer decoded timesteps (``1 - r``); in random mode the
decode band sets keep ``decode_ratio = 0.5``. The visible-token count rises with ``r``,
so the encoder costs more per step than ``rc_tconv_pix512``.

W&B project ``20260921_perceiver_shapes``; trained as ``v1_3_rc_tconv_mr5075_pix512``.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import (  # noqa: E402
    build_common_components,
    build_dataset_config,
    build_visualize_config,
)
from base import build_dataloader_config as _base_build_dataloader_config  # noqa: E402
from base import (
    build_train_module_config as _base_build_train_module_config,  # noqa: E402
)
from pure_perceiver_mix import build_mix_trainer_config  # noqa: E402
from rc_tconv_pix512 import build_model_config  # noqa: E402

from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig  # noqa: E402
from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.train.masking import MaskingConfig  # noqa: E402
from olmoearth_pretrain.train.train_module.latent_mim import (  # noqa: E402
    LatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/ablations/rc_tconv_mr5075_pix512.py"

WITHIN_BANDSET_ENCODE_RATIO_RANGE = (0.5, 0.75)


def apply_within_bandset_encode_ratio(masking_config: MaskingConfig) -> MaskingConfig:
    """Draw the within-band-set encode ratio per instance, in place."""
    strategy = masking_config.strategy_config
    if strategy["type"] != "random_time_with_decode":
        raise ValueError(
            "within_bandset_encode_ratio_range is a random_time_with_decode option, "
            f"got {strategy['type']!r}"
        )
    masking_config.strategy_config = {
        **strategy,
        "within_bandset_encode_ratio_range": list(WITHIN_BANDSET_ENCODE_RATIO_RANGE),
    }
    return masking_config


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """base.py's dataloader (which applies the masking) with the drawn ratio."""
    config = _base_build_dataloader_config(common)
    assert config.masking_config is not None
    apply_within_bandset_encode_ratio(config.masking_config)
    return config


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """base.py's train module with the same masking config as the dataloader."""
    config = _base_build_train_module_config(common)
    apply_within_bandset_encode_ratio(config.masking_config)
    return config


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

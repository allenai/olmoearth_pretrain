"""Control for ``rc_tconv_hp10_pix512``: the same head on the RAW S2 10 m target.

Everything is ``rc_tconv_hp10_pix512.py`` (same time-conditioned head on the d768
latents, same B02/B03/B04/B08 bands, same variance normalization and weight 0.5) except
``highpass_patch_mean=False``: the head reconstructs the normalized inputs themselves,
patch mean and seasonal curve included, at every patch size. The difference between
the two runs is the high-pass target alone.

W&B project ``20260921_perceiver_shapes``; trained as
``v1_3_rc_tconv_fullrecon10_pix512``.
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
from rc_tconv_hp10_pix512 import apply_s2_10m_reconstruction  # noqa: E402
from rc_tconv_pix512 import (  # noqa: E402
    build_model_config as _rc_tconv_build_model_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/ablations/rc_tconv_fullrecon10_pix512.py"


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """rc_tconv_pix512's model + the raw-target S2 10 m reconstruction head."""
    return apply_s2_10m_reconstruction(
        _rc_tconv_build_model_config(common), highpass=False
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

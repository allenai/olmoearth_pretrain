"""``rc_pix512`` + MASKED per-pixel raw-band reconstruction of S2 L2A and S1.

Everything is ``rc_pix512.py`` (the RC with per-pixel random-stride Perceiver latents
under a 512 budget, one latent per pixel at eval) plus the reconstruction heads of
``pixreg_maskedrecon`` (``origin/favyen/20260917-pixreg-v1_3``): one time-conditioned
supervision head per time-series input modality on the d768 latents
(``SupervisionModalityConfig.time_conditioned``, see ``nn/supervision_head.py``). A
small MLP on the latent covering the pixel, a day-of-year basis (4 harmonics) and the
pixel's offset inside that latent's ``s x s`` footprint predicts the normalized bands
of Sentinel-2 L2A (12) and Sentinel-1 (2) per (pixel, timestep). MSE, scored only on
the units the online encoder did NOT see (``masked_timesteps_only``: mask ``DECODER``
/ ``TARGET_ENCODER_ONLY``), so the heads do temporal inpainting from the visible
observations rather than copying inputs the reads were just handed.

WEIGHT: 1.0 per head. ``pixreg_maskedrecon`` used 0.1 per head at map-supervision base
weight 0.1; the RC trains the map heads at 1.0, so both scale 10x and the recon:map
ratio is unchanged (the rule of ``pixreg_pixrecon_w1``).

W&B project ``20260921_perceiver_shapes``; trained as ``v1_3_rc_pix512_pixrecon``.
Compare against ``rc_pix512``.
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

from olmoearth_pretrain.data.constants import Modality  # noqa: E402
from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402
from olmoearth_pretrain.nn.supervision_head import (  # noqa: E402
    SupervisionModalityConfig,
    SupervisionTaskType,
)

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/ablations/rc_pix512_pixrecon.py"

PIXEL_RECON_MODALITIES = (
    Modality.SENTINEL2_L2A.name,
    Modality.SENTINEL1.name,
)
# pixreg_maskedrecon's 0.1 per head at map base 0.1, scaled to the RC's map base 1.0.
PIXEL_RECON_WEIGHT = 1.0
PIXEL_RECON_TIME_HARMONICS = 4


def apply_masked_pixel_reconstruction(config: LatentMIMConfig) -> LatentMIMConfig:
    """Add the masked-timestep S2 L2A / S1 reconstruction heads, in place."""
    assert config.supervision_head_config is not None
    for name in PIXEL_RECON_MODALITIES:
        config.supervision_head_config.modality_configs[name] = (
            SupervisionModalityConfig(
                task_type=SupervisionTaskType.REGRESSION,
                num_output_channels=Modality.get(name).num_bands,
                weight=PIXEL_RECON_WEIGHT,
                regression_loss_type="mse",
                time_conditioned=True,
                time_harmonics=PIXEL_RECON_TIME_HARMONICS,
                masked_timesteps_only=True,
            )
        )
    return config


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """rc_pix512's model + the masked S2 L2A / S1 reconstruction heads."""
    return apply_masked_pixel_reconstruction(_rc_pix512_build_model_config(common))


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

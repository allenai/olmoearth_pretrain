"""``rc_tconv_pix512`` + HIGH-PASS per-pixel reconstruction of the S2 10 m bands.

Everything is ``rc_tconv_pix512.py`` plus one time-conditioned supervision head on the
d768 teacher latents (``SupervisionModalityConfig.time_conditioned``, see
``olmoearth_pretrain/nn/supervision_head.py``). It reconstructs Sentinel-2 L2A
B02/B03/B04/B08 per (pixel, timestep) with a small MLP on the latent covering the
pixel, a day-of-year basis, and the pixel's offset inside that latent's ``s x s``
footprint. The target is HIGH-PASS: each pixel minus the mean of the valid pixels of
its token patch (``P x P`` at the forward pass's patch size) at that timestep.

WHY: in the pixreg runs the reconstruction MSE was dominated by the patch mean and the
seasonal curve, which the trunk token already carries, so the loss could plateau
without the latents learning any sub-patch detail. Subtracting the patch mean leaves
exactly the detail the trunk token lacks. Only the 10 m bands are used: the 20 m and
60 m bands are upsampled to 10 m, so their sub-patch variation is mostly resampling.
At ``P = 1`` there is no sub-patch detail and the loss is zero.

The loss is MSE normalized by the batch's per-band target variance (detached), so it
starts near 1, at weight 0.5 (the pixreg_pixrecon_w1 ratio of reconstruction to the
map base weight 1.0). The control ``rc_tconv_fullrecon10_pix512`` uses the same head,
bands, normalization and weight on the raw target, isolating the high-pass part.

W&B project ``20260921_perceiver_shapes``; trained as ``v1_3_rc_tconv_hp10_pix512``.
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
from rc_tconv_pix512 import (  # noqa: E402
    build_model_config as _rc_tconv_build_model_config,
)

from olmoearth_pretrain.data.constants import Modality  # noqa: E402
from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402
from olmoearth_pretrain.nn.supervision_head import (  # noqa: E402
    SupervisionModalityConfig,
    SupervisionTaskType,
)

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/ablations/rc_tconv_hp10_pix512.py"

RECON_MODALITY = Modality.SENTINEL2_L2A.name
RECON_BANDS = ("B02", "B03", "B04", "B08")
RECON_WEIGHT = 0.5
RECON_TIME_HARMONICS = 4
RECON_HIDDEN_DIM = 64


def apply_s2_10m_reconstruction(
    config: LatentMIMConfig, highpass: bool
) -> LatentMIMConfig:
    """Add the time-conditioned S2 10 m reconstruction head, in place."""
    assert config.supervision_head_config is not None
    band_order = Modality.SENTINEL2_L2A.band_order
    config.supervision_head_config.modality_configs[RECON_MODALITY] = (
        SupervisionModalityConfig(
            task_type=SupervisionTaskType.REGRESSION,
            num_output_channels=len(RECON_BANDS),
            weight=RECON_WEIGHT,
            time_conditioned=True,
            time_harmonics=RECON_TIME_HARMONICS,
            time_mlp_hidden_dim=RECON_HIDDEN_DIM,
            band_indices=[band_order.index(b) for b in RECON_BANDS],
            highpass_patch_mean=highpass,
            normalize_by_target_variance=True,
        )
    )
    return config


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """rc_tconv_pix512's model + the high-pass S2 10 m reconstruction head."""
    return apply_s2_10m_reconstruction(
        _rc_tconv_build_model_config(common), highpass=True
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

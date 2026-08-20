"""d768 teacher + lin supstu0p1 student + time-conditioned NDVI supervision, w1.

``regbtl_v1_2_gdyn_d768_proj128lin_supstu0p1_w1_newsampling_psuniform`` (see its
docstring for the supstu0p1 dose-response rationale) with the time-conditioned
NDVI regsup arm added. With ``supervision_source="both"`` the NDVI head, like
the other supervision heads, attaches at BOTH widths -- so the student gets
direct NDVI pressure at 0.1x, which is the strongest remaining route for the
Ethiopia-kNN NDVI effect to survive into the shipped 128-d embedding (in the
gram sweep, ndvi arms held Ethiopia kNN 0.47-0.50 vs 0.44-0.46 without).

One of four supstu arms launched 2026-08-20 (plain / +ndvi /
+ndvi+cloudmask0p5 / +ndvi+cloudmask0p5+stuunif0p1).

In-loop evals: the year-aligned early-read set on both heads (student tasks
first), including aeftrial metrics -- ``set_proj_earlyread_loop_evals``.
"""

import logging

from base import build_trainer_config as _base_build_trainer_config
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_common_components,
    build_visualize_config,
)
from regbtl_v1_2_newsampling_common import (
    apply_microbatch,
    apply_new_sampling,
    apply_uniform_patch_sizes,
)
from regbtl_v1_2_proj_common import (
    SUPERVISION_BASE_WEIGHT_W1,
    build_proj_model_config,
    set_proj_earlyread_loop_evals,
)
from regbtl_v1_2_regsup_common import (
    build_extra_decode_dataloader_config,
    build_extra_decode_dataset_config,
    build_extra_decode_train_module_config,
)

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

EXTRA_DECODE_MODALITIES = [Modality.NDVI.name]
PROJECTION_SUPERVISION_SCALE = 0.1
MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_gdyn_d768_proj128lin_supstu0p1_ndvi_w1_newsampling_psuniform.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Supervision incl. NDVI on both widths, student heads at 0.1x."""
    return build_proj_model_config(
        common,
        base_weight=SUPERVISION_BASE_WEIGHT_W1,
        projection_type="linear",
        supervision_source="both",
        projection_supervision_weight_scale=PROJECTION_SUPERVISION_SCALE,
        include_ndvi=True,
    )


def build_dataset_config(common: CommonComponents):
    """Base dataset config, additionally deriving ndvi from the raw S2 bands."""
    return build_extra_decode_dataset_config(common, EXTRA_DECODE_MODALITIES)


def build_dataloader_config(common: CommonComponents):
    """ndvi-aware newsampling dataloader at uniform patch sizes."""
    return apply_uniform_patch_sizes(
        apply_new_sampling(
            build_extra_decode_dataloader_config(common, EXTRA_DECODE_MODALITIES)
        )
    )


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """ndvi-aware 1fwd + fused AdamW train module at the newsampling microbatch."""
    return apply_microbatch(
        build_extra_decode_train_module_config(common, EXTRA_DECODE_MODALITIES)
    )


def build_trainer_config(common: CommonComponents):
    """Base trainer + the year-aligned early-read evals on both heads."""
    return set_proj_earlyread_loop_evals(
        _base_build_trainer_config(common), MODULE_PATH
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

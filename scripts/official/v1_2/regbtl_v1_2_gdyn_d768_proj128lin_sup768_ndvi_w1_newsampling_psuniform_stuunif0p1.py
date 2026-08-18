"""d768 teacher + detached linear student, sup768 w1 + NDVI + student-only uniformity.

The combined cell of the ndvi x student-uniformity 2x2 on the lin_sup768_w1 base:
the time-conditioned NDVI supervision arm (see ``..._sup768_ndvi_w1_newsampling_
psuniform``) AND AlphaEarth's batch-uniformity term on the detached student alone
(``projection_uniformity_weight=0.1``, no register_unit_norm -- see
``..._sup768_w1_newsampling_psuniform_stuunif0p1``). Both single-knob arms document
their own rationale; this cell answers whether they compose.

In-loop evals are the early-read year-aligned set on BOTH heads (teacher + proj128
student), including the aeftrial_* balanced-trial metrics -- see
``set_proj_earlyread_loop_evals``.

A/B partners: the two single-knob arms and (via checkpoint sweep) the base
``regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform``.
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
# Matches the single-knob stuunif0p1 arm.
PROJECTION_UNIFORMITY_WEIGHT = 0.1
MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_gdyn_d768_proj128lin_sup768_ndvi_w1_newsampling_psuniform_stuunif0p1.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d768 teacher + detached linear student, register supervision incl. NDVI."""
    return build_proj_model_config(
        common,
        base_weight=SUPERVISION_BASE_WEIGHT_W1,
        projection_type="linear",
        supervision_source="registers",
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
    """ndvi-aware train module + the student-only uniformity term."""
    config = apply_microbatch(
        build_extra_decode_train_module_config(common, EXTRA_DECODE_MODALITIES)
    )
    config.projection_uniformity_weight = PROJECTION_UNIFORMITY_WEIGHT
    return config


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

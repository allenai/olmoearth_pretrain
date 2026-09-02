"""d768 teacher + detached linear [128, 64] student, sup768 w1 + time-conditioned NDVI.

``regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform`` (the current
release-readiness distillation candidate's training run) with the time-conditioned
NDVI regsup arm added: an MLP on [register_cell ; phi(day_of_year)] regressing
per-cell NDVI at every observed timestep. NDVI is a derived decode-only modality
(computed in the dataset from raw S2 L2A B04/B08), excluded from the token budget,
and costs nothing at serving -- the head is dropped at inference.

WHY: one axis of the ndvi x student-uniformity 2x2 on the lin_sup768_w1 base. The
NDVI head is the isolated driver of the Ethiopia effect in the pretrained family
(+8.6 LP from the channel alone), and in the gram sweep every ndvi distillation arm
held Ethiopia kNN (0.47-0.50 vs 0.44-0.46 for the no-ndvi twins) -- but all of those
carried a gram-scope change AND tanchor, so the NDVI marginal on the unmodified
lin_sup768_w1 recipe was never measured. This is that cell. The teacher here differs
from the gram-sweep arms in neither carrying tanchor nor any gram-scope change: the
student keeps the LatentMIMTrainModule default cosine + Gram distillation.

In-loop evals are the early-read year-aligned set on BOTH heads (teacher + proj128
student), including the aeftrial_* balanced-trial metrics -- see
``set_proj_earlyread_loop_evals``. The base run's checkpoints must be swept under
this task set for the A/B (its own in-loop set was the PASTIS/fifty_cities one).

A/B partner: ``regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform``.
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
MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_gdyn_d768_proj128lin_sup768_ndvi_w1_newsampling_psuniform.py"
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

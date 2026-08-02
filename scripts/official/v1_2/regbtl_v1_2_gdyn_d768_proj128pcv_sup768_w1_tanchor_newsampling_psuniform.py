"""d768 teacher + detached perceiver [128, 64] student, sup768 w1 + anchored register read (tanchor).

``regbtl_v1_2_gdyn_d768_proj128pcv_sup768_w1_newsampling_psuniform`` with
``register_temporal_anchor="year_start"`` added: the register (and pcv student)
reads run axial 3D RoPE with registers at t=0 and patch keys at year-anchored
relative days, so season-selective read heads are expressible.
Both temporal arms measured ~null at d128/w0p1; this re-tests on the d768/w1 teacher
whose ceiling the student inherits (and, for tanchor, on the pcv student too --
the student mirrors the primary's ``register_temporal_anchor``, so the anchored
read applies to BOTH heads at once). A/B partner:
``regbtl_v1_2_gdyn_d768_proj128pcv_sup768_w1_newsampling_psuniform``.

Embedding evals every 40k steps with the PASTIS (base + projected) tasks first --
see ``add_proj_loop_eval_beaker_job``.
"""

import logging

from base import build_trainer_config as _base_build_trainer_config
from regbtl_v1_2_faster_common import build_faster_train_module_config
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_common_components,
    build_visualize_config,
)
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_dataloader_config as _base_build_dataloader_config,
)
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_dataset_config as _base_build_dataset_config,
)
from regbtl_v1_2_newsampling_common import (
    apply_microbatch,
    apply_new_sampling,
    apply_uniform_patch_sizes,
)
from regbtl_v1_2_proj_common import (
    SUPERVISION_BASE_WEIGHT_W1,
    add_proj_loop_eval_beaker_job,
    build_proj_model_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

REGISTER_TEMPORAL_ANCHOR = "year_start"
MODULE_PATH = "scripts/official/v1_2/regbtl_v1_2_gdyn_d768_proj128pcv_sup768_w1_tanchor_newsampling_psuniform.py"


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d768 teacher + detached perceiver student, supervision on the registers."""
    return build_proj_model_config(
        common,
        base_weight=SUPERVISION_BASE_WEIGHT_W1,
        projection_type="perceiver",
        supervision_source="registers",
        temporal_anchor=REGISTER_TEMPORAL_ANCHOR,
    )


def build_dataset_config(common: CommonComponents):
    """Base v1.2 dataset config."""
    return _base_build_dataset_config(common)


def build_dataloader_config(common: CommonComponents):
    """Newsampling dataloader at uniform patch sizes."""
    return apply_uniform_patch_sizes(
        apply_new_sampling(_base_build_dataloader_config(common))
    )


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """1fwd + fused AdamW train module at the newsampling microbatch size."""
    return apply_microbatch(build_faster_train_module_config(common))


def build_trainer_config(common: CommonComponents):
    """Base trainer + in-loop evals on the d768, 128d and 64d heads."""
    return add_proj_loop_eval_beaker_job(
        _base_build_trainer_config(common),
        MODULE_PATH,
        embedding_eval_interval_steps=40000,
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

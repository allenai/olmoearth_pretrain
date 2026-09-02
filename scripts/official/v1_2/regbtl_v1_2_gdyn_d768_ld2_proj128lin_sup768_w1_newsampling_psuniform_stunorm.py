"""stunorm recipe at register_latent_depth=2: block-count ablation, stunorm line.

TWO flags off the RC: ``register_latent_depth = 2`` (the depth contrast -- see the
plain ``ld2`` sibling for the full motivation and MACs arithmetic) and
``register_projection_output_norm = True`` (the student output LayerNorm, matching
``regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform_stunorm``). Its
one-flag depth partner is that stunorm run (4 blocks, same W&B project); the plain
``ld2`` twin isolates depth on the un-normed RC line.

IN-LOOP EVALS: ``set_proj_aeftrial_loop_evals`` (AEF balanced trials +
year-aligned PASTIS, unmasked S1+S2+Landsat, both student widths, 80k interval) --
identical to every arm in the stunorm project.

EVAL-ARM NAMING: the ``ld2`` token sits BEFORE ``proj128lin`` so no existing arm
name is a prefix of this one (the CSV export merges arms by longest prefix).
"""

import logging

from base import build_trainer_config as _base_build_trainer_config
from regbtl_v1_2_faster_common import build_faster_train_module_config
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_common_components,
    build_dataset_config,
    build_visualize_config,
)
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_dataloader_config as _base_build_dataloader_config,
)
from regbtl_v1_2_newsampling_common import (
    apply_microbatch,
    apply_new_sampling,
    apply_uniform_patch_sizes,
)
from regbtl_v1_2_proj_common import (
    SUPERVISION_BASE_WEIGHT_W1,
    build_proj_model_config,
    set_proj_aeftrial_loop_evals,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_2/regbtl_v1_2_gdyn_d768_ld2_proj128lin_sup768_w1_newsampling_psuniform_stunorm.py"

REGISTER_LATENT_DEPTH = 2


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """The stunorm model with a 2-block ([read -> self] x 2) bottleneck."""
    config = build_proj_model_config(
        common,
        base_weight=SUPERVISION_BASE_WEIGHT_W1,
        projection_type="linear",
        supervision_source="registers",
    )
    config.encoder_config.register_latent_depth = REGISTER_LATENT_DEPTH
    config.encoder_config.register_projection_output_norm = True
    return config


def build_dataloader_config(common: CommonComponents):
    """Newsampling dataloader at uniform patch sizes."""
    return apply_uniform_patch_sizes(
        apply_new_sampling(_base_build_dataloader_config(common))
    )


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """1fwd + fused AdamW train module at the newsampling microbatch size."""
    return apply_microbatch(build_faster_train_module_config(common))


def build_trainer_config(common: CommonComponents):
    """Base trainer + AEF trials and PASTIS on both student widths."""
    return set_proj_aeftrial_loop_evals(_base_build_trainer_config(common), MODULE_PATH)


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

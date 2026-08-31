"""RC recipe at register_latent_depth=2: price the bottleneck's block count.

THE CHANGE, AND ONLY IT: ``register_latent_depth = 2`` -- the interleaved
bottleneck runs 2 ``[read -> latent self-attention]`` block pairs instead of the
recipe's 4. Teacher, student, supervision, sampler and data are byte-identical to
``regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform`` (the shipped
release candidate), so the pair is a one-flag depth contrast. The ld3 sibling
completes the depth ladder, and the ``_stunorm`` twins repeat both depths with the
student output LayerNorm.

WHY. On the 13-task MACs protocol (``scripts/tools/20251111_flops.py``) the
perceiver head costs +79.8G MACs over the v1.2 Base encoder (1.51x), scaling
linearly at ~19.9G per block pair: ld3 is 1.38x, ld2 1.25x. The only existing
read-count evidence is from the May mdr family (mdr3's 4 reads vs the 3-read
mdr_6_9_12 / mdr_8_10_12: -0.95 / -1.03 pts over 54 bounded metrics), which is a
different read design (multi-depth sources, non-interleaved), single-seed, and
barely above the LP noise floor. These arms price depth on the RC recipe itself,
where the efficiency claim in the v1.3 report needs it.

IN-LOOP EVALS: same set as the stunorm arms (``set_proj_aeftrial_loop_evals``):
AEF balanced trials + year-aligned PASTIS on unmasked S1+S2+Landsat at both
student widths, 80k interval -- NOT the RC's own 40k proj chain -- so all four
depth arms and the 4-block stunorm run read out identically in one W&B project.
The plain 4-block anchor is the RC run itself (2026_07_02_perceiver).

EVAL-ARM NAMING: the ``ld2`` token sits BEFORE ``proj128lin`` so no existing arm
name is a prefix of this one (the CSV export merges arms by longest prefix; a
trailing ``_ld2`` suffix would fold these rows into the RC's).
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

MODULE_PATH = "scripts/official/v1_2/regbtl_v1_2_gdyn_d768_ld2_proj128lin_sup768_w1_newsampling_psuniform.py"

REGISTER_LATENT_DEPTH = 2


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """The RC model with a 2-block ([read -> self] x 2) bottleneck."""
    config = build_proj_model_config(
        common,
        base_weight=SUPERVISION_BASE_WEIGHT_W1,
        projection_type="linear",
        supervision_source="registers",
    )
    config.encoder_config.register_latent_depth = REGISTER_LATENT_DEPTH
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

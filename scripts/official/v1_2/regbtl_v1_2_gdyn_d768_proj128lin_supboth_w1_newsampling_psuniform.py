"""d768 teacher + detached linear [128, 64] student (lin), supervision: supboth, w1, newsampling psuniform.

One of the 6 detached-student ("proj128") runs -- see ``regbtl_v1_2_proj_common``
for the full motivation and the variant matrix. The student is trained at 128d with
its first 64 dims as a self-sufficient Matryoshka prefix (per-prefix back-projection,
Gram term and, when enabled, supervision head), so one artifact ships both widths.
This arm:

* student architecture: **per-cell Linear(768, 128) on the detached registers**
* supervision heads: **separate heads on BOTH the d768 registers and the 128d student**

A/B against the sup768 sibling isolates what supervision directly on the student
buys on top of distillation.

Recipe: d768 wideread regbtl, regsup base_weight 1.0 (w1), decorrelated sampler at
UNIFORM patch sizes (the newsampling psuniform recipe). The student is trained by
the distillation terms (cosine + Gram) plus its own supervision heads; its input is detached, so the
encoder and the primary bottleneck train exactly as regsup_w1 does.
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
    add_proj_loop_eval_beaker_job,
    build_proj_model_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_2/regbtl_v1_2_gdyn_d768_proj128lin_supboth_w1_newsampling_psuniform.py"


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d768 teacher + detached linear [128, 64] student, supervision on both widths."""
    return build_proj_model_config(
        common,
        projection_type="linear",
        supervision_source="both",
    )


def build_dataloader_config(common: CommonComponents):
    """Newsampling dataloader at uniform patch sizes."""
    return apply_uniform_patch_sizes(
        apply_new_sampling(_base_build_dataloader_config(common))
    )


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """1fwd + fused AdamW train module at the newsampling microbatch size."""
    return apply_microbatch(build_faster_train_module_config(common))


def build_trainer_config(common: CommonComponents):
    """Base trainer + in-loop evals on BOTH the d768 and projected 128d heads."""
    return add_proj_loop_eval_beaker_job(
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

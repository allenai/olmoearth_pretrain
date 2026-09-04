"""Query-token compaction arm: native d128 registers, no distillation student.

The compaction ablation for the v1.3 report. The report's Compaction section
claims query-token compaction and distillation yield similar d128 performance;
this arm is the query-token side of that A/B, matched to the shipped RC
(``regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsamp_psuniform_stunorm_mlpgram1``)
on every component that exists in both:

* ``register_dim = 128`` with wideread (attention at encoder width), so the
  Perceiver's query tokens ARE the served 128-d embedding -- no student, no
  distillation losses, no back-projections, no stunorm (those only exist on the
  distillation side).
* register supervision at w1 on the (d128) registers, newsampling at uniform
  patch sizes, faster train module -- all identical to the RC.

The earlier d128 wideread runs do NOT qualify as this arm: they were w0p1 with
extra supervision arms (since removed), three recipe steps behind the RC.

IN-LOOP EVALS: the AEF trials + PASTIS on the REGISTER grid itself (no
``eval_on_projected_registers`` -- there is no projection). 9 tasks at one
width, so the 40k interval is safe (the 12-task early-read job fit in 40k).

EVAL-ARM NAMING: the ``qtc`` token sits directly after ``regbtl_v1_2`` so no
existing arm name is a prefix of this one (the CSV export merges arms by
longest prefix).
"""

import logging
from dataclasses import replace

from base import build_trainer_config as _base_build_trainer_config
from olmo_core.train.common import Duration
from regbtl_v1_2_common import LOOP_EVAL_CLUSTERS
from regbtl_v1_2_faster_common import (
    build_faster_train_module_config,
    build_wideread_regbtl_model_config,
)
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
    _AEFTRIAL_LOOP_EVAL_NAMES,
    SUPERVISION_BASE_WEIGHT_W1,
)
from regbtl_v1_2_regsup_common import add_register_supervision

from olmoearth_pretrain.internal.all_evals import EMBEDDING_EVAL_TASKS
from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_2/regbtl_v1_2_qtc_gdyn_d128_wideread_regsup_w1_newsampling_psuniform.py"

REGISTER_DIM = 128


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d128 wideread registers + w1 register supervision; no student."""
    config = build_wideread_regbtl_model_config(
        common, latent_self_attn=True, register_dim=REGISTER_DIM
    )
    config = add_register_supervision(
        config,
        include_latlon=False,
        base_weight=SUPERVISION_BASE_WEIGHT_W1,
    )
    config.supervision_source = "registers"
    return config


def build_dataloader_config(common: CommonComponents):
    """Newsampling dataloader at uniform patch sizes."""
    return apply_uniform_patch_sizes(
        apply_new_sampling(_base_build_dataloader_config(common))
    )


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """1fwd + fused AdamW train module at the newsampling microbatch size."""
    return apply_microbatch(build_faster_train_module_config(common))


def set_register_aeftrial_loop_evals(
    trainer_config, module_path: str, *, interval_steps: int = 40000
):
    """AEF trials + PASTIS scored on the register grid (no student exists)."""
    base_tasks = {
        name: replace(
            EMBEDDING_EVAL_TASKS[name], eval_interval=Duration.steps(interval_steps)
        )
        for name in _AEFTRIAL_LOOP_EVAL_NAMES
    }
    # PASTIS first: eval jobs get preempted and trailing tasks lose their metrics.
    tasks = {
        name: base_tasks[name]
        for name in sorted(base_tasks, key=lambda n: (not n.startswith("pastis"), n))
    }
    evaluator = trainer_config.callbacks["downstream_evaluator"]
    evaluator.tasks = tasks
    evaluator.run_as_beaker_job = True
    evaluator.beaker_eval_module_path = module_path
    evaluator.beaker_eval_clusters = list(LOOP_EVAL_CLUSTERS)
    return trainer_config


def build_trainer_config(common: CommonComponents):
    """Base trainer + AEF trials and PASTIS on the d128 registers."""
    return set_register_aeftrial_loop_evals(
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

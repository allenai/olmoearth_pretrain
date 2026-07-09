"""v1.2 base all-speedups arm + torch.compile, judged against a seed-noise envelope.

Re-tests torch.compile on the new runtime stack. The previous compile arms
(2026_07_09_compile_fused_ab) ran on FSDP2 with bf16 param-cast -- compile's
worst environment (graph breaks at unshard boundaries, mixed-precision casts
inside compiled regions) -- and were rejected for diverging from the production
loss curve at the ~3000-step phase transition. That criterion is unachievable
for any change that alters floating-point execution order: seed changes diverge
there too. This arm changes exactly two things vs that experiment:

1. Runtime stack: replicated DP (plain fp32 module + bf16 autocast, manual
   grad all-reduce) + projection-only target, i.e. ``base_proj_target_ddp_e2e``
   -- torch.compile's happy path, no FSDP interactions. The Disc loss is
   already forced to fp32 outside autocast, so inductor cannot touch loss math
   (``compile_loss`` stays False).
2. Verdict: compare against the SEED-NOISE ENVELOPE arms (same config, no
   compile, different ``--init_seed``) launched by ``launch_compile_envelope.sh``
   -- accept iff loss level and 4k/8k eval-job metrics fall inside the envelope,
   NOT point-wise curve tracking.

Compile settings are deliberately unchanged from the old arms
(max-autotune-no-cudagraphs, dynamic=False per block) so stack+criterion are
the only moving parts. TORCH_LOGS=recompiles is set on the Beaker job: check
rank-0 stderr for recompile storms (variable patch sizes may shape-specialize;
if the count is large, a follow-up arm with mark_dynamic/bucketing is needed
before drawing quality conclusions).
"""

import logging

from base import build_common_components as _base_build_common_components
from base import (
    build_dataloader_config,
    build_dataset_config,
    build_visualize_config,
)
from base import build_trainer_config as _base_build_trainer_config
from base_proj_target_ddp_e2e import (
    build_model_config,
)
from base_proj_target_ddp_e2e import (
    build_train_module_config as _proj_ddp_build_train_module_config,
)
from olmo_core.launch.beaker import BeakerEnvVar

from olmoearth_pretrain.internal.experiment import CommonComponents, main

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_2/base_proj_target_ddp_compile_e2e.py"
WANDB_PROJECT = "2026_07_08_fused_adamw_e2e"
LOOP_EVAL_CLUSTERS = ["ai2/jupiter", "ai2/ceres"]


def build_common_components(*args, **kwargs) -> CommonComponents:
    """Common components with TORCH_LOGS=recompiles for the recompile audit."""
    common = _base_build_common_components(*args, **kwargs)
    if common.launch is not None:
        common.launch.env_vars.append(
            BeakerEnvVar(name="TORCH_LOGS", value="recompiles")
        )
    return common


def build_train_module_config(common: CommonComponents):
    """proj_target+ddp train module config with torch.compile enabled."""
    config = _proj_ddp_build_train_module_config(common)
    config.compile_model = True
    return config


def build_trainer_config(common: CommonComponents):
    """Base trainer config with the in-loop evals run as separate Beaker jobs."""
    trainer_config = _base_build_trainer_config(common)
    evaluator = trainer_config.callbacks["downstream_evaluator"]
    evaluator.run_as_beaker_job = True
    evaluator.beaker_eval_module_path = MODULE_PATH
    evaluator.beaker_eval_clusters = list(LOOP_EVAL_CLUSTERS)
    trainer_config.callbacks["wandb"].project = WANDB_PROJECT
    return trainer_config


if __name__ == "__main__":
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )

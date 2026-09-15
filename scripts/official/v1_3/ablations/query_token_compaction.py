"""Query-token compaction ablation: native d128 registers, no distillation student.

The compaction ablation for the v1.3 report. The Compaction section compares
compacting to 128 dimensions in the Perceiver's query tokens directly against
distilling a 128-d student from a 768-d teacher; this arm is the query-token side
of that A/B, matched to ``base.py`` on every component that exists in both:

* ``register_dim = 128`` with attention at encoder width, so the register grid IS
  the served 128-d embedding -- no student, no distillation losses, no
  back-projections, no student LayerNorm (those only exist on the distillation side);
* register supervision at the same weight on the (d128) registers, the same sampler,
  the same train module.

IN-LOOP EVALS: the AEF trials + PASTIS scored on the register grid itself (there is
no projection). Nine tasks at one width, so a 40k interval is safe.

Trained as ``regbtl_v1_2_qtc_gdyn_d128_wideread_regsup_w1_newsamp_psuniform``
(W&B project ``2026_08_26_student_norm``).
"""

import logging
import sys
from pathlib import Path

# The ablations import the release recipe from the directory above.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import (  # noqa: E402
    aeftrial_loop_eval_tasks,
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_register_bottleneck_model_config,
    build_supervision_head_config,
    build_train_module_config,
    build_visualize_config,
    route_loop_evals_through_beaker,
)
from base import build_trainer_config as _base_build_trainer_config  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/ablations/query_token_compaction.py"

REGISTER_DIM = 128
# Nine single-width tasks fit comfortably in 40k steps.
LOOP_EVAL_INTERVAL_STEPS = 40000


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d128 registers + register supervision; no student."""
    config = build_register_bottleneck_model_config(common, register_dim=REGISTER_DIM)
    config.supervision_head_config = build_supervision_head_config()
    return config


def build_trainer_config(common: CommonComponents):
    """AEF trials + PASTIS scored on the register grid, routed through Beaker."""
    trainer_config = _base_build_trainer_config(common)
    return route_loop_evals_through_beaker(
        trainer_config, MODULE_PATH, aeftrial_loop_eval_tasks(LOOP_EVAL_INTERVAL_STEPS)
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

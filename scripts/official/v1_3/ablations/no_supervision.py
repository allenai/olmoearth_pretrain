"""No-supervision ablation: v1.3 with the register supervision heads deleted.

The direct-supervision ablation for the v1.3 report. The Aggregation section
claims direct supervision on the register (query) tokens is one of the mechanisms
ensuring spatial locality; this arm removes it and nothing else.

THE CHANGE, AND ONLY IT: ``supervision_head_config = None``. The teacher then
trains on the LatentMIM loss alone, and the student on the distillation losses
alone. Everything else -- d768 registers, the linear+LayerNorm student, MLP
back-projections, Gram weight, sampler, data, in-loop evals -- is byte-identical to
``base.py``.

Trained as ``regbtl_v1_2_nosup_gdyn_d768_proj128lin_newsamp_psuniform_stunorm_mlpgram1``
(W&B project ``2026_08_26_student_norm``).
"""

import logging
import sys
from pathlib import Path

# The ablations import the release recipe from the directory above.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import (  # noqa: E402
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_train_module_config,
    build_visualize_config,
    set_student_loop_evals,
)
from base import build_model_config as _base_build_model_config  # noqa: E402
from base import build_trainer_config as _base_build_trainer_config  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/ablations/no_supervision.py"


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """The v1.3 model with the register supervision heads removed."""
    config = _base_build_model_config(common)
    config.supervision_head_config = None
    return config


def build_trainer_config(common: CommonComponents):
    """Same student in-loop evals as the release run, pointed at this module."""
    trainer_config = _base_build_trainer_config(common)
    return set_student_loop_evals(trainer_config, MODULE_PATH)


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

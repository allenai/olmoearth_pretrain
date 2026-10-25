"""NISAR from-scratch arm. Without NISAR: the v1.3 control modalities only (the control for ``with_nisar.py``).

v1.3 control recipe (claude_findings/control_pixtgt.py) from scratch on the
random-time osm_sampling 100k + osm_recent 100k mix; see common.py.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "claude_findings"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    build_dataloader_config,
    build_model_config,
    build_visualize_config,
)
from common import build_train_module_config as _build_train_module_config  # noqa: E402
from nisar_common import (  # noqa: E402
    build_common_components_fn,
    build_dataset_config,
)
from nisar_common import build_trainer_config as _build_trainer_config  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.train.train_module.latent_mim import (  # noqa: E402
    LatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/nisar/no_nisar.py"
WITH_NISAR = False
# checkpoint_sweep_evals.py (in-loop evals) looks up a module-level
# build_common_components; without it the eval model falls back to the default
# multi-bandset tokenization and checkpoint loading fails.
build_common_components = build_common_components_fn(WITH_NISAR)


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """The v1.3 train module with pixel-resolution MIM targets."""
    return _build_train_module_config(common, pixel_targets=True)


def build_trainer_config(common: CommonComponents):
    """The v1.3 control in-loop evals, re-importing THIS module."""
    return _build_trainer_config(common, MODULE_PATH)


def run() -> None:
    """Run the experiment."""
    main(
        common_components_builder=build_common_components_fn(WITH_NISAR),
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )


if __name__ == "__main__":
    run()

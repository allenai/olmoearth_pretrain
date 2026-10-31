"""Multi-year pretraining arm A3: CaCo-style change-gated weighted InfoNCE between two token-dropout views of each (location, year) on the time-hidden year-chunk encodings.

v1.3 control recipe (claude_findings/control_pixtgt.py) from scratch on the random-time
osm_sampling 100k (monthly, 1 year) + osm_quarterly4y 100k (quarterly, 4 years) mix, plus
the temporal aux loss; see multiyear_common.py.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "claude_findings"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import build_visualize_config  # noqa: E402
from multiyear_common import (  # noqa: E402
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_model_config_fn,
    build_train_module_config_fn,
)
from multiyear_common import build_trainer_config as _build_trainer_config  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/multiyear/gated_contrast.py"
VARIANT = "gated_contrast"
build_model_config = build_model_config_fn(VARIANT)
build_train_module_config = build_train_module_config_fn(VARIANT)
# checkpoint_sweep_evals.py (in-loop evals) looks up the module-level
# build_common_components, build_model_config (which carries the aux head, so the
# checkpoint keys match) and build_train_module_config.


def build_trainer_config(common: CommonComponents):
    """The v1.3 control in-loop evals, re-importing THIS module."""
    return _build_trainer_config(common, MODULE_PATH)


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

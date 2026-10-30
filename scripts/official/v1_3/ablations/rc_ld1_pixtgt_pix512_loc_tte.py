"""v1.3 release recipe + frozen TTE (Tessellating The Earth, MVRL/TTE, 512-d) location input.

See ``loc_common.py``. Trained as ``v1_3_loc_tte``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import loc_common  # noqa: E402

loc_common.LOC_ENCODER = "tte"
loc_common.MODULE_PATH = (
    "scripts/official/v1_3/ablations/rc_ld1_pixtgt_pix512_loc_tte.py"
)

from loc_common import (  # noqa: E402
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_model_config,
    build_train_module_config,
    build_trainer_config,
    build_visualize_config,
)

from olmoearth_pretrain.internal.experiment import main  # noqa: E402


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

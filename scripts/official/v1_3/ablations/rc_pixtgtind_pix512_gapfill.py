"""Gap-fill eval module for ``v1_3_rc_pixtgtind_pix512`` (see ``rc_gapfill.py``): its model, the gap-fill tasks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rc_gapfill import build_gapfill_trainer_config  # noqa: E402
from rc_pixtgtind_pix512 import (  # noqa: E402,F401
    build_common_components,
    build_model_config,
    build_train_module_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents  # noqa: E402

MODULE_PATH = "scripts/official/v1_3/ablations/rc_pixtgtind_pix512_gapfill.py"


def build_trainer_config(common: CommonComponents):
    """The gap-fill tasks, re-importing THIS module in the eval job."""
    return build_gapfill_trainer_config(common, MODULE_PATH)

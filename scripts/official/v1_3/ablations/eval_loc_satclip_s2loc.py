"""Eval-only module: rc_ld1_pixtgt_pix512_loc_satclip.py's model on the S2 student tasks, S2 + window-centre location.

See ``loc_evals.py``. Launched by the session's bin/launch_s2_evals.sh.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loc_evals import set_eval_latlon, with_s2_tasks  # noqa: E402

set_eval_latlon(True)

from rc_ld1_pixtgt_pix512_loc_satclip import (  # noqa: E402
    build_common_components,
    build_model_config,
    build_train_module_config,
)
from rc_ld1_pixtgt_pix512_loc_satclip import (  # noqa: E402
    build_trainer_config as _build_trainer_config,
)

build_trainer_config = with_s2_tasks(_build_trainer_config)

__all__ = [
    "build_common_components",
    "build_model_config",
    "build_train_module_config",
    "build_trainer_config",
]

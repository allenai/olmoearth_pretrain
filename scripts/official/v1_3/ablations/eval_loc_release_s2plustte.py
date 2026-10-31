"""Eval-only module: rc_ld1_pixtgt_pix512.py's S2 student embedding with the frozen TTE location vector appended (post-hoc baseline).

See ``loc_evals.py``. Launched by the session's bin/launch_s2_evals.sh.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loc_evals import (  # noqa: E402
    set_eval_latlon,
    set_posthoc_location,
    with_s2_tasks,
)

set_eval_latlon(True)
set_posthoc_location("concat", "tte")

from rc_ld1_pixtgt_pix512 import (  # noqa: E402
    build_common_components,
    build_model_config,
    build_train_module_config,
)
from rc_ld1_pixtgt_pix512 import (  # noqa: E402
    build_trainer_config as _build_trainer_config,
)

build_trainer_config = with_s2_tasks(_build_trainer_config)

__all__ = [
    "build_common_components",
    "build_model_config",
    "build_train_module_config",
    "build_trainer_config",
]

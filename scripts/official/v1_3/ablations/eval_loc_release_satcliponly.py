"""Eval-only module: the frozen SatCLIP L40 location vector alone as the embedding (post-hoc baseline, kNN tasks; the model is not run).

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
set_posthoc_location("only", "satclip_l40")

from rc_ld1_pixtgt_pix512 import (  # noqa: E402
    build_common_components,
    build_model_config,
    build_train_module_config,
)
from rc_ld1_pixtgt_pix512 import (  # noqa: E402
    build_trainer_config as _build_trainer_config,
)

build_trainer_config = with_s2_tasks(_build_trainer_config, knn_only=True)

__all__ = [
    "build_common_components",
    "build_model_config",
    "build_train_module_config",
    "build_trainer_config",
]

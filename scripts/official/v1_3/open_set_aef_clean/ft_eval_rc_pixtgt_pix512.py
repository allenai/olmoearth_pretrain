"""``TRAIN_SCRIPT_PATH`` wrapper: fine-tune evals (``ft_eval_tasks``) on the v1.3 base.

The base is ``v1_3_rc_pixtgt_pix512`` (trained on gabi's branch; its script is not on
this branch). Its checkpoint, converted with ``convert_legacy_checkpoint.py``, loads
strictly into ``base.py``'s model: the pix512 / pixtgt settings are a dataloader budget
and a train-module loss, not model parameters.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import base as _train  # noqa: E402
from ft_eval_tasks import make_wrapper  # noqa: E402

globals().update(make_wrapper(_train))

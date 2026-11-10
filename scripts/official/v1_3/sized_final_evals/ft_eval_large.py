"""``TRAIN_SCRIPT_PATH`` wrapper: fine-tune evals (``ft_eval_tasks``) on ``v1_3_large_rc_ld1_pixtgt_pix512`` (width 1024, 24 layers).

Model, common components and train module come from the training script
(``ablations/rc_ld1_pixtgt_pix512_large.py``), which loads the raw training checkpoint.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ablations"))

import rc_ld1_pixtgt_pix512_large as _train  # noqa: E402
from ft_eval_tasks import make_wrapper  # noqa: E402

globals().update(make_wrapper(_train))

"""``TRAIN_SCRIPT_PATH`` wrapper: fine-tune evals (``ft_eval_tasks``) on the released v1.3 Base (``v1_3_rc_ld1_pixtgt_pix512``, HF OlmoEarth-v1_3-Base).

Model, common components and train module come from the training script
(``ablations/rc_ld1_pixtgt_pix512.py``), which loads the raw training checkpoint.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ablations"))

import rc_ld1_pixtgt_pix512 as _train  # noqa: E402
from ft_eval_tasks import make_wrapper  # noqa: E402

globals().update(make_wrapper(_train))

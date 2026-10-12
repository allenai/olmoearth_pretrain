"""``TRAIN_SCRIPT_PATH`` wrapper: fine-tune evals (``ft_eval_tasks``) on osm_text_aef_clean."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import open_set_osm_text_aef_clean as _train  # noqa: E402
from ft_eval_tasks import make_wrapper  # noqa: E402

globals().update(make_wrapper(_train))

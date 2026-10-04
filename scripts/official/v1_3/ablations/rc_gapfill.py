"""Gap-fill evals for the rc_pix512 grid: tasks the in-loop evals do not run.

Used as the ``TRAIN_SCRIPT_PATH`` of a one-off checkpoint eval job (one small module
per model, e.g. ``rc_pix512_gapfill.py``): the eval job rebuilds the model from that
module and, with ``OE_LOOP_EVAL_FROM_TRAIN_CONFIG=1``, runs exactly the tasks below
(narrowed further with ``tasks_to_run``). Every task reads the shipped d128 student.

* ps2: the eight AEF-trial tasks + year-aligned PASTIS at patch size 2, one latent per
  pixel (``*_ws16_ps2_*_proj128``, the names the 1-layer runs log in-loop).
* fine-grained: Favyen's ``worldcover_fine_grained`` (S2) and ``pastis_fine_grained``
  (S1+S2) at patch sizes 1, 2 and 4 on a 16x16 window, one latent per pixel. They score
  only small objects (<= 10 px components / <= 50 px parcels) plus a capped ring of
  their border pixels, so they measure sub-patch detail.
"""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import (  # noqa: E402
    STUDENT_LOOP_EVAL_INTERVAL_STEPS,
    aeftrial_loop_eval_tasks,
)
from pure_perceiver_mix import build_mix_trainer_config  # noqa: E402

from olmoearth_pretrain.internal.all_evals import EMBEDDING_EVAL_TASKS  # noqa: E402
from olmoearth_pretrain.internal.experiment import CommonComponents  # noqa: E402

STUDENT_DIM = 128
FINE_GRAINED_TASKS = (
    "worldcover_fine_grained_ws16_ps1",
    "pastis_fine_grained_ws16_ps1_sentinel1_sentinel2",
)
FINE_GRAINED_PATCH_SIZES = (1, 2, 4)


def gapfill_tasks() -> dict:
    """The ps2 AEF + PASTIS tasks and the fine-grained tasks, all on the d128 student."""
    student = {"eval_on_student_registers": True, "eval_student_dim": STUDENT_DIM}
    tasks = {}
    for name, task in aeftrial_loop_eval_tasks(
        STUDENT_LOOP_EVAL_INTERVAL_STEPS
    ).items():
        assert "_ps1_" in name, name
        tasks[name.replace("_ps1_", "_ps2_") + f"_proj{STUDENT_DIM}"] = replace(
            task, patch_size=2, **student
        )
    for base_name in FINE_GRAINED_TASKS:
        for ps in FINE_GRAINED_PATCH_SIZES:
            name = base_name.replace("_ps1", f"_ps{ps}") + f"_proj{STUDENT_DIM}"
            tasks[name] = replace(
                EMBEDDING_EVAL_TASKS[base_name], patch_size=ps, **student
            )
    return tasks


def build_gapfill_trainer_config(common: CommonComponents, module_path: str):
    """The grid's trainer config with its eval set replaced by ``gapfill_tasks``."""
    trainer_config = build_mix_trainer_config(common, module_path)
    trainer_config.callbacks["downstream_evaluator"].tasks = gapfill_tasks()
    return trainer_config

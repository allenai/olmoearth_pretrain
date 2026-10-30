"""S2-only embedding evals for the location-encoder arms (card t-i9widi).

favyen 2026-10-30: the headline comparison is the shipped embedding with S2 + location vs
S2 alone. The tasks are the v1.3 in-loop panel (8 AEF year-aligned balanced-trial tasks
+ year-aligned PASTIS, ws16, patch size 1, the d128 student) with Sentinel-2 as the only
input. Location reaches the eval samples only when ``OE_EVAL_LATLON=norm`` (window
centre, normalized like the pretraining latlon); without it the location arms take
their learned "no location" path. The eval-only modules ``eval_loc_*.py`` pick the
model and set (or not) that variable at import, and are launched by
``checkpoint_sweep_evals.py`` with ``OE_LOOP_EVAL_FROM_TRAIN_CONFIG=1``.
"""

import os
from dataclasses import replace

from base import aeftrial_loop_eval_tasks

from olmoearth_pretrain.data.constants import Modality

S2_ONLY = [Modality.SENTINEL2_L2A.name]
STUDENT_DIM = 128


def s2_student_eval_tasks() -> dict:
    """The in-loop AEF + PASTIS tasks with S2 as the only input, on the d128 student."""
    tasks = {}
    for name, task in aeftrial_loop_eval_tasks(interval_steps=1).items():
        assert "_sentinel1_sentinel2_landsat" in name, name
        new_name = name.replace("_sentinel1_sentinel2_landsat", "_sentinel2")
        tasks[f"{new_name}_proj{STUDENT_DIM}"] = replace(
            task,
            input_modalities=list(S2_ONLY),
            eval_on_student_registers=True,
            eval_student_dim=STUDENT_DIM,
        )
    return tasks


def set_eval_latlon(with_location: bool) -> None:
    """Give eval samples the normalized window-centre latlon, or none."""
    if with_location:
        os.environ["OE_EVAL_LATLON"] = "norm"
    else:
        os.environ.pop("OE_EVAL_LATLON", None)


def with_s2_tasks(build_trainer_config):  # noqa: ANN001, ANN201
    """Wrap a trainer-config builder so its evaluator runs only the S2 tasks."""

    def build(common):  # noqa: ANN001, ANN202
        trainer_config = build_trainer_config(common)
        trainer_config.callbacks["downstream_evaluator"].tasks = s2_student_eval_tasks()
        return trainer_config

    return build

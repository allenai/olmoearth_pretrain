"""Eval-only module: S1-only LP tasks on the v1.3 control (card t-grl7q6, favyen 2026-11-10).

The reference for run C's S1 + NISAR evals: same windows (``nisar_osm_<task>_s1`` datasets), S1 only, tasks elevation
(SRTM), building, canopy height, wind turbine, LFMC (nisar_osm_lfmc_*). Same model / common components as ``ablations/rc_ld1_pixtgt_pix512.py``
so the checkpoint loads (as eval_osm_control.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ablations"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nisar"))

from rc_ld1_pixtgt_pix512 import (  # noqa: E402, F401
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_model_config,
    build_train_module_config,
    build_visualize_config,
)
from rc_ld1_pixtgt_pix512 import build_trainer_config as _build_trainer_config  # noqa: E402
from nisar_osm_evals import nisar_osm_lfmc_eval_tasks, nisar_osm_s1nisar_eval_tasks  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents  # noqa: E402


def build_trainer_config(common: CommonComponents):
    """The control's trainer config with only the S1-only tasks."""
    trainer_config = _build_trainer_config(common)
    evaluator = trainer_config.callbacks["downstream_evaluator"]
    evaluator.tasks = {
        **nisar_osm_s1nisar_eval_tasks(conditions=("s1",)),
        **nisar_osm_lfmc_eval_tasks(conditions=("s1",)),
    }
    return trainer_config

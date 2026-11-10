"""Eval-only module: S1 + NISAR LP tasks on the run C checkpoint (card t-grl7q6, favyen 2026-11-10).

Conditions: S1 + NISAR asc + desc in their own slots (``_s1nisar``) and S1 only (``_s1``), same windows; tasks
elevation (SRTM), building, canopy height, wind turbine, and LFMC (nisar_osm_lfmc_*). Same model / common components as ``nisar_naip_c.py`` so
the checkpoint loads. Launched by checkpoint_sweep_evals.py with OE_LOOP_EVAL_FROM_TRAIN_CONFIG=1.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nisar_naip_c import (  # noqa: E402, F401
    build_common_components,
    build_model_config,
    build_train_module_config,
)
from nisar_naip_c import build_trainer_config as _build_trainer_config  # noqa: E402
from nisar_osm_evals import nisar_osm_lfmc_eval_tasks, nisar_osm_s1nisar_eval_tasks  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents  # noqa: E402


def build_trainer_config(common: CommonComponents):
    """Run C's trainer config with only the S1 (+ NISAR) tasks."""
    trainer_config = _build_trainer_config(common)
    evaluator = trainer_config.callbacks["downstream_evaluator"]
    evaluator.tasks = {
        **nisar_osm_s1nisar_eval_tasks(conditions=("s1nisar", "s1")),
        **nisar_osm_lfmc_eval_tasks(conditions=("s1nisar", "s1")),
    }
    return trainer_config

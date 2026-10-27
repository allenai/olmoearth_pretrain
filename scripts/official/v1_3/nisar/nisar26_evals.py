"""2025-26 input-ablation evals for the NISAR arms (card t-7xgg82).

Five existing eval label sets re-dated to the osm_recent year (12 monthly layers
2025-10-20..2026-10-15), assuming the labels still hold in 2026: WorldCover (the
nisar_land_eval tiles), lcmap_lu, us_trees, descals and surface_fuels. Built by
scripts/create_nisar26_evals.py in the session folder
/weka/dfive-default/agent/favyen/2026-10-23-favyen-nisar-pretrain; registry entries
``nisar26_*``. S2 and S1 (12 months) and NISAR asc + desc (mo09-mo12, the PROVISIONAL
period) are required inputs, so every input set below scores the same windows; Landsat is
optional as in the year-aligned evals.

The question: does S2+S1+NISAR beat S2+S1, and S2+S1+Landsat+NISAR beat S2+S1+Landsat?

- lcmap_lu / us_trees / descals / surface_fuels: the in-loop protocol (center-pixel kNN
  with AEF's balanced trials, ws16 ps1, student location tokens at d128, as in claude_findings/common.py).
- worldcover: the nisar_land_eval protocol (ws16 ps1 LP on student location tokens at
  d128, mIoU), so numbers line up with the earlier land evals.
"""

from dataclasses import replace

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.evals.balanced_trial import BalancedTrialConfig
from olmoearth_pretrain.evals.metrics import EvalMetric
from olmoearth_pretrain.internal.all_evals import (
    _aef_max_trial_cap,
    _aef_ps1_task,
    _pastis_ps1_task,
)
from olmoearth_pretrain.train.callbacks.evaluator_callback import EvalMode

S2 = Modality.SENTINEL2_L2A.name
S1 = Modality.SENTINEL1.name
LANDSAT = Modality.LANDSAT.name
ASC = Modality.NISAR_ASC.name
DESC = Modality.NISAR_DESC.name

POINT_TASKS = ("lcmap_lu", "us_trees", "descals", "surface_fuels")

# name suffix -> input modalities
INPUT_SETS_NO_NISAR = {
    "s2_s1": [S2, S1],
    "s2_s1_landsat": [S2, S1, LANDSAT],
}
INPUT_SETS_WITH_NISAR = {
    "s2_s1": [S2, S1],
    "s2_s1_nisar": [S2, S1, ASC, DESC],
    "s2_s1_landsat": [S2, S1, LANDSAT],
    "s2_s1_landsat_nisar": [S2, S1, LANDSAT, ASC, DESC],
}


def nisar26_eval_tasks(with_nisar: bool) -> dict:
    """Every (task, input set) pair for the arm; WorldCover first."""
    input_sets = INPUT_SETS_WITH_NISAR if with_nisar else INPUT_SETS_NO_NISAR
    tasks = {}
    for suffix, modalities in input_sets.items():
        tasks[f"nisar26_worldcover_ws16_ps1_{suffix}_proj128"] = replace(
            _pastis_ps1_task(modalities, window_size=16),
            dataset="nisar26_worldcover",
            eval_on_student_location_tokens=True,
            eval_student_dim=128,
            primary_metric=EvalMetric.MIOU,
        )
    for task in POINT_TASKS:
        for suffix, modalities in input_sets.items():
            tasks[f"nisar26_{task}_ws16_ps1_{suffix}_knn_proj128"] = replace(
                _aef_ps1_task(
                    f"nisar26_{task}",
                    EvalMode.KNN,
                    window_size=16,
                    input_modalities=modalities,
                ),
                # The registry name's nisar26_ prefix hides the parent's AEF cap.
                balanced_trial=BalancedTrialConfig(cap=_aef_max_trial_cap(task)),
                eval_on_student_location_tokens=True,
                eval_student_dim=128,
            )
    return tasks

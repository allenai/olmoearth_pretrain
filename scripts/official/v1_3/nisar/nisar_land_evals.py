"""NISAR land-cover LP evals for the NISAR arms (card t-7xgg82).

Dataset: registry ``nisar_land_eval`` (1,500 osm_sampling tiles held out from both
pretraining selections, 64x64 at 10 m, four 30-day layers 2026-06-17..10-15, WorldCover
2021 labels; see data/rslearn_dataset_configs/nisar_land_eval/model.yaml). The
``nisar_land_eval_as_s1`` entry feeds NISAR ascending HH/HV through the sentinel1 slot,
so the no_nisar arm can be probed on NISAR too.

Tasks follow the v1.3 student PASTIS convention: ws16 tiles of the stored samples,
student location tokens (d128) at patch size 1, and at patch size 4 with per-pixel
location queries. Run them on a checkpoint with ``launch_nisar_land_evals.sh``
(eval-only modules ``eval_with_nisar.py`` / ``eval_no_nisar.py``).
"""

from dataclasses import replace

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.evals.metrics import EvalMetric
from olmoearth_pretrain.internal.all_evals import _pastis_ps1_task

S1 = Modality.SENTINEL1.name
ASC = Modality.NISAR_ASC.name
DESC = Modality.NISAR_DESC.name

# name suffix -> (registry dataset, input modalities)
INPUT_SETS_NO_NISAR = {
    "s1": ("nisar_land_eval", [S1]),
    "nisar_asc_as_s1": ("nisar_land_eval_as_s1", [S1]),
}
INPUT_SETS_WITH_NISAR = {
    **INPUT_SETS_NO_NISAR,
    "nisar": ("nisar_land_eval", [ASC, DESC]),
    "s1_nisar": ("nisar_land_eval", [S1, ASC, DESC]),
}


def nisar_land_eval_tasks(with_nisar: bool) -> dict:
    """ws16 ps1 and ps4 student (d128) LP tasks for each input set of the arm."""
    input_sets = INPUT_SETS_WITH_NISAR if with_nisar else INPUT_SETS_NO_NISAR
    tasks = {}
    for suffix, (dataset, modalities) in input_sets.items():
        base = replace(
            _pastis_ps1_task(modalities, window_size=16),
            dataset=dataset,
            eval_on_student_location_tokens=True,
            eval_student_dim=128,
            primary_metric=EvalMetric.MIOU,
        )
        tasks[f"nisar_land_ws16_ps1_{suffix}_proj128"] = base
        tasks[f"nisar_land_ws16_ps4_{suffix}_proj128"] = replace(
            base, patch_size=4, location_query_patch_size=1
        )
    return tasks

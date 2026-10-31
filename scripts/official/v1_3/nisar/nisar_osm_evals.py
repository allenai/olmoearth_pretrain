"""NISAR-slot vs S1-slot LP evals on held-out OSM tasks (card t-grl7q6, favyen 2026-10-30).

Datasets: registry ``nisar_osm_<task>`` (NISAR ascending HH/HV mo09-mo12 through the nisar_asc slot) and
``nisar_osm_<task>_as_s1`` (the same layers through the sentinel1 vv/vh slot), built by
nisar_osm_evals/bin/build_windows.py in /weka/dfive-default/agent/favyen/2026-10-27-favyen-nisar-naip-pretrain:
64x64 crops at 10 m of osm_sampling tiles held out from both NISAR pretraining selections.
- wind_turbine / storage_tank / building: binary per-pixel labels from the tiles' OSM features drawn as the
  pretraining openstreetmap_raster (turbines dilated to 3x3); primary metric = F1 of class 1.
- canopy_height: per-pixel regression on WRI canopy height (m); primary metric = RMSE.
Conditions (favyen): with_nisar NISAR slot, with_nisar S1 slot, v1.3 control S1 slot. Same windows for all.
Protocol: the v1.3 student PASTIS LP convention (ws16 tiles of the stored 64x64 samples, student location tokens
d128), at patch size 1 and at patch size 4 with per-pixel location queries, as nisar_land_evals.py.
This copy (control branch, pre-AttentionPooler API at 9f5a25b02) uses eval_on_student_registers; the control's
model config sets eval_latent_stride=1, the equivalent of location_query_patch_size=1 on main's API.
"""

from dataclasses import replace

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.evals.metrics import EvalMetric
from olmoearth_pretrain.internal.all_evals import _pastis_ps1_task

S1 = Modality.SENTINEL1.name
ASC = Modality.NISAR_ASC.name
TASKS = ("wind_turbine", "storage_tank", "building", "canopy_height")


def nisar_osm_eval_tasks(native_slot: bool, only: tuple[str, ...] = TASKS) -> dict:
    """ws16 ps1 / ps4 student (d128) LP tasks: S1 slot always, NISAR slot if the model has one."""
    slots = {"nisar_as_s1": ("_as_s1", [S1])}
    if native_slot:
        slots["nisar_slot"] = ("", [ASC])
    tasks = {}
    for task in only:
        for suffix, (ds_suffix, modalities) in slots.items():
            base = replace(
                _pastis_ps1_task(modalities, window_size=16),
                dataset=f"nisar_osm_{task}{ds_suffix}",
                eval_on_student_registers=True,
                eval_student_dim=128,
                primary_metric=EvalMetric.RMSE if task == "canopy_height" else EvalMetric.CLASS_F1,
                primary_metric_class=None if task == "canopy_height" else 1,
            )
            tasks[f"nisar_osm_{task}_ws16_ps1_{suffix}_proj128"] = base
            tasks[f"nisar_osm_{task}_ws16_ps4_{suffix}_proj128"] = replace(
                base, patch_size=4
            )
    return tasks

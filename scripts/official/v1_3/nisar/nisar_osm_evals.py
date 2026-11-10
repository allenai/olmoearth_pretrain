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
                eval_on_student_location_tokens=True,
                eval_student_dim=128,
                primary_metric=EvalMetric.RMSE if task == "canopy_height" else EvalMetric.CLASS_F1,
                primary_metric_class=None if task == "canopy_height" else 1,
            )
            tasks[f"nisar_osm_{task}_ws16_ps1_{suffix}_proj128"] = base
            tasks[f"nisar_osm_{task}_ws16_ps4_{suffix}_proj128"] = replace(
                base, patch_size=4, location_query_patch_size=1
            )
    return tasks


# 2026-11-10 (favyen): S1 + NISAR vs S1 only. Datasets nisar_osm_<task>_{s1nisar,s1} share one window set per task
# (nisar_osm_evals/bin/register_s1nisar.py): all 4 S1 RTC months, plus NISAR asc and/or desc fed when present.
DESC = Modality.NISAR_DESC.name
S1NISAR_TASKS = ("elevation", "building", "canopy_height", "wind_turbine")
S1NISAR_CONDITIONS = {"s1nisar": [S1, ASC, DESC], "s1": [S1]}
REGRESSION_TASKS = ("canopy_height", "elevation")


def nisar_osm_s1nisar_eval_tasks(conditions: tuple[str, ...], only: tuple[str, ...] = S1NISAR_TASKS) -> dict:
    """ws16 ps1 / ps4 student (d128) LP tasks on the S1 (+ NISAR) datasets, same protocol as nisar_osm_eval_tasks."""
    tasks = {}
    for task in only:
        regression = task in REGRESSION_TASKS
        for cond in conditions:
            base = replace(
                _pastis_ps1_task(S1NISAR_CONDITIONS[cond], window_size=16),
                dataset=f"nisar_osm_{task}_{cond}",
                eval_on_student_location_tokens=True,
                eval_student_dim=128,
                primary_metric=EvalMetric.RMSE if regression else EvalMetric.CLASS_F1,
                primary_metric_class=None if regression else 1,
            )
            tasks[f"nisar_osm_{task}_ws16_ps1_{cond}_proj128"] = base
            tasks[f"nisar_osm_{task}_ws16_ps4_{cond}_proj128"] = replace(base, patch_size=4, location_query_patch_size=1)
    return tasks


# 2026-11-10: LFMC in the NISAR era (lfmc_nisar_2026, FEMS samples 2026-06..10; view nisar_osm_evals/s1nisar_lfmc via
# nisar_osm_evals/bin/build_lfmc_view.py). Label = LFMC % at the center pixel only; the 63x63 window is center-cropped
# to 32x32 (lfmc_woody_3k convention) and the per-pixel probe is scored on that one pixel. S1: all 4 x 30 d periods;
# NISAR asc/desc: whatever periods exist (missing ones are zero placeholders the loader masks MISSING). Same student
# features as the OSM tasks; two probe LRs (pick by val).
LFMC_PROBE_LRS = {"lr1e-1": 0.1, "lr1e-2": 0.01}


def nisar_osm_lfmc_eval_tasks(conditions: tuple[str, ...]) -> dict:
    """ws32 ps1 / ps4 student (d128) LP tasks on nisar_osm_lfmc_{cond}, derived from the elevation tasks above."""
    tasks = {}
    for cond in conditions:
        refs = nisar_osm_s1nisar_eval_tasks((cond,), only=("elevation",))
        for ps in (1, 4):
            ref = refs[f"nisar_osm_elevation_ws16_ps{ps}_{cond}_proj128"]
            for lr_name, lr in LFMC_PROBE_LRS.items():
                tasks[f"nisar_osm_lfmc_ws32_ps{ps}_{cond}_{lr_name}_proj128"] = replace(
                    ref,
                    dataset=f"nisar_osm_lfmc_{cond}",
                    window_size=32,
                    tile_samples=False,
                    embedding_batch_size=32,
                    probe_batch_size=8,
                    num_workers=8,
                    probe_lr=lr,
                )
    return tasks

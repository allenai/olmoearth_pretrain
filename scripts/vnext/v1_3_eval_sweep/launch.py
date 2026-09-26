"""Launch the v1.3 eval sweep: one single-GPU Beaker job per (task bundle, checkpoint).

Each job runs ``checkpoint_sweep_evals.py`` on one checkpoint with the tasks of one
bundle from ``sweep.py``, and logs to its own W&B run in a shared group; every metric
carries ``checkpoint_step``. Bundles are packed from rough per-task runtimes so a job
stays well inside the launcher's 8h ``min_runtime``. Tasks with no clean run on record
(revived, ported) get bundles of their own, since one crashing task ends its job.

    python scripts/vnext/v1_3_eval_sweep/launch.py --steps 640000 --dry-run
    python scripts/vnext/v1_3_eval_sweep/launch.py --steps 0,20000,...

Run from the repo root with the repo's venv. The launch ledger, per-job launch logs
and the bundle assignment go to ``--state-dir`` (outside the repo: Beaker launches
refuse a dirty tree). Already-launched (bundle, step) pairs in the ledger are
skipped, so a partial launch can be resumed by re-running.
"""

import argparse
import json
import os
import subprocess  # nosec
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CHECKPOINT_DIR = (
    "/weka/dfive-default/olmoearth_pretrain/checkpoints/gabrielt/"
    "regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsamp_psuniform_stunorm_mlpgram1"
)
MODULE_PATH = "scripts/vnext/v1_3_eval_sweep/sweep.py"
WANDB_PROJECT = "2026_09_25_v1_3_eval_sweep"
WANDB_GROUP = "v13_eval_sweep"
RUN_PREFIX = "v13sweep"
BUNDLE_SECONDS = 4 * 3600

# Median seconds per task on one GPU. Measured: the 2026-06 checkpoint sweep
# (W&B bw7i04bu) and this run's in-loop evals; everything else is a guess.
MEASURED_SECONDS = {
    "gb2_spacenet2": 4027,
    "gb2_cloudsen12": 3570,
    "gb2_caffe": 3115,
    "gb2_flair2": 2912,
    "gb2_spacenet7": 2423,
    "tolbi_crop": 1032,
    "geo_ecosystem_annual_test": 933,
    "gb2_kuro_siwo": 897,
    "m_sa_crop_type": 891,
    "gb2_biomassters": 782,
    "gb2_fotw": 736,
    "gb2_burn_scars": 438,
    "nigeria_settlement": 417,
    "nandi_crop_map": 409,
    "forest_loss_driver": 343,
    "pastis128_sentinel1_sentinel2": 296,
    "m_cashew_plant": 296,
    "awf_lulc_map": 281,
    "m_forestnet": 251,
    "pastis_sentinel2": 243,
    "pastis_sentinel1_sentinel2": 220,
    "pastis128_sentinel2": 196,
    "m_bigearthnet": 186,
    "gb2_treesatai": 175,
    "gb2_benv2": 172,
    "m_so2sat": 162,
    "m_brick_kiln": 158,
    "pastis_sentinel1": 149,
    "pastis128_sentinel1": 149,
    "yemen_crop": 124,
    "sen1floods11": 109,
    "mados": 86,
    "pastis_sentinel2_embed_diag": 45,
    "m_eurosat": 42,
    "m_eurosat_embed_diag": 8,
    "pastis_year_aligned_ws16_ps1_sentinel1_sentinel2_landsat": 3700,
}
# Year-aligned kNN times from the in-loop evals; their LP twins embed the same data.
YEAR_ALIGNED_SECONDS = {
    "africa_crop_mask": 400,
    "canada_crops_coarse": 1300,
    "canada_crops_fine": 1400,
    "descals": 1300,
    "ethiopia_crops": 300,
    "glance": 2500,
    "lcmap_lu": 2700,
    "us_trees": 4000,
}
GUESSED_SECONDS = {
    "pretrain_srtm_regression_sentinel2_l2a_sentinel1": 1367,
    "pretrain_srtm_regression_geo_sentinel2_l2a_sentinel1": 1334,
    "oil_spill_detection": 3000,
    "burnrisk_8d_nbac": 1500,
    "swisscrop_sentinel2": 2500,
    "planteur_2019_probe_sentinel2": 1500,
    "pastis_ws16_ps1_sentinel2_pretrain_export": 2000,
    "pastis_ws16_ps1_sentinel1_sentinel2_pretrain_export": 3000,
}
DEFAULT_SECONDS = 700
ISOLATED = {  # task -> bundle label; no clean run of these on record
    "burnrisk_8d_nbac": "revived",
    "oil_spill_detection": "oilspill",
    "mapbiomas_3k_sparse": "revived",
    "canada_wildfire_sat_eval_split": "revived",
    "breizhcrops": "revived",
    "swisscrop_sentinel2": "ported",
    "planteur_2019_probe_sentinel2": "ported",
}


def estimate_seconds(task: str) -> int:
    """Rough single-GPU runtime of one task on one checkpoint."""
    if task in MEASURED_SECONDS:
        return MEASURED_SECONDS[task]
    if task in GUESSED_SECONDS:
        return GUESSED_SECONDS[task]
    for name, seconds in YEAR_ALIGNED_SECONDS.items():
        if task.startswith(f"{name}_year_aligned"):
            return seconds
        if task.startswith(f"{name}_ws16_ps1"):  # S2-only: about half the tokens
            return seconds // 2
    if task.startswith("pretrain_"):
        return 700
    return DEFAULT_SECONDS


def pack_bundles(tasks: list[str]) -> dict[str, list[str]]:
    """Greedy longest-first packing into bundles of about BUNDLE_SECONDS."""
    bundles: dict[str, list[str]] = {}
    for task in tasks:
        if task in ISOLATED:
            bundles.setdefault(ISOLATED[task], []).append(task)
    loads: list[tuple[int, list[str]]] = []
    rest = sorted(
        (t for t in tasks if t not in ISOLATED), key=estimate_seconds, reverse=True
    )
    for task in rest:
        seconds = estimate_seconds(task)
        fits = [b for b in loads if b[0] + seconds <= BUNDLE_SECONDS]
        if fits:
            target = min(fits, key=lambda b: b[0])
            loads.remove(target)
            loads.append((target[0] + seconds, target[1] + [task]))
        else:
            loads.append((seconds, [task]))
    loads.sort(key=lambda b: -b[0])
    for i, (_, names) in enumerate(loads):
        bundles[f"b{i:02d}"] = names
    return bundles


def launch_command(bundle: str, tasks: list[str], step: int, args: argparse.Namespace):
    """Environment and argv for one checkpoint_sweep_evals.py launch."""
    env = dict(
        os.environ,
        TRAIN_SCRIPT_PATH=MODULE_PATH,
        CHECKPOINT_DIR=CHECKPOINT_DIR,
        CHECKPOINT_STEPS=str(step),
        OE_LOOP_EVAL_FROM_TRAIN_CONFIG="1",
    )
    run_name = f"{RUN_PREFIX}_{bundle}_step{step}"
    argv = [
        sys.executable,
        "olmoearth_pretrain/internal/checkpoint_sweep_evals.py",
        "launch_evaluate",
        run_name,
        args.clusters[0],
        f"--launch.priority={args.priority}",
        "--launch.num_gpus=1",
        "--launch.task_name=eval",
        f"--launch.clusters=[{','.join(args.clusters)}]",
        "--trainer.no_checkpoints=False",
        "--trainer.max_duration.value=10000000",
        "--trainer.max_duration.unit=steps",
        f"--trainer.callbacks.wandb.project={WANDB_PROJECT}",
        f"--trainer.callbacks.wandb.group={WANDB_GROUP}",
        f"--trainer.callbacks.downstream_evaluator.tasks_to_run=[{','.join(tasks)}]",
    ]
    return run_name, env, argv


def main() -> None:
    """Pack bundles and launch every (bundle, step) job not yet in the ledger."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--steps", required=True, help="comma-separated steps")
    parser.add_argument("--bundles", default=None, help="comma-separated bundle ids")
    parser.add_argument("--clusters", default="ai2/saturn", help="comma-separated")
    parser.add_argument("--priority", default="high")
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.clusters = args.clusters.split(",")
    args.state_dir.mkdir(parents=True, exist_ok=True)

    from olmoearth_pretrain.internal.all_evals import load_user_module

    task_names = sorted(load_user_module(MODULE_PATH).SWEEP_TASKS)
    bundles = pack_bundles(task_names)
    (args.state_dir / "bundles.json").write_text(json.dumps(bundles, indent=1) + "\n")
    for name, tasks in bundles.items():
        hours = sum(map(estimate_seconds, tasks)) / 3600
        print(f"{name}: {len(tasks)} tasks, ~{hours:.1f} h")
    if args.bundles:
        bundles = {b: bundles[b] for b in args.bundles.split(",")}
    steps = [int(s) for s in args.steps.split(",")]

    ledger_path = args.state_dir / "launched.jsonl"
    done = set()
    if ledger_path.exists():
        for line in ledger_path.read_text().splitlines():
            done.add(json.loads(line)["run_name"])
    jobs = [
        launch_command(b, tasks, step, args)
        for step in steps
        for b, tasks in bundles.items()
    ]
    jobs = [j for j in jobs if j[0] not in done]
    print(f"{len(jobs)} jobs to launch")
    if args.dry_run:
        print(" ".join(jobs[0][2]) if jobs else "nothing to launch")
        return

    log_dir = args.state_dir / "launch_logs"
    log_dir.mkdir(exist_ok=True)

    def run(job):
        run_name, env, argv = job
        log = log_dir / f"{run_name}.log"
        with log.open("w") as f:
            code = subprocess.call(argv, env=env, stdout=f, stderr=subprocess.STDOUT)  # nosec
        return run_name, code

    with ThreadPoolExecutor(args.parallel) as pool:
        for run_name, code in pool.map(run, jobs):
            print(run_name, "ok" if code == 0 else f"FAILED ({code})", flush=True)
            if code == 0:
                with ledger_path.open("a") as f:
                    f.write(json.dumps({"run_name": run_name}) + "\n")


if __name__ == "__main__":
    main()

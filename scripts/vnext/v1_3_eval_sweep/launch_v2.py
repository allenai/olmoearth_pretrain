"""Launch the v1.3 eval sweep v2 re-runs: only the (task, checkpoint) pairs v2 changes.

v2 (2026-10-16) fixes evals the v1 sweep (``launch.py``, results in ``RESULTS_DIR``)
got wrong, and re-runs only those tasks, on the v1 checkpoint grid of each task
(LIGHT: every 20k, 35 checkpoints; HEAVY: every 40k, 18). Results go to
``RESULTS_DIR_V2/step{N}/{task}.json``; v1 results are never overwritten, and
``publish_v2.py`` merges the two (v2 wins) into one W&B run.

Each job is one single-GPU Beaker job over a group of checkpoints; at every
checkpoint it runs each selected task whose grid contains that step (done tasks
are skipped, so a preempted job resumes where it stopped). Steps are spread over
``--jobs`` jobs by estimated cost.

    python scripts/vnext/v1_3_eval_sweep/launch_v2.py --tasks main --jobs 14 \
        --state-dir DIR --dry-run

``--results-dir``/``--dump-dir``/``--dump-tasks`` run diagnostics instead: results
go elsewhere and the probe inputs of the dump tasks are saved
(``SWEEP_DUMP_DIR``, see checkpoint_sweep_evals.py).
"""

import argparse
import json
import os
import re
import subprocess  # nosec
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from launch import CHECKPOINT_DIR, MODULE_PATH, RESULTS_DIR  # noqa: F401

RESULTS_DIR_V2 = (
    "/weka/dfive-default/olmoearth_pretrain/checkpoints/joer/v13_eval_sweep_v2"
)
LIGHT = [0, *range(20000, 660001, 20000), 667200]
HEAVY = [0, *range(40000, 640001, 40000), 667200]

# Task -> (v1 grid, rough warm seconds per checkpoint). What v2 changes per task:
RERUN = {
    # S1 converted to dB twice (rslearn Sentinel1ToDecibels + loader convert_to_db):
    # every S1 pixel sat at the -100 dB floor, so the embeddings were constant.
    "oil_spill_detection": (LIGHT, 2400),
    "lfmc_woody_3k_s1_s2": (HEAVY, 1800),
    # Stored S1 is GRD amplitude DN, not sigma0: dataset-stats normalization.
    "small_sentinel1_vessel_length": (LIGHT, 120),
    "small_sentinel1_vessel_type": (LIGHT, 120),
    # Probe configs fixed after the v2 diagnostics (sweep.py PROBE_FIXES); the LFMC
    # S1+S2 task above gets both its dB fix and the LR fix.
    "lfmc_woody_3k": (LIGHT, 400),
    "canada_wildfire_sat_eval_split": (LIGHT, 120),
}
TASK_GROUPS = {
    "main": [
        "oil_spill_detection",
        "lfmc_woody_3k_s1_s2",
        "small_sentinel1_vessel_length",
        "small_sentinel1_vessel_type",
    ],
    "probe": ["lfmc_woody_3k", "lfmc_woody_3k_s1_s2", "canada_wildfire_sat_eval_split"],
}
RUN_PREFIX = "agent-joer-v13evals2"
# Diagnostics runs (results outside RESULTS_DIR_V2) may use any sweep task.
DIAG_DEFAULT = (LIGHT, 1200)


def grid_cost(task: str) -> tuple[list[int], int]:
    """The task's checkpoint grid and estimated seconds per checkpoint."""
    return RERUN.get(task, DIAG_DEFAULT)


def plan(tasks: list[str], steps: list[int], n_jobs: int) -> list[dict]:
    """Spread checkpoints over n_jobs jobs, balancing estimated seconds.

    A job's steps share one task list (checkpoint_sweep_evals runs every task at
    every step it gets), so steps are first grouped by which tasks they need.
    """
    groups: dict[tuple[str, ...], list[int]] = {}
    for step in steps:
        need = tuple(t for t in tasks if step in grid_cost(t)[0])
        if need:
            groups.setdefault(need, []).append(step)
    cost = {need: sum(grid_cost(t)[1] for t in need) for need in groups}
    total = sum(cost[need] * len(s) for need, s in groups.items())
    jobs = []
    for need, group_steps in groups.items():
        share = cost[need] * len(group_steps) / total
        k = max(1, min(len(group_steps), round(share * n_jobs)))
        buckets: list[list[int]] = [[] for _ in range(k)]
        for i, step in enumerate(sorted(group_steps)):
            buckets[i % k].append(step)
        for b in buckets:
            jobs.append(
                {"tasks": list(need), "steps": b, "seconds": cost[need] * len(b)}
            )
    return jobs


def launch_command(job: dict, name: str, args: argparse.Namespace):
    """Environment and argv for one checkpoint_sweep_evals.py launch."""
    env = dict(
        os.environ,
        TRAIN_SCRIPT_PATH=MODULE_PATH,
        CHECKPOINT_DIR=CHECKPOINT_DIR,
        CHECKPOINT_STEPS=",".join(str(s) for s in job["steps"]),
        OE_LOOP_EVAL_FROM_TRAIN_CONFIG="1",
        SWEEP_RESULTS_DIR=args.results_dir,
    )
    if args.dump_dir:
        env["SWEEP_DUMP_DIR"] = args.dump_dir
        env["SWEEP_DUMP_TASKS"] = args.dump_tasks or ",".join(job["tasks"])
    argv = [
        sys.executable,
        "olmoearth_pretrain/internal/checkpoint_sweep_evals.py",
        "launch_evaluate",
        name,
        args.clusters[0],
        f"--launch.priority={args.priority}",
        f"--launch.min_runtime={args.min_runtime}",
        "--launch.num_gpus=1",
        "--launch.task_name=eval",
        f"--launch.clusters=[{','.join(args.clusters)}]",
        "--trainer.no_checkpoints=False",
        "--trainer.max_duration.value=10000000",
        "--trainer.max_duration.unit=steps",
        # Jobs only write result files; publish_v2.py is the single W&B writer.
        "--trainer.callbacks.wandb.enabled=False",
        f"--trainer.callbacks.downstream_evaluator.tasks_to_run=[{','.join(job['tasks'])}]",
    ]
    return env, argv


def main() -> None:
    """Plan the jobs and launch every one not yet in the ledger."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tasks", required=True, help="group name or comma list")
    parser.add_argument("--steps", default=None, help="comma list (default: grids)")
    parser.add_argument("--jobs", type=int, required=True)
    parser.add_argument("--tag", default="", help="run-name tag, e.g. diag")
    parser.add_argument("--date", default="20261016")
    parser.add_argument("--clusters", default="ai2/saturn,ai2/jupiter,ai2/ceres")
    parser.add_argument("--priority", default="high")
    parser.add_argument("--min-runtime", default="2h")
    parser.add_argument("--results-dir", default=RESULTS_DIR_V2)
    parser.add_argument("--dump-dir", default=None)
    parser.add_argument("--dump-tasks", default=None)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.clusters = args.clusters.split(",")
    args.state_dir.mkdir(parents=True, exist_ok=True)

    tasks = TASK_GROUPS.get(args.tasks) or args.tasks.split(",")
    unknown = set(tasks) - set(RERUN)
    if args.results_dir == RESULTS_DIR_V2:
        assert not unknown, f"not v2 re-run tasks: {sorted(unknown)}"
    steps = [int(s) for s in args.steps.split(",")] if args.steps else LIGHT
    jobs = plan(tasks, steps, args.jobs)
    for i, job in enumerate(jobs):
        job["name"] = f"{RUN_PREFIX}-{args.tag}{i:02d}-{args.date}"
        print(
            f"{job['name']}: {len(job['steps'])} ckpts x {len(job['tasks'])} tasks,"
            f" ~{job['seconds'] / 3600:.1f} h: {job['steps']} {job['tasks']}"
        )
    print(f"{len(jobs)} jobs, ~{sum(j['seconds'] for j in jobs) / 3600:.0f} GPU-h")

    ledger_path = args.state_dir / "launched.jsonl"
    done = set()
    if ledger_path.exists():
        done = {json.loads(x)["name"] for x in ledger_path.read_text().splitlines()}
    todo = [j for j in jobs if j["name"] not in done]
    print(f"{len(todo)} jobs to launch")
    if args.dry_run:
        if todo:
            env, argv = launch_command(todo[0], todo[0]["name"], args)
            print(" ".join(argv))
            print({k: env[k] for k in env if k.startswith(("SWEEP_", "CHECKPOINT_"))})
        return

    log_dir = args.state_dir / "launch_logs"
    log_dir.mkdir(exist_ok=True)

    def run(job):
        env, argv = launch_command(job, job["name"], args)
        log = log_dir / f"{job['name']}.log"
        with log.open("w") as f:
            code = subprocess.call(argv, env=env, stdout=f, stderr=subprocess.STDOUT)  # nosec
        ids = re.findall(r"beaker\.org/ex/(\w+)", log.read_text())
        return job, code, ids[-1] if ids else None

    with ThreadPoolExecutor(args.parallel) as pool:
        for job, code, exp_id in pool.map(run, todo):
            print(
                job["name"],
                "ok" if code == 0 else f"FAILED ({code})",
                exp_id,
                flush=True,
            )
            if code == 0:
                with ledger_path.open("a") as f:
                    f.write(json.dumps({**job, "experiment": exp_id}) + "\n")


if __name__ == "__main__":
    main()

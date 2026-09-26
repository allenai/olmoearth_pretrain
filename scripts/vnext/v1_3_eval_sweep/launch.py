"""Launch the v1.3 eval sweep: one single-GPU Beaker job per (task bundle, checkpoint).

Each job runs ``checkpoint_sweep_evals.py`` on one checkpoint with the tasks of one
bundle from ``sweep.py``. Jobs do not log to W&B: each (task, step) result is written
to ``RESULTS_DIR/step{N}/{task}.json`` (``SWEEP_RESULTS_DIR``), done tasks are
skipped on restart, and ``publish.py`` merges every result into one W&B run.

Bundles are packed longest-first to about ``BUNDLE_SECONDS`` from the runtimes
measured on step 640000 (``phase_a_seconds.json``; guesses for the rest). Tasks
with no clean run on record get bundles of their own, so a crash in one of them
cannot stop the tested tasks.

    python scripts/vnext/v1_3_eval_sweep/launch.py --steps 640000 --state-dir DIR --dry-run

Run from the repo root with the repo's venv. The launch ledger and per-job launch
logs go to ``--state-dir`` (outside the repo: Beaker launches refuse a dirty tree);
(bundle, step) pairs already in the ledger are not relaunched.
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
RESULTS_DIR = "/weka/dfive-default/olmoearth_pretrain/checkpoints/joer/v13_eval_sweep"
MODULE_PATH = "scripts/vnext/v1_3_eval_sweep/sweep.py"
RUN_PREFIX = "v13evals"
BUNDLE_SECONDS = 4 * 3600
DEFAULT_SECONDS = 1800
MEASURED_SECONDS = json.loads(
    (Path(__file__).resolve().parent / "phase_a_seconds.json").read_text()
)


def pack(tasks: list[str], prefix: str) -> dict[str, list[str]]:
    """Greedy longest-first packing into bundles of about BUNDLE_SECONDS."""
    seconds = {t: MEASURED_SECONDS.get(t, DEFAULT_SECONDS) for t in tasks}
    loads: list[tuple[int, list[str]]] = []
    for task in sorted(tasks, key=seconds.__getitem__, reverse=True):
        fits = [b for b in loads if b[0] + seconds[task] <= BUNDLE_SECONDS]
        if fits:
            target = min(fits, key=lambda b: b[0])
            loads.remove(target)
            loads.append((target[0] + seconds[task], target[1] + [task]))
        else:
            loads.append((seconds[task], [task]))
    loads.sort(key=lambda b: -b[0])
    return {f"{prefix}{i:02d}": names for i, (_, names) in enumerate(loads)}


def launch_command(bundle: str, tasks: list[str], step: int, args: argparse.Namespace):
    """Environment and argv for one checkpoint_sweep_evals.py launch."""
    env = dict(
        os.environ,
        TRAIN_SCRIPT_PATH=MODULE_PATH,
        CHECKPOINT_DIR=CHECKPOINT_DIR,
        CHECKPOINT_STEPS=str(step),
        OE_LOOP_EVAL_FROM_TRAIN_CONFIG="1",
        SWEEP_RESULTS_DIR=RESULTS_DIR,
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
        "--trainer.callbacks.wandb.enabled=False",
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

    tasks = sorted(load_user_module(MODULE_PATH).SWEEP_TASKS)
    bundles = {
        **pack([t for t in tasks if t in MEASURED_SECONDS], "b"),
        **pack([t for t in tasks if t not in MEASURED_SECONDS], "u"),
    }
    (args.state_dir / "bundles.json").write_text(json.dumps(bundles, indent=1) + "\n")
    total = 0
    for name, names in bundles.items():
        seconds = sum(MEASURED_SECONDS.get(t, DEFAULT_SECONDS) for t in names)
        total += seconds
        print(f"{name}: {len(names)} tasks, ~{seconds / 3600:.1f} h")
    print(f"~{total / 3600:.0f} GPU-h per checkpoint")
    if args.bundles:
        bundles = {b: bundles[b] for b in args.bundles.split(",")}
    steps = [int(s) for s in args.steps.split(",")]

    ledger_path = args.state_dir / "launched.jsonl"
    done = set()
    if ledger_path.exists():
        for line in ledger_path.read_text().splitlines():
            done.add(json.loads(line)["run_name"])
    jobs = [
        launch_command(b, names, step, args)
        for step in steps
        for b, names in bundles.items()
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

"""Submit the `_cloudless_l8pixmask` embedding-eval sweep.

The masking grid on the input-ablation page is S2-side cleaner x Landsat-side
filter. Everything existed except the `cloudless` row's two masked-Landsat cells,
because the pixel mask was registered when `sclmask` was the leading S2-side
cleaner. This sweep fills them for the two arms that carry the full ladder:
`cand_ndvi` (the base d128 candidate) and the distilled `lin_sup768_w1_d128`.

Shards, one Beaker experiment per (shard, probe/learning rate):

    {arm} x {S2+L8, S1+S2+L8}   8 LP learning rates + 1 KNN =  9 jobs each
    {arm} x PASTIS              LP-only, both variants      =  8 jobs each

so 26 jobs per arm, 52 in total.

Two ways to run it:

    # submit from here -- one beaker create per job, several minutes serially
    python3 scripts/tools/launch_cloudless_pixmask_sweep.py --go

    # hand the submitting itself to a 0-GPU Beaker job and return immediately
    python3 scripts/tools/launch_cloudless_pixmask_sweep.py --as_beaker_job

The second form is the reason this file lives in the repo rather than a
scratchpad: the submitter job clones the repo at the launching commit's SHA
(`GIT_REF`), so the script has to be *in* that commit, and the commit has to be
pushed. The job inherits BEAKER_TOKEN as a secret, which is what lets it create
the eval experiments in turn -- the same mechanism in-loop evals use
(`loop_eval_launch.py`). Eval jobs spawned from inside it pin that same SHA.

Re-running is safe: a name collision comes back as ExperimentConflict and is
counted as already-submitted rather than retried.
"""

import argparse
import re
import subprocess  # nosec
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from olmoearth_pretrain.internal.common import build_launch_config  # noqa: E402
from olmoearth_pretrain.internal.embedding_eval_sweep import (  # noqa: E402
    build_commands,
)

PROJECT = "20260807_perceiver_updated_evals"
CLUSTERS = ["ai2/jupiter", "ai2/ceres", "ai2/saturn"]

# Smallest-first, matching registry insertion order (commit 3a936aaeb): a shard
# that dies partway through has then already produced the cheap datasets.
DATASETS = [
    "ethiopia_crops",
    "africa_crop_mask",
    "canada_crops_fine",
    "canada_crops_coarse",
    "descals",
    "lcmap_lu",
    "glance",
    "us_trees",
]

VARIANTS = {
    "clpx1": "sentinel2_landsat_cloudless_l8pixmask",
    "clpx2": "sentinel1_sentinel2_landsat_cloudless_l8pixmask",
}

ARMS = {
    "cand_ndvi": {
        "checkpoint_path": (
            "/weka/dfive-default/helios/checkpoints/gabrielt/"
            "regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsamp_psuniform/"
            "step667200"
        ),
        "module_path": (
            "scripts/official/v1_2/"
            "regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform.py"
        ),
        # Base d128 reads its registers directly; nothing to project.
        "projected": False,
    },
    "lin_sup768_w1_d128": {
        "checkpoint_path": (
            "/weka/dfive-default/olmoearth_pretrain/checkpoints/gabrielt/"
            "regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsamp_psuniform/step667200"
        ),
        "module_path": (
            "scripts/official/v1_2/"
            "regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform.py"
        ),
        # d768 parent: the student embedding is the 128-d projection, so every
        # task has to be told to read it instead of the raw registers.
        "projected": True,
    },
}


def _shard_commands(arm: str, tag: str, tasks: list[str]) -> list[str]:
    """Build the per-learning-rate and KNN commands for one shard."""
    cfg = ARMS[arm]
    args = argparse.Namespace(
        cluster=CLUSTERS[0],
        checkpoint_path=cfg["checkpoint_path"],
        module_path=cfg["module_path"],
        model=None,
        model_name=f"{arm}_{tag}",
        project_name=PROJECT,
        task_names=",".join(tasks),
        select_best_val=False,
        priority="urgent",
        window_size=None,
        dry_run=False,
        no_balanced_trials=True,
        balanced_trial_max_folds=None,
        balanced_trial_draw_pool=None,
        balanced_trial_eval_split=None,
    )
    extra = [f"--launch.clusters={json_list(CLUSTERS)}"]
    if cfg["projected"]:
        for task in tasks:
            prefix = f"--trainer.callbacks.downstream_evaluator.tasks.{task}"
            extra += [
                f"{prefix}.eval_on_projected_registers=True",
                f"{prefix}.eval_projection_dim=128",
            ]
    return build_commands(args, extra)


def json_list(values: list[str]) -> str:
    """Render a list as the JSON literal the launch CLI expects."""
    return "[" + ",".join(f'"{v}"' for v in values) + "]"


def require_pushed_head() -> str:
    """Fail before submitting anything if HEAD is not on the remote.

    Every job -- eval or submitter -- starts by cloning the repo and running
    `git checkout $GIT_REF`, where GIT_REF is the SHA of whatever HEAD was at
    launch time. An unpushed HEAD therefore produces jobs that queue, schedule,
    spend four minutes on conda and pip, and only then die with
    `fatal: reference is not a tree` -- all of them, silently, with the failure
    visible nowhere except each job's logs. Checking costs one fetch.
    """
    sha = subprocess.run(  # nosec
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    subprocess.run(  # nosec
        ["git", "fetch", "--quiet", "origin"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    contains = subprocess.run(  # nosec
        ["git", "branch", "--remotes", "--contains", sha],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if not contains:
        raise SystemExit(
            f"HEAD ({sha[:9]}) is not on any remote branch, so every job would fail at\n"
            f"`git checkout {sha}` after wasting its setup time. Push it first:\n"
            f"    git push origin HEAD"
        )
    return sha


def plan(arms: list[str], tags: list[str]) -> list[tuple[str, str]]:
    """Return (shard name, command) for every job the sweep would submit."""
    jobs: list[tuple[str, str]] = []
    for arm in arms:
        for tag in tags:
            if tag == "pastis":
                # PASTIS has no KNN probe, and only two tasks, so both variants
                # share one shard instead of one shard each.
                tasks = [f"pastis_year_aligned_ws16_ps1_{v}" for v in VARIANTS.values()]
                shard = f"{arm}_clpxpastis"
            else:
                variant = VARIANTS[tag]
                tasks = [f"{d}_year_aligned_ws16_ps1_{variant}" for d in DATASETS] + [
                    f"{d}_year_aligned_ws16_ps1_{variant}_knn" for d in DATASETS
                ]
                shard = f"{arm}_{tag}"
            jobs += [
                (shard, c) for c in _shard_commands(arm, shard[len(arm) + 1 :], tasks)
            ]
    return jobs


def submit(jobs: list[tuple[str, str]]) -> int:
    """Create every experiment, retrying transient Beaker failures.

    Beaker's create can spuriously 409 on a fresh name, so each job gets three
    attempts; a genuine ExperimentConflict (the run already exists) is success,
    which is what makes re-running the sweep cheap.
    """
    ok = failed = 0
    for _, command in jobs:
        run_name = re.search(r"launch_evaluate (\S+)", command).group(1)  # type: ignore[union-attr]
        for attempt in (1, 2, 3):
            result = subprocess.run(  # nosec
                command, shell=True, cwd=REPO_ROOT, capture_output=True, text=True
            )
            output = (result.stdout or "") + (result.stderr or "")
            if result.returncode == 0 or "ExperimentConflict" in output:
                exists = " (exists)" if result.returncode else ""
                print(f"OK {run_name}{exists}", flush=True)
                ok += 1
                break
            tail = (
                output.strip().splitlines()[-1][:110] if output.strip() else "no output"
            )
            print(f"retry {attempt} {run_name}: {tail}", flush=True)
        else:
            print(f"FAILED {run_name}", flush=True)
            failed += 1
    print(f"\nDONE ok={ok} failed={failed}", flush=True)
    return failed


def launch_submitter(args: argparse.Namespace) -> None:
    """Run the sweep's submission loop inside a 0-GPU Beaker job.

    Nothing here needs a GPU -- the job only shells out to `beaker create` once
    per eval job -- so it asks for none and schedules against whatever capacity
    is free.
    """
    inner = [
        "python3",
        "scripts/tools/launch_cloudless_pixmask_sweep.py",
        "--go",
        f"--arms={','.join(args.arms)}",
        f"--only={','.join(args.only)}",
    ]
    config = build_launch_config(
        name="cloudless-pixmask-submit",
        cmd=inner,
        clusters=args.cluster,
        task_name="submit",
    )
    config.num_gpus = 0
    config.num_nodes = 1
    config.shared_memory = "16GiB"
    # torchrun would try to initialise a process group on a task with no GPUs.
    experiment = config.launch(follow=False, torchrun=False)
    print(f"submitted {experiment.id}: {experiment.name}")
    print(f"  https://beaker.allen.ai/ex/{experiment.id}")


def main() -> None:
    """Parse arguments and either plan, submit, or hand off to Beaker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arms",
        default=",".join(ARMS),
        help="comma-separated arm names (default: both)",
    )
    parser.add_argument(
        "--only",
        default="clpx1,clpx2,pastis",
        help="comma-separated shard tags (default: all three)",
    )
    parser.add_argument("--go", action="store_true", help="actually submit")
    parser.add_argument(
        "--as_beaker_job",
        action="store_true",
        help="submit a 0-GPU Beaker job that runs this sweep, instead of running it here",
    )
    parser.add_argument("--cluster", default=",".join(CLUSTERS))
    args = parser.parse_args()
    args.arms = [a for a in args.arms.split(",") if a]
    args.only = [t for t in args.only.split(",") if t]
    for arm in args.arms:
        if arm not in ARMS:
            parser.error(f"unknown arm {arm!r}; choose from {list(ARMS)}")
    for tag in args.only:
        if tag not in {*VARIANTS, "pastis"}:
            parser.error(f"unknown shard tag {tag!r}")

    if args.as_beaker_job:
        args.cluster = [c for c in args.cluster.split(",") if c]
        print(f"launching at {require_pushed_head()[:9]}")
        launch_submitter(args)
        return

    jobs = plan(args.arms, args.only)
    shards = sorted({s for s, _ in jobs})
    print(f"{len(jobs)} jobs across {len(shards)} shards")
    for shard in shards:
        commands = [c for s, c in jobs if s == shard]
        n_tasks = commands[0].split("tasks_to_run=")[1].count(",") + 1
        projected = "yes" if "eval_on_projected_registers=True" in commands[0] else "no"
        print(
            f"  {shard:32} {len(commands):2} jobs | {n_tasks} tasks/job | proj {projected}"
        )
    if not args.go:
        print("\n(dry run; pass --go to submit, or --as_beaker_job to submit remotely)")
        return
    print(f"submitting at {require_pushed_head()[:9]}")
    sys.exit(1 if submit(jobs) else 0)


if __name__ == "__main__":
    main()

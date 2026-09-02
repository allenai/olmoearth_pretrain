"""Re-run the per-pixel Landsat mask (`_l8pixmask`) evals after the no-op fix.

Supersedes launch_cloudless_pixmask_sweep.py, which only knew the two cloudless
variants and launched them before the flag was known to be broken.

BACKGROUND. `l8_pixel_cloud_mask` never reached the dataset until the fix: the
registry branch of evals/datasets/__init__.py dropped it into `**kwargs`, so
every `_l8pixmask` task silently re-ran its unmasked sibling. Those numbers are
void. This script produces replacements, and its design is shaped entirely by
two ways the replacements could be confusing rather than useful:

1. DO NOT let old and new numbers meet. The dashboards collapse the CSV by
   taking `max` over every row that maps to an arm -- that is how sharded runs
   are stitched together -- so a void row and a fixed row for the same task
   would silently merge, biased upward, with nothing in the output showing it.
   Renaming runs does not help; the merge is by arm. Hence a SEPARATE W&B
   PROJECT, which is also what the gram sweep does for the same reason.

2. DO NOT compare across waves. Two runs of one configuration differ by 0.88
   pts on average under the linear probe (5.71 at worst) -- measured from the
   very replicates this bug produced. The effect being chased is smaller than
   that: the scene-level filter costs 0.46-0.73. So a fixed variant scored
   against a sibling from an earlier wave would confound the mask with
   run-to-run drift. Every shard here therefore carries THE VARIANT AND ITS
   SIBLING TOGETHER, in one job, on one GPU, from one checkpoint load.

THREE TASK FAMILIES, and the second is nearly free. "AEF sampling" is not a
separate set of tasks: it is the same KNN tasks with `balanced_trial` left
enabled, which reports each predictor under its own synthetic task name
(`{host}_aeftrial_{ridge,knn5,knn20}`) and is, per evals/balanced_trial.py,
"purely additive: the caller's own train -> val probe result is untouched". So
ONE KNN job yields both our-sampling and AEF-sampling numbers, and the only
thing required is to NOT pass --no_balanced_trials. Defaults match the w3 wave,
so the new trial numbers are comparable to the existing ones. PASTIS is the
exception: it has no KNN probe, so it can only be scored under the linear probe.

Phases, cheapest first -- KNN replicate noise is 0.09 pts against LP's 0.88, so
KNN resolves effects ten times smaller for a ninth of the jobs:

    sanity  1 job    one pair, two datasets, KNN. Passes only if variant and
                     sibling DIFFER. This is the check that would have caught
                     the original no-op before a hundred jobs were spent.
    knn     12 jobs  every pair x both arms over the 8 AEF datasets, KNN with
                     balanced trials on -- covers `aef` AND `aef w aef sampling`.
    knn_strict
            12 jobs  the same, for the NARROW policy (`_l8pixstrict`, cloud bit
                     only) against the same unmasked siblings, so the two
                     policies' deltas can be read against each other. NOTE that
                     `_l8pixstrict` masks LESS than `_l8pixmask`: "strict" is
                     the criterion for calling a pixel cloudy, so the ladder is
                     unmasked < _l8pixstrict < _l8pixmask.
    pastis  16 jobs  PASTIS, LP-only (8 learning rates), all six pairs per arm.
    pastis_strict
            16 jobs  the same for the narrow policy. Mostly a negative control:
                     the aggressive policy was already null on PASTIS, which has
                     no cloudy-season confound.
    lp / lp_strict
            48 jobs per arm, per policy. The AEF datasets under the linear probe.
            SIBLINGS ARE NOT OPTIONAL HERE, unlike the KNN phases: LP replicate
            noise is 0.88 pts (5.71 worst) against effects of ~0.5-2, so a
            cross-wave sibling would swamp what is being measured. Each job
            therefore carries variant + sibling on all 8 datasets.

    python3 scripts/tools/launch_pixmask_sweep.py --phase sanity --as_beaker_job
    python3 scripts/tools/launch_pixmask_sweep.py --phase knn    --as_beaker_job
    python3 scripts/tools/launch_pixmask_sweep.py --phase pastis --as_beaker_job
"""

import argparse
import re
import subprocess  # nosec
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from olmoearth_pretrain.internal.all_evals import (  # noqa: E402
    EMBEDDING_EVAL_TASKS,
)
from olmoearth_pretrain.internal.common import build_launch_config  # noqa: E402
from olmoearth_pretrain.internal.embedding_eval_sweep import (  # noqa: E402
    build_commands,
)

# Deliberately NOT 20260807_perceiver_updated_evals: see (1) above.
PROJECT = "20260813_l8pixmask_fixed"
CLUSTERS = ["ai2/jupiter", "ai2/ceres", "ai2/saturn"]

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

# (tag, masked variant, the config it must be compared against). The sibling is
# the same stack with the Landsat side left alone, so the only difference inside
# a shard is the per-pixel mask.
PAIRS = [
    ("pxfix1", "sentinel2_landsat_l8pixmask", "sentinel2_landsat"),
    ("pxfix2", "sentinel2_landsat_sclmask_l8pixmask", "sentinel2_landsat_sclmask"),
    ("pxfix3", "sentinel2_landsat_cloudless_l8pixmask", "sentinel2_landsat_cloudless"),
    ("pxfix4", "sentinel1_sentinel2_landsat_l8pixmask", "sentinel1_sentinel2_landsat"),
    (
        "pxfix5",
        "sentinel1_sentinel2_landsat_sclmask_l8pixmask",
        "sentinel1_sentinel2_landsat_sclmask",
    ),
    (
        "pxfix6",
        "sentinel1_sentinel2_landsat_cloudless_l8pixmask",
        "sentinel1_sentinel2_landsat_cloudless",
    ),
]

# descals is the cloudiest dataset in the suite (27.1% cloudy centre pixels) and
# ethiopia the one where masking has historically hurt -- between them, a mask
# that is actually applied cannot fail to move at least one number.
# The narrow policy (`_l8pixstrict`, cloud bit only) against the same unmasked
# siblings, so its delta is directly comparable with the aggressive variant's.
STRICT_PAIRS = [
    (tag.replace("pxfix", "pxstrict"), var.replace("_l8pixmask", "_l8pixstrict"), sib)
    for tag, var, sib in PAIRS
]

SANITY_PAIR = "pxfix5"
SANITY_DATASETS = ["descals", "ethiopia_crops"]

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
        "projected": True,
    },
}


def json_list(values: list[str]) -> str:
    """Render a list as the JSON literal the launch CLI expects."""
    return "[" + ",".join(f'"{v}"' for v in values) + "]"


def require_pushed_head() -> str:
    """Fail before submitting if HEAD is not on the remote.

    Every job clones the repo and runs `git checkout $GIT_REF`. An unpushed HEAD
    makes all of them queue, install for four minutes, and die with `reference is
    not a tree` -- which is how 26 jobs were lost on 2026-08-13.
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
    on_remote = subprocess.run(  # nosec
        ["git", "branch", "--remotes", "--contains", sha],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if not on_remote:
        raise SystemExit(
            f"HEAD ({sha[:9]}) is on no remote branch; every job would fail at\n"
            f"`git checkout {sha}`. Push it first:  git push origin HEAD"
        )
    return sha


def tasks_for(
    variant: str,
    sibling: str,
    datasets: list[str],
    knn: bool,
    siblings: bool = True,
) -> list[str]:
    """Task names for one matched pair, variant and sibling interleaved.

    Both configs land in one job so they share a checkpoint load and a GPU; the
    interleaving is only cosmetic, but it makes the shard's log readable as
    pairs.

    ``siblings=False`` drops the sibling half. Worth it once the sibling is
    ALREADY measured: it is half the compute, and because the evaluator runs
    tasks in registration order -- every sibling before every variant -- it is
    also the half that runs FIRST, so carrying it doubles the time to the first
    number you actually wanted. The cost is that the delta is then read against
    a sibling from another wave, which the replicate study prices at ~0.06 pts
    under KNN -- an order of magnitude below the effects here.
    """
    suffix = "_knn" if knn else ""
    specs = (variant, sibling) if siblings else (variant,)
    names = []
    for dataset in datasets:
        for spec in specs:
            names.append(f"{dataset}_year_aligned_ws16_ps1_{spec}{suffix}")
    unknown = [n for n in names if n not in EMBEDDING_EVAL_TASKS]
    if unknown:
        raise SystemExit(f"unregistered task(s): {', '.join(unknown)}")
    return names


def shard_commands(arm: str, tag: str, tasks: list[str], trials: bool) -> list[str]:
    """Build the Beaker commands for one shard.

    ``trials`` leaves the balanced-trial protocol enabled on the KNN tasks, which
    is what produces the AEF-sampling numbers alongside our own. The other
    balanced_trial_* fields stay None so the draw matches the w3 wave. On LP-only
    shards the flag is inert -- LP tasks carry no balanced_trial config.
    """
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
        no_balanced_trials=not trials,
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


def plan(phase: str, arms: list[str], siblings: bool = True) -> list[tuple[str, str]]:
    """Return (shard name, command) for every job in the given phase."""
    jobs: list[tuple[str, str]] = []
    if phase == "sanity":
        tag, variant, sibling = next(p for p in PAIRS if p[0] == SANITY_PAIR)
        tasks = tasks_for(variant, sibling, SANITY_DATASETS, knn=True)
        shard = f"cand_ndvi_{tag}sanity"
        return [
            (shard, c)
            for c in shard_commands("cand_ndvi", f"{tag}sanity", tasks, trials=False)
        ]
    if phase.startswith("pastis"):
        # PASTIS is LP-only and only two tasks per pair, so all six pairs ride
        # one shard per arm rather than six shards of eight jobs each.
        pastis_pairs = STRICT_PAIRS if phase.endswith("_strict") else PAIRS
        for arm in arms:
            tasks = []
            for _, variant, sibling in pastis_pairs:
                tasks += tasks_for(
                    variant, sibling, ["pastis"], knn=False, siblings=siblings
                )
            shard = f"{arm}_px{'strict' if phase.endswith('_strict') else 'fix'}pastis"
            jobs += [
                (shard, c)
                for c in shard_commands(arm, shard[len(arm) + 1 :], tasks, trials=False)
            ]
        return jobs
    pairs = STRICT_PAIRS if phase.endswith("_strict") else PAIRS
    for arm in arms:
        for tag, variant, sibling in pairs:
            tasks = tasks_for(
                variant,
                sibling,
                DATASETS,
                knn=phase.startswith("knn"),
                siblings=siblings,
            )
            shard = f"{arm}_{tag}"
            jobs += [
                (shard, c)
                for c in shard_commands(arm, tag, tasks, trials=phase.startswith("knn"))
            ]
    return jobs


def submit(jobs: list[tuple[str, str]]) -> int:
    """Create every experiment, retrying Beaker's spurious create failures."""
    ok = failed = 0
    for _, command in jobs:
        run_name = re.search(r"launch_evaluate (\S+)", command).group(1)  # type: ignore[union-attr]
        for attempt in (1, 2, 3):
            result = subprocess.run(  # nosec
                command, shell=True, cwd=REPO_ROOT, capture_output=True, text=True
            )
            output = (result.stdout or "") + (result.stderr or "")
            if result.returncode == 0 or "ExperimentConflict" in output:
                print(f"OK {run_name}", flush=True)
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
    """Run the submission loop in a 0-GPU Beaker job rather than here."""
    inner = [
        "python3",
        "scripts/tools/launch_pixmask_sweep.py",
        "--go",
        f"--phase={args.phase}",
        f"--arms={','.join(args.arms)}",
    ]
    if args.no_siblings:
        inner.append("--no_siblings")
    config = build_launch_config(
        name=f"pixmask-{args.phase}{'-solo' if args.no_siblings else ''}-submit",
        cmd=inner,
        clusters=args.cluster,
        task_name="submit",
    )
    config.num_gpus = 0
    config.num_nodes = 1
    config.shared_memory = "16GiB"
    experiment = config.launch(follow=False, torchrun=False)
    print(f"submitted {experiment.id}: {experiment.name}")
    print(f"  https://beaker.allen.ai/ex/{experiment.id}")


def main() -> None:
    """Parse arguments and plan, submit, or hand off to Beaker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=(
            "sanity",
            "knn",
            "knn_strict",
            "pastis",
            "pastis_strict",
            "lp",
            "lp_strict",
        ),
        default="knn",
    )
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--go", action="store_true")
    parser.add_argument(
        "--no_siblings",
        action="store_true",
        help="omit the sibling half (use when it is already measured; halves the job)",
    )
    parser.add_argument("--as_beaker_job", action="store_true")
    parser.add_argument("--cluster", default=",".join(CLUSTERS))
    args = parser.parse_args()
    args.arms = [a for a in args.arms.split(",") if a]
    for arm in args.arms:
        if arm not in ARMS:
            parser.error(f"unknown arm {arm!r}; choose from {list(ARMS)}")

    if args.as_beaker_job:
        args.cluster = [c for c in args.cluster.split(",") if c]
        print(f"launching at {require_pushed_head()[:9]}")
        launch_submitter(args)
        return

    jobs = plan(args.phase, args.arms, siblings=not args.no_siblings)
    shards = sorted({s for s, _ in jobs})
    print(f"phase={args.phase}  project={PROJECT}")
    print(f"{len(jobs)} jobs across {len(shards)} shards")
    for shard in shards:
        commands = [c for s, c in jobs if s == shard]
        n_tasks = commands[0].split("tasks_to_run=")[1].count(",") + 1
        projected = "yes" if "eval_on_projected_registers=True" in commands[0] else "no"
        print(
            f"  {shard:34} {len(commands):2} jobs | {n_tasks:2} tasks/job | proj {projected}"
        )
    if not args.go:
        print("\n(dry run; pass --go to submit, or --as_beaker_job to submit remotely)")
        return
    print(f"submitting at {require_pushed_head()[:9]}")
    sys.exit(1 if submit(jobs) else 0)


if __name__ == "__main__":
    main()

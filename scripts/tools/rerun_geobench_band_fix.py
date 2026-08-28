"""Re-score m-eurosat and m-brick-kiln after the GeoBench band-labelling fix.

GeoBench mislabelled the bands of both 13-band S2 tasks (see
``EvalDatasetConfig.band_name_corrections``), so every number we have published
for them was computed on the wrong inputs. Correcting the loader invalidates
those two columns for every model, which means re-running the full sweep --
32 KNN/LP configs and 3 finetuning LRs -- once per model.

Launched the usual way, that is ~35 configs x N models of individually tiny
Beaker jobs, and the two datasets are small enough (m-eurosat 4,000 images,
m-brick-kiln 17,061, all 64x64) that per-job container setup would dominate the
actual compute by an order of magnitude. So this driver inverts the shape: it
runs the existing sweeps with ``--cluster=local``, which executes every
hyperparameter in-process instead of submitting an experiment per config, and
loops models inside one long-lived GPU job. A handful of these run as shards.

What this deliberately does NOT do is skip work to save time. Every model gets
the whole grid, because the previously-selected best hyperparameters were chosen
on validation splits computed from the broken bands and the optimum is free to
move.

Resumability matters more here than in a fan-out launch: one preemption would
otherwise cost a whole shard. Each (model, sweep) writes a marker on completion
and is skipped on a restart, so a preempted shard resumes at model granularity.
A model that fails is recorded and does not stop its shard -- ``full_eval_sweep``
runs its configs under ``check=True``, so one bad config aborts that model's
remaining configs, and the retry pass is what picks it back up.

Usage (inside a GPU job; see rerun_geobench_band_fix_beaker.yaml):

    python scripts/tools/rerun_geobench_band_fix.py \
        --models_json models.json --shard 0 --num_shards 4 \
        --state_dir /weka/dfive-default/.../rerun_state

``models.json`` is a list of entries, each either an OlmoEarth checkpoint::

    {"name": "cand_ndvi", "checkpoint_path": "/weka/...", "module_path": "scripts/..."}

or a baseline the sweeps already know how to build::

    {"name": "croma", "model": "croma"}

Add ``"extra_args": ["--load_arch_from_checkpoint"]`` for runs whose
architecture must be rebuilt from the checkpoint's config.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess  # nosec
import sys
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("rerun_geobench_band_fix")

# The two tasks the band fix invalidated. Everything else is unaffected, and
# scoping to these is what makes a sequential re-run affordable at all.
TASKS = ["m_eurosat", "m_brick_kiln"]

# Module paths, not entry points: run as `python -m <module>`. Defaults to the
# cropharvest wrappers because those are what produced the numbers being
# replaced -- swapping in olmoearth_pretrain.* would change the task catalogue
# and make the new column non-comparable with the rest of the table.
SWEEPS = {
    "knn_lp": "olmoearth_plus_cropharvest.full_eval_sweep",
    "finetune": "olmoearth_plus_cropharvest.full_eval_sweep_finetune",
}

# These runs must be quarantined in their own W&B projects. Both sweeps fall
# back to the shared EVAL_WANDB_PROJECT when --project_name is absent, so a
# dropped flag would silently interleave band-fix numbers with the published
# ones in the project everyone reads -- and the two are not comparable, since
# they were computed on different inputs. Refusing the shared project by name
# makes that a startup error instead of a contaminated dashboard.
DEFAULT_KNN_LP_PROJECT = "20260828_geobench_bandfix_knn_lp"
DEFAULT_FINETUNE_PROJECT = "20260828_geobench_bandfix_finetune"
_BANDFIX_MARKER = "bandfix"


@dataclass
class ModelSpec:
    """One model to re-score: either a checkpoint of ours or a named baseline."""

    name: str
    checkpoint_path: str | None = None
    module_path: str | None = None
    model: str | None = None
    extra_args: list[str] | None = None

    def __post_init__(self) -> None:
        """Reject specs that are neither a checkpoint nor a baseline."""
        if not self.model and not self.checkpoint_path:
            raise ValueError(
                f"{self.name}: give either 'model' (a baseline) or "
                f"'checkpoint_path' (one of ours)"
            )
        if self.model and self.checkpoint_path:
            raise ValueError(
                f"{self.name}: 'model' and 'checkpoint_path' are mutually "
                f"exclusive -- a baseline has no checkpoint of ours to load"
            )

    def sweep_args(self) -> list[str]:
        """The model-identifying half of a sweep invocation."""
        args = []
        if self.model:
            args += [f"--model={self.model}"]
        else:
            args += [f"--checkpoint_path={self.checkpoint_path}"]
            if self.module_path:
                args += [f"--module_path={self.module_path}"]
        return args + list(self.extra_args or [])


def load_models(path: Path) -> list[ModelSpec]:
    """Read and validate the model list."""
    entries = json.loads(path.read_text())
    if not isinstance(entries, list):
        raise ValueError(f"{path}: expected a JSON list of model entries")
    models = [ModelSpec(**entry) for entry in entries]
    names = [m.name for m in models]
    duplicated = sorted({n for n in names if names.count(n) > 1})
    if duplicated:
        # Names are the marker filenames, so a duplicate would make one model
        # silently inherit another's "already done" state.
        raise ValueError(f"{path}: duplicate model names: {duplicated}")
    return models


def shard(models: list[ModelSpec], index: int, count: int) -> list[ModelSpec]:
    """Take every ``count``-th model.

    Strided rather than contiguous so that a slow family of models (the big
    checkpoints, say) is spread across shards instead of landing entirely in
    one, which is what decides the wall clock when shards run in parallel.
    """
    if not 0 <= index < count:
        raise ValueError(f"shard {index} out of range for {count} shards")
    return models[index::count]


def check_project_is_quarantined(project_name: str, sweep: str) -> None:
    """Refuse to write band-fix results into a shared or unmarked W&B project.

    The point of the re-run is that these numbers replace published ones, so
    they must be separable from them until someone deliberately merges. Both a
    shared project and a plausible-but-unmarked name are rejected: the marker is
    what lets a reader tell at a glance which inputs a project was computed on.
    """
    from olmoearth_pretrain.internal.constants import EVAL_WANDB_PROJECT

    if project_name == EVAL_WANDB_PROJECT:
        raise ValueError(
            f"{sweep}: refusing to write to the shared eval project "
            f"'{EVAL_WANDB_PROJECT}' -- band-fix numbers are not comparable "
            f"with what is already there."
        )
    if _BANDFIX_MARKER not in project_name:
        raise ValueError(
            f"{sweep}: project '{project_name}' does not contain "
            f"'{_BANDFIX_MARKER}'. Name it so the band fix is visible from the "
            f"project list, e.g. '{DEFAULT_KNN_LP_PROJECT}'."
        )


def build_command(
    spec: ModelSpec, sweep: str, project_name: str, priority_env: str | None
) -> list[str]:
    """Build one sweep invocation for one model."""
    module = SWEEPS[sweep]
    # --cluster=local is the whole point: it selects SubCmd.evaluate, so the
    # sweep runs each hyperparameter itself rather than submitting a Beaker
    # experiment per config.
    cmd = [
        sys.executable,
        "-m",
        module,
        "--cluster=local",
        f"--project_name={project_name}",
        # Hyphens, not underscores. Both sweeps declare this one as
        # "--task-names" while their other flags use underscores, and both parse
        # with parse_known_args -- so "--task_names" is not a parse error, it is
        # silently forwarded to the eval command as an unknown override and the
        # sweep quietly runs all 91 (or 21) tasks instead of these two.
        f"--task-names={','.join(TASKS)}",
    ]
    cmd += spec.sweep_args()
    if priority_env:
        cmd += [priority_env]
    return cmd


def run_one(
    spec: ModelSpec,
    sweep: str,
    project_name: str,
    state_dir: Path,
    dry_run: bool,
) -> str:
    """Run one (model, sweep), or skip it if a previous attempt finished it."""
    marker = state_dir / f"{spec.name}.{sweep}.done"
    if marker.exists():
        logger.info("SKIP %s/%s (already done)", spec.name, sweep)
        return "skipped"

    cmd = build_command(spec, sweep, project_name, None)
    logger.info("RUN  %s/%s: %s", spec.name, sweep, " ".join(cmd))
    if dry_run:
        return "dry_run"

    started = time.monotonic()
    result = subprocess.run(cmd, check=False)  # nosec
    elapsed = time.monotonic() - started
    if result.returncode == 0:
        marker.write_text(f"ok after {elapsed:.0f}s\n")
        logger.info("DONE %s/%s in %.0fs", spec.name, sweep, elapsed)
        return "ok"

    # Deliberately not fatal. A shard is hours of work across many models, and
    # one model's failure should not cost the rest of them; the failure file is
    # the queue for the retry pass.
    (state_dir / f"{spec.name}.{sweep}.failed").write_text(
        f"exit {result.returncode} after {elapsed:.0f}s\n"
    )
    logger.error(
        "FAIL %s/%s: exit %d after %.0fs", spec.name, sweep, result.returncode, elapsed
    )
    return "failed"


def main() -> None:
    """Run this shard's models through both sweeps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models_json", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--state_dir", type=Path, required=True)
    parser.add_argument(
        "--sweeps",
        type=str,
        default="knn_lp,finetune",
        help=f"Comma-separated subset of {sorted(SWEEPS)}, in run order.",
    )
    parser.add_argument("--knn_lp_project", type=str, default=DEFAULT_KNN_LP_PROJECT)
    parser.add_argument(
        "--finetune_project", type=str, default=DEFAULT_FINETUNE_PROJECT
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    sweeps = [s.strip() for s in args.sweeps.split(",") if s.strip()]
    unknown = sorted(set(sweeps) - set(SWEEPS))
    if unknown:
        raise ValueError(f"Unknown sweeps: {unknown}. Known: {sorted(SWEEPS)}")
    projects = {
        "knn_lp": args.knn_lp_project,
        "finetune": args.finetune_project,
    }
    for sweep in sweeps:
        check_project_is_quarantined(projects[sweep], sweep)

    models = shard(load_models(args.models_json), args.shard, args.num_shards)
    # Check every checkpoint up front. A wrong weka root or an unfilled
    # placeholder would otherwise surface deep inside the first sweep, after
    # the container build and however many models came before it.
    missing = [
        f"{m.name}: {m.checkpoint_path}"
        for m in models
        if m.checkpoint_path and not Path(m.checkpoint_path, "config.json").is_file()
    ]
    if missing and not args.dry_run:
        raise SystemExit(
            "checkpoints not found (expected <path>/config.json):\n  "
            + "\n  ".join(missing)
        )

    args.state_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "shard %d/%d: %d models x %d sweeps -> %s",
        args.shard,
        args.num_shards,
        len(models),
        len(sweeps),
        [m.name for m in models],
    )

    tally: dict[str, int] = {}
    # Sweeps in the outer loop so the cheap KNN/LP results for every model in
    # the shard land before any of the slow finetuning starts. If the shard is
    # cut short, that is the half worth having.
    for sweep in sweeps:
        for spec in models:
            outcome = run_one(
                spec, sweep, projects[sweep], args.state_dir, args.dry_run
            )
            tally[outcome] = tally.get(outcome, 0) + 1

    logger.info("shard %d finished: %s", args.shard, tally)
    if tally.get("failed"):
        # Non-zero so the Beaker task is visibly unhealthy, but only after
        # everything runnable has run.
        sys.exit(1)


if __name__ == "__main__":
    main()

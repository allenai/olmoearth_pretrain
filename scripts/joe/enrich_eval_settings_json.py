"""Add per-task ``norm_mode`` to a max-eval-settings JSON.

The JSON written by ``get_max_eval_metrics_from_wandb.py`` only records
``norm_stats_from_pretrained``. For a few foundation models (notably Galileo,
Tessera, Clay, ...) the per-task helper hardcodes that flag regardless of which
sweep arm produced the val-best result, so the JSON cannot disambiguate the
two modes. The actual mode is encoded in the W&B *run name*
(``<base>_pre_trained_lr...`` vs ``<base>_dataset_lr...``).

This tool walks the JSON, looks up each run on W&B (searching the entity's
projects until it finds a match), parses ``pre_trained`` / ``dataset`` out of
the run name, and writes a new JSON next to the input with a ``.enriched.json``
suffix and an extra ``settings.norm_mode`` field per task.
"""

import argparse
import json
from logging import getLogger

import wandb

logger = getLogger(__name__)

ENTITY = "eai-ai2"


def _infer_mode_from_config(run: wandb.Run, task_name: str) -> str | None:
    """Determine pre_trained vs dataset by reading the run's config.

    Authoritative signal (in priority order):
    1. ``model.use_pretrained_normalizer`` is set -> use that.
    2. Per-task ``norm_stats_from_pretrained`` -> True means pretrained mode.
    3. Fall back to parsing the run name for ``_pre_trained_`` / ``_dataset_``.
    """
    cfg = run.config or {}
    model_cfg = cfg.get("model", {}) or {}
    if "use_pretrained_normalizer" in model_cfg:
        return "pre_trained" if model_cfg["use_pretrained_normalizer"] else "dataset"
    tasks = (
        cfg.get("trainer", {})
        .get("callbacks", {})
        .get("downstream_evaluator", {})
        .get("tasks", {})
        or {}
    )
    task_cfg = tasks.get(task_name) or {}
    if "norm_stats_from_pretrained" in task_cfg:
        return "pre_trained" if task_cfg["norm_stats_from_pretrained"] else "dataset"
    name = run.name or ""
    if "_pre_trained_" in name:
        return "pre_trained"
    if "_dataset_" in name:
        return "dataset"
    return None


def _find_run(
    api: wandb.Api, run_id: str, project_cache: list[str]
) -> wandb.Run | None:
    """Search known projects for a run id, falling back to scanning all projects."""
    for proj in project_cache:
        try:
            return api.run(f"{ENTITY}/{proj}/{run_id}")
        except Exception:  # noqa: BLE001,S112 - wandb raises various lookup errors
            continue  # nosec B112
    # Fallback: enumerate every project under the entity.
    for p in api.projects(entity=ENTITY):
        if p.name in project_cache:
            continue
        try:
            run = api.run(f"{ENTITY}/{p.name}/{run_id}")
            project_cache.append(p.name)  # cache for next lookup
            return run
        except Exception:  # noqa: BLE001,S112
            continue  # nosec B112
    return None


def main() -> None:
    """Enrich the JSON in-place with norm_mode."""
    p = argparse.ArgumentParser()
    p.add_argument("input_json")
    p.add_argument(
        "--output",
        default=None,
        help="Output path. Default: input with .enriched.json suffix.",
    )
    p.add_argument(
        "--projects",
        default=None,
        help="Comma-separated project names to try first (faster than scanning all).",
    )
    args = p.parse_args()

    with open(args.input_json) as f:
        data = json.load(f)

    api = wandb.Api()
    project_cache: list[str] = (
        [s.strip() for s in args.projects.split(",")] if args.projects else []
    )

    n_total = 0
    n_resolved = 0
    n_unresolved: list[tuple[str, str]] = []
    for group, tasks in data.items():
        for task_name, entry in tasks.items():
            n_total += 1
            run_id = entry.get("run_id")
            if run_id is None:
                continue
            run = _find_run(api, run_id, project_cache)
            if run is None:
                n_unresolved.append((group, task_name))
                logger.warning(f"Could not find run {run_id} ({group}/{task_name})")
                continue
            mode = _infer_mode_from_config(run, task_name)
            if mode is None:
                logger.warning(
                    f"Could not parse mode from run name {run.name!r} "
                    f"({group}/{task_name})"
                )
                n_unresolved.append((group, task_name))
                continue
            entry["settings"]["norm_mode"] = mode
            n_resolved += 1
            print(f"  {group}/{task_name}: {mode} (from {run.name})")

    out_path = args.output or args.input_json.replace(".json", ".enriched.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print()
    print(f"Resolved {n_resolved}/{n_total} task entries.")
    if n_unresolved:
        print(f"Unresolved ({len(n_unresolved)}):")
        for g, t in n_unresolved:
            print(f"  {g}/{t}")
    print(f"Wrote {out_path}")
    print(f"Project cache (pass --projects to skip search): {','.join(project_cache)}")


if __name__ == "__main__":
    main()

"""Find the best LR (and corresponding val + test) per task in a sweep of wandb runs.

Given a prefix that matches multiple wandb runs (each differing in probe LR), for each
downstream eval task report:
  - the LR from the run with the highest val score
  - that val score
  - the corresponding test score

Usage:
    python -m scripts.tools.get_best_lr_per_task_from_wandb \
        -p my_project --run-prefix my_experiment_step300000
"""

import argparse
import csv
import re

import wandb

WANDB_ENTITY = "eai-ai2"

# Match the eval LR encoded in run names like
# "..._pre_trained_lr0.0005_ptmean" or "..._dataset_lr0.01_ptmax".
_LR_FROM_NAME_RE = re.compile(r"_lr([0-9.eE+-]+)_pt")


def _lr_from_run_name(name: str) -> float | None:
    matches = _LR_FROM_NAME_RE.findall(name)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _normalize_eval_key(key: str) -> str:
    """eval_other/... -> eval/... so we can compare across both namespaces."""
    if key.startswith("eval_other/"):
        return key.replace("eval_other/", "eval/", 1)
    return key


def _get_test_value(
    run: wandb.apis.public.Run, task: str, val_key: str
) -> tuple[float | None, str | None]:
    """Look up the test value matching a normalized validation key."""
    test_key = val_key.replace("eval/", "eval/test/", 1)
    candidates = [
        test_key,
        test_key.replace("eval/", "eval_other/", 1),
        f"eval/test/{task}",
    ]
    for k in candidates:
        v = run.summary.get(k)
        if v is not None:
            return float(v), k
    return None, None


def _task_config(run: wandb.apis.public.Run, task: str) -> dict:
    return (
        run.config.get("trainer", {})
        .get("callbacks", {})
        .get("downstream_evaluator", {})
        .get("tasks", {})
        .get(task, {})
    )


def get_best_per_task(
    project: str, run_prefix: str, metric_key: str | None = None
) -> dict[str, dict]:
    """Return per-task best-val info across runs matching `run_prefix`.

    Args:
        project: W&B project name under the entity.
        run_prefix: prefix shared by the runs in the sweep.
        metric_key: if provided, only this exact metric key is used for
            selection (e.g. ``eval/lfmc_woody_3k``).  When None, all
            eval metrics compete per task (beware that sub-metrics like
            positive RMSE can shadow the primary).
    """
    api = wandb.Api()
    wandb_path = f"{WANDB_ENTITY}/{project}"
    runs = [
        r for r in api.runs(wandb_path, lazy=False) if r.name.startswith(run_prefix)
    ]
    print(f"Found {len(runs)} runs with prefix '{run_prefix}'")
    for r in runs:
        print(f"  {r.name} ({r.id})")

    # task -> {"val": float, "run": Run, "val_key": str}
    best: dict[str, dict] = {}
    for run in runs:
        for key, value in run.summary.items():
            if not (key.startswith("eval/") or key.startswith("eval_other/")):
                continue
            if key.startswith("eval/test/") or key.startswith("eval_other/test/"):
                continue
            if not isinstance(value, int | float):
                continue
            normalized = _normalize_eval_key(key)
            if metric_key is not None and normalized != metric_key:
                continue
            parts = normalized.split("/")
            if len(parts) < 2:
                continue
            task = parts[1]
            if task not in best or value > best[task]["val"]:
                best[task] = {"val": float(value), "run": run, "val_key": normalized}

    results: dict[str, dict] = {}
    for task, info in best.items():
        run = info["run"]
        cfg = _task_config(run, task)
        test_val, test_key = _get_test_value(run, task, info["val_key"])
        lr = cfg.get("ft_lr") or cfg.get("probe_lr")
        if lr is None:
            lr = _lr_from_run_name(run.name)
        results[task] = {
            "lr": lr,
            "val": info["val"],
            "val_key": info["val_key"],
            "test": test_val,
            "test_key": test_key,
            "pooling_type": cfg.get("pooling_type"),
            "norm_stats_from_pretrained": cfg.get("norm_stats_from_pretrained"),
            "run_id": run.id,
            "run_name": run.name,
        }
    return results


def _print_table(results: dict[str, dict]) -> None:
    header = f"{'task':<40} {'lr':>12} {'val':>10} {'test':>10}  run"
    print("\n" + header)
    print("-" * len(header))
    for task in sorted(results):
        r = results[task]
        lr = f"{r['lr']:.2e}" if isinstance(r["lr"], int | float) else "—"
        val = f"{r['val']:.4f}"
        test = f"{r['test']:.4f}" if r["test"] is not None else "—"
        print(f"{task:<40} {lr:>12} {val:>10} {test:>10}  {r['run_name']}")


def _save_csv(results: dict[str, dict], path: str) -> None:
    fields = [
        "task",
        "lr",
        "val",
        "test",
        "val_key",
        "test_key",
        "pooling_type",
        "norm_stats_from_pretrained",
        "run_id",
        "run_name",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for task in sorted(results):
            row = {"task": task, **results[task]}
            w.writerow({k: row.get(k) for k in fields})
    print(f"\nWrote {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-p", "--project", required=True, help="W&B project under eai-ai2"
    )
    parser.add_argument(
        "--run-prefix",
        required=True,
        help="Prefix shared by the runs in the sweep (typically the run name with the LR suffix stripped).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional CSV output path (default: {run_prefix}_best_lr.csv).",
    )
    parser.add_argument(
        "--metric-key",
        default=None,
        help="Only use this exact metric key for best-run selection "
        "(e.g. 'eval/lfmc_woody_3k'). Prevents sub-metrics like positive "
        "RMSE from shadowing the primary negative-RMSE metric.",
    )
    args = parser.parse_args()

    results = get_best_per_task(args.project, args.run_prefix, args.metric_key)
    if not results:
        print("No matching tasks found.")
        raise SystemExit(1)

    _print_table(results)
    out = args.output or f"{args.run_prefix}_best_lr.csv"
    _save_csv(results, out)

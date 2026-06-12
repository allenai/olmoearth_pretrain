"""Check per-class Presto OSM eval metrics in a W&B project.

Run after `wandb login`.

Example:
    python -m scripts.tools.analyze_presto_osm_wandb_classes \
      --project 2026_06_10_presto_osm_balanced_evals
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

import pandas as pd
import wandb

WANDB_ENTITY = "eai-ai2"

PRESTO_OSM_TASKS = [
    "presto_osm_populous12_seg_probe_sentinel2_l2a",
    "presto_osm_diverse_context_probe_sentinel2_l2a",
    "presto_osm_rare4_seg_probe_sentinel2_l2a",
]

OSM_CLASS_NAMES = {
    0: "aerialway_pylon",
    1: "aerodrome",
    2: "airstrip",
    3: "amenity_fuel",
    4: "building",
    5: "chimney",
    6: "communications_tower",
    7: "crane",
    8: "flagpole",
    9: "fountain",
    10: "generator_wind",
    11: "helipad",
    12: "highway",
    13: "leisure",
    14: "lighthouse",
    15: "obelisk",
    16: "observatory",
    17: "parking",
    18: "petroleum_well",
    19: "power_plant",
    20: "power_substation",
    21: "power_tower",
    22: "river",
    23: "runway",
    24: "satellite_dish",
    25: "silo",
    26: "storage_tank",
    27: "taxiway",
    28: "water_tower",
    29: "works",
}
POPULOUS_12_RAW_CLASS_IDS = [1, 3, 4, 12, 13, 17, 19, 20, 21, 22, 23, 29]
POPULOUS_12_CLASS_NAMES = {
    new_id: OSM_CLASS_NAMES[raw_id]
    for new_id, raw_id in enumerate(POPULOUS_12_RAW_CLASS_IDS)
}
RARE_FOCUS_CLASS_IDS = [9, 10, 26, 27]
RARE_FOCUS_CLASS_NAMES = {
    new_id: OSM_CLASS_NAMES[raw_id]
    for new_id, raw_id in enumerate(RARE_FOCUS_CLASS_IDS)
}


@dataclass(frozen=True)
class TaskSpec:
    """Expected per-class layout for a Presto OSM eval task."""

    name: str
    class_names: dict[int, str]
    split_variant: str
    label_mode: str


TASK_SPECS = {
    "presto_osm_populous12_seg_probe_sentinel2_l2a": TaskSpec(
        name="presto_osm_populous12_seg_probe_sentinel2_l2a",
        class_names=POPULOUS_12_CLASS_NAMES,
        split_variant="osm_base_balanced",
        label_mode="all_classes",
    ),
    "presto_osm_diverse_context_probe_sentinel2_l2a": TaskSpec(
        name="presto_osm_diverse_context_probe_sentinel2_l2a",
        class_names=OSM_CLASS_NAMES,
        split_variant="osm_diverse_context",
        label_mode="tile_presence",
    ),
    "presto_osm_rare4_seg_probe_sentinel2_l2a": TaskSpec(
        name="presto_osm_rare4_seg_probe_sentinel2_l2a",
        class_names=RARE_FOCUS_CLASS_NAMES,
        split_variant="osm_rare_class_focused",
        label_mode="all_classes",
    ),
}


def numeric(value: Any) -> float | None:
    """Return a float for numeric W&B summary values."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def summary_metric(summary: dict[str, Any], *keys: str) -> float | None:
    """Return the first numeric summary metric found for candidate keys."""
    for key in keys:
        value = numeric(summary.get(key))
        if value is not None:
            return value
    return None


def primary_key(task: str, split: str) -> str:
    """Return the primary metric key for a task/split."""
    if split == "val":
        return f"eval/{task}"
    if split == "test":
        return f"eval/test/{task}"
    raise ValueError(f"Unsupported split: {split}")


def other_key(task: str, metric: str, split: str) -> str:
    """Return the non-primary metric key for a task/split."""
    if split == "val":
        return f"eval_other/{task}/{metric}"
    if split == "test":
        return f"eval_other/test/{task}/{metric}"
    raise ValueError(f"Unsupported split: {split}")


def class_f1(
    summary: dict[str, Any], task: str, class_id: int, split: str
) -> float | None:
    """Read a per-class F1 value from W&B summary."""
    return summary_metric(summary, other_key(task, f"f1_class_{class_id}", split))


def split_root() -> Any:
    """Return the canonical packaged Presto OSM split directory."""
    return files("olmoearth_pretrain.evals.datasets").joinpath(
        "splits/presto_osm_balanced"
    )


def supported_classes_for_split(spec: TaskSpec, split: str) -> list[int]:
    """Return class ids that should be evaluated for one task split."""
    if spec.label_mode == "all_classes":
        return sorted(spec.class_names)

    csv_path = split_root().joinpath(spec.split_variant, f"{split}.csv")
    rows = pd.read_csv(str(csv_path))
    if spec.label_mode == "anchor_class":
        if "anchor_class_id" not in rows.columns:
            raise ValueError(f"{csv_path} is missing anchor_class_id")
        return sorted(rows["anchor_class_id"].astype(int).unique().tolist())
    if spec.label_mode == "tile_presence":
        if "labels" not in rows.columns:
            raise ValueError(f"{csv_path} is missing labels")
        class_ids: set[int] = set()
        for labels in rows["labels"].astype(str):
            class_ids.update(int(label_id) for label_id in labels.split())
        return sorted(class_ids)
    raise ValueError(f"Unsupported label mode: {spec.label_mode}")


def task_config(run: wandb.apis.public.Run, task: str) -> dict[str, Any]:
    """Return the downstream task config stored on a W&B run."""
    return (
        run.config.get("trainer", {})
        .get("callbacks", {})
        .get("downstream_evaluator", {})
        .get("tasks", {})
        .get(task, {})
    )


def run_matches(run: wandb.apis.public.Run, run_prefix: str | None) -> bool:
    """Return whether a run should be included."""
    if run_prefix is None:
        return True
    return run.name.startswith(run_prefix)


def fetch_runs(project: str, run_prefix: str | None) -> list[wandb.apis.public.Run]:
    """Fetch matching W&B runs."""
    api = wandb.Api()
    runs = [
        run
        for run in api.runs(f"{WANDB_ENTITY}/{project}", lazy=False)
        if run_matches(run, run_prefix)
    ]
    return runs


def best_run_for_task(
    runs: list[wandb.apis.public.Run], task: str
) -> wandb.apis.public.Run | None:
    """Pick the run with the highest validation primary metric for a task."""
    best_run = None
    best_score = float("-inf")
    for run in runs:
        score = summary_metric(run.summary, primary_key(task, "val"))
        if score is None:
            continue
        if score > best_score:
            best_run = run
            best_score = score
    return best_run


def rows_for_task(
    run: wandb.apis.public.Run,
    task: str,
    zero_threshold: float,
) -> list[dict[str, Any]]:
    """Build per-class report rows for the selected task/run."""
    spec = TASK_SPECS[task]
    rows = []
    val_supported = set(supported_classes_for_split(spec, "valid"))
    test_supported = set(supported_classes_for_split(spec, "test"))
    supported = sorted(val_supported | test_supported)
    for class_id in supported:
        class_name = spec.class_names[class_id]
        val_f1 = class_f1(run.summary, task, class_id, "val")
        test_f1 = class_f1(run.summary, task, class_id, "test")
        split_values = []
        if class_id in val_supported and val_f1 is not None:
            split_values.append(val_f1)
        if class_id in test_supported and test_f1 is not None:
            split_values.append(test_f1)
        has_zero = any(value <= zero_threshold for value in split_values)
        missing = not split_values
        status = "missing" if missing else "zero" if has_zero else "ok"
        rows.append(
            {
                "task": task,
                "run_name": run.name,
                "run_id": run.id,
                "class_id": class_id,
                "class_name": class_name,
                "val_supported": class_id in val_supported,
                "val_f1": val_f1,
                "test_supported": class_id in test_supported,
                "test_f1": test_f1,
                "status": status,
            }
        )
    return rows


def print_task_summary(
    task: str, run: wandb.apis.public.Run, rows: list[dict[str, Any]]
) -> None:
    """Print a concise per-task summary."""
    val_primary = summary_metric(run.summary, primary_key(task, "val"))
    test_primary = summary_metric(run.summary, primary_key(task, "test"))
    cfg = task_config(run, task)
    bad_rows = [row for row in rows if row["status"] != "ok"]
    print(f"\n{task}")
    print(f"  run: {run.name} ({run.id})")
    print(f"  val primary: {val_primary}")
    print(f"  test primary: {test_primary}")
    if cfg:
        print(
            "  config: "
            f"eval_mode={cfg.get('eval_mode')} "
            f"probe_lr={cfg.get('probe_lr')} "
            f"epochs={cfg.get('epochs')} "
            f"pooling={cfg.get('pooling_type')}"
        )
    if not bad_rows:
        print("  all expected classes have non-zero per-class F1")
        return
    print("  classes needing review:")
    for row in bad_rows:
        print(
            "   "
            f"class {row['class_id']:>2} {row['class_name']:<22} "
            f"val_f1={format_supported_metric(row['val_f1'], row['val_supported'])} "
            f"test_f1={format_supported_metric(row['test_f1'], row['test_supported'])} "
            f"status={row['status']}"
        )


def format_metric(value: float | None) -> str:
    """Format optional metric values."""
    if value is None:
        return "missing"
    return f"{value:.4f}"


def format_supported_metric(value: float | None, supported: bool) -> str:
    """Format metric values while marking unsupported splits."""
    if not supported:
        return "n/a"
    return format_metric(value)


def write_csv(rows: list[dict[str, Any]], output: str) -> None:
    """Write all per-class rows to CSV."""
    fieldnames = [
        "task",
        "run_name",
        "run_id",
        "class_id",
        "class_name",
        "val_supported",
        "val_f1",
        "test_supported",
        "test_f1",
        "status",
    ]
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nWrote {output}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-p",
        "--project",
        default="2026_06_10_presto_osm_balanced_evals",
        help="W&B project under eai-ai2.",
    )
    parser.add_argument(
        "--run-prefix",
        default=None,
        help="Optional run-name prefix filter.",
    )
    parser.add_argument(
        "--tasks",
        default=",".join(PRESTO_OSM_TASKS),
        help="Comma-separated task names to inspect.",
    )
    parser.add_argument(
        "--zero-threshold",
        type=float,
        default=0.0,
        help="Treat per-class F1 <= this threshold as collapsed.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="presto_osm_class_f1_wandb.csv",
        help="CSV output path.",
    )
    return parser.parse_args()


def main() -> None:
    """Fetch W&B metrics and report zero/missing per-class F1."""
    args = parse_args()
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    unknown_tasks = sorted(set(tasks) - set(TASK_SPECS))
    if unknown_tasks:
        raise ValueError(f"Unknown Presto OSM task(s): {', '.join(unknown_tasks)}")

    runs = fetch_runs(project=args.project, run_prefix=args.run_prefix)
    print(f"Found {len(runs)} matching runs in {WANDB_ENTITY}/{args.project}")
    if not runs:
        raise SystemExit(1)

    all_rows = []
    for task in tasks:
        run = best_run_for_task(runs, task)
        if run is None:
            print(f"\n{task}")
            print("  no run has a validation primary metric for this task")
            continue
        rows = rows_for_task(
            run=run,
            task=task,
            zero_threshold=args.zero_threshold,
        )
        all_rows.extend(rows)
        print_task_summary(task, run, rows)

    if all_rows:
        write_csv(all_rows, args.output)


if __name__ == "__main__":
    main()

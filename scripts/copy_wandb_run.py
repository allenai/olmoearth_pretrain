#!/usr/bin/env python3
"""Copy specific metrics from one wandb run to another project, preserving step values."""

import argparse
from collections import defaultdict

from tqdm import tqdm

import wandb

METRICS = [
    "eval/pastis",
    "eval/mados",
    "eval/nandi_sentinel2",
    "eval/m_so2sat",
    "eval/m-eurosat",
    "eval/awf_sentinel2",
    "eval/awf_sentinel1",
    "eval/awf_landsat",
    "optim/LR (group 0)",
    "train/ModalityPatchDisc",
    "train/InfoNCE",
]


def copy_run(
    src_entity: str,
    src_project: str,
    run_id: str,
    dest_entity: str,
    dest_project: str,
    run_name: str | None = None,
    dry_run: bool = False,
):
    """Copy specific metrics from a source run to a new run in a destination project.

    Uses scan_history with keys filter to efficiently fetch only the needed metrics
    without downloading the entire run history.
    """
    api = wandb.Api()

    # Fetch source run
    src_run_path = f"{src_entity}/{src_project}/{run_id}"
    print(f"Fetching source run: {src_run_path}")
    src_run = api.run(src_run_path)

    # Get run metadata
    original_name = src_run.name
    original_config = {k: v for k, v in src_run.config.items() if not k.startswith("_")}
    original_tags = list(src_run.tags)

    print(f"Source run name: {original_name}")
    print(f"Tags: {original_tags}")

    # Fetch history for specific metrics using scan_history (unsampled, full fidelity)
    # We include _step to preserve step values
    keys_to_fetch = ["_step"] + METRICS
    print(f"\nFetching metrics: {METRICS}")
    print("Using scan_history for full fidelity (no sampling)...")

    # Collect all data points, grouped by step
    # Some metrics may be logged at different steps, so we group by step
    step_data: dict[int, dict[str, float]] = defaultdict(dict)

    # scan_history returns an iterator - efficient for large runs
    history_iter = src_run.scan_history(keys=keys_to_fetch, page_size=1000)

    row_count = 0
    for row in tqdm(history_iter, desc="Scanning history"):
        step = int(row.get("_step", 0))
        for metric in METRICS:
            if metric in row and row[metric] is not None:
                step_data[step][metric] = row[metric]
        row_count += 1

    print(f"\nScanned {row_count} history rows")
    print(f"Found {len(step_data)} unique steps with data")

    # Count metrics found
    metric_counts = defaultdict(int)
    for step, metrics in step_data.items():
        for metric in metrics:
            metric_counts[metric] += 1

    print("\nMetrics found:")
    for metric in METRICS:
        count = metric_counts.get(metric, 0)
        print(f"  {metric}: {count} data points")

    if dry_run:
        print("\n[DRY RUN] Would create run with the above data. Exiting.")
        return

    # Create new run in destination project
    dest_run_name = run_name or f"{original_name} (copied)"
    print(f"\nCreating new run in {dest_entity}/{dest_project}")
    print(f"Run name: {dest_run_name}")

    wandb.init(
        entity=dest_entity,
        project=dest_project,
        name=dest_run_name,
        config=original_config,
        tags=original_tags + ["copied-run"],
        notes=f"Copied from {src_run_path}",
    )

    # Log metrics preserving step values
    # Sort by step to log in order
    sorted_steps = sorted(step_data.keys())
    print(f"\nLogging {len(sorted_steps)} steps to new run...")

    for step in tqdm(sorted_steps, desc="Logging metrics"):
        metrics = step_data[step]
        if metrics:  # Only log if we have data for this step
            wandb.log(metrics, step=step)

    # Finish the run
    wandb.finish()
    print(f"\nDone! New run created in {dest_entity}/{dest_project}")


def main():
    """Run the script."""
    parser = argparse.ArgumentParser(
        description="Copy specific metrics from one wandb run to another project"
    )
    parser.add_argument("--src-entity", required=True, help="Source entity/team")
    parser.add_argument("--src-project", required=True, help="Source project name")
    parser.add_argument("--run-id", required=True, help="Source run ID")
    parser.add_argument("--dest-entity", required=True, help="Destination entity/team")
    parser.add_argument(
        "--dest-project", required=True, help="Destination project name"
    )
    parser.add_argument(
        "--run-name", help="Name for the new run (default: original name + ' (copied)')"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and display data without creating a new run",
    )

    args = parser.parse_args()

    copy_run(
        src_entity=args.src_entity,
        src_project=args.src_project,
        run_id=args.run_id,
        dest_entity=args.dest_entity,
        dest_project=args.dest_project,
        run_name=args.run_name,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

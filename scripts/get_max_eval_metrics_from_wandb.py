"""Get metrics summary from W&B. See get_max_metrics."""

import argparse
from collections import defaultdict

import pandas as pd
import wandb

from helios.internal.all_evals import EVAL_TASKS

WANDB_ENTITY = "eai-ai2"
METRICS = EVAL_TASKS.keys()

# Dataset partitions to consider (excluding default)
PARTITIONS = [
    "0.01x_train",
    # "0.02x_train",
    "0.05x_train",
    "0.10x_train",
    "0.20x_train",
    "0.50x_train",
]


def get_run_group_name(run_name: str) -> str:
    """Extracts the group name from a run name, e.g., 'my_experiment_step_100' -> 'my_experiment'."""
    # just split on _step and take the first part
    return run_name.split("_step")[0]


def get_max_metrics_grouped(
    project_name: str, run_prefix: str | None = None
) -> dict[str, dict[str, float]]:
    """Get the maximum value for each metric grouped by run prefix before '_step'.

    Args:
        project_name: the W&B project for the run.
        run_prefix: optional prefix to filter runs. If None, processes all runs.

    Returns:
        a dictionary mapping from group name to a dict of metric name to max value.
    """
    api = wandb.Api()

    # Group runs by their prefix before "_step"
    grouped_runs = defaultdict(list)
    for run in api.runs(f"{WANDB_ENTITY}/{project_name}"):
        if run_prefix and not run.name.startswith(run_prefix):
            continue
        group_name = get_run_group_name(run.name)
        grouped_runs[group_name].append(run)
        print(f"Found run {run.name} ({run.id}) -> group: {group_name}")

    print(f"\nFound {len(grouped_runs)} groups")

    # print all the groups found and stop here
    print(f"\nGroups found: {grouped_runs.keys()}")

    # Get max metrics for each group
    group_metrics = {}
    for group_name, runs in grouped_runs.items():
        print(f"\nProcessing group: {group_name} ({len(runs)} runs)")
        metrics = {}
        for run in runs:
            for key, value in run.summary.items():
                if not key.startswith("eval/"):
                    continue
                metrics[key] = max(metrics.get(key, value), value)
        group_metrics[group_name] = metrics

    return group_metrics


def get_max_metrics_per_partition(
    project_name: str, run_prefix: str
) -> dict[str, dict[str, float]]:
    """Get the maximum value for each metric per dataset partition (excluding default).

    This function finds runs for each partition and computes the maximum for each metric
    within each partition separately.

    Args:
        project_name: the W&B project for the run.
        run_prefix: the prefix to search for. We will compute the maximum for each
            metric across all runs sharing this prefix within each partition.

    Returns:
        a dictionary mapping from partition to a dict of metric name to max value.
    """
    api = wandb.Api()

    # Dictionary to store max metrics for each partition
    partition_metrics = {}

    # For each partition, find runs and get max metrics
    for partition in PARTITIONS:
        print(f"\nProcessing partition: {partition}")

        # List all the runs in the project and find the subset matching the prefix and partition
        run_ids: list[str] = []
        for run in api.runs(f"{WANDB_ENTITY}/{project_name}"):
            if not run.name.startswith(run_prefix):
                continue
            # Check if run name contains the partition
            if partition not in run.name:
                continue
            print(f"Found run {run.name} ({run.id}) for partition {partition}")
            run_ids.append(run.id)

        if not run_ids:
            print(f"No runs found for partition {partition}")
            continue

        print(
            f"Found {len(run_ids)} runs with prefix {run_prefix} and partition {partition}"
        )

        # Get the metrics for each run in this partition, and save max across runs
        partition_max_metrics = {}
        for run_id in run_ids:
            run = api.run(f"{WANDB_ENTITY}/{project_name}/{run_id}")
            for key, value in run.summary.items():
                if not key.startswith("eval/"):
                    continue
                partition_max_metrics[key] = max(
                    partition_max_metrics.get(key, value), value
                )

        partition_metrics[partition] = partition_max_metrics

    return partition_metrics


def get_max_metrics(project_name: str, run_prefix: str) -> dict[str, float]:
    """Get the maximum value for each metric across runs sharing the prefix.

    This assumes you have run a sweep like scripts/2025_06_23_naip/eval_sweep.py and now
    want to get the maximum for each metric across probe learning rates.

    Args:
        project_name: the W&B project for the run.
        run_prefix: the prefix to search for. We will compute the maximum for each
            metric across all runs sharing this prefix.

    Returns:
        a dictionary mapping from the metric name to the max value.
    """
    api = wandb.Api()

    # List all the runs in the project and find the subset matching the prefix.
    run_ids: list[str] = []
    for run in api.runs(f"{WANDB_ENTITY}/{project_name}"):
        if not run.name.startswith(run_prefix):
            continue
        print(f"Found run {run.name} ({run.id})")
        run_ids.append(run.id)
    print(f"Found {len(run_ids)} runs with prefix {run_prefix}")

    # Get the metrics for each run, and save max across runs.
    metrics = {}
    for run_id in run_ids:
        run = api.run(f"{WANDB_ENTITY}/{project_name}/{run_id}")
        for key, value in run.summary.items():
            if not key.startswith("eval/"):
                continue
            metrics[key] = max(metrics.get(key, value), value)
    return metrics


def save_metrics_to_csv(metrics_dict: dict[str, dict[str, float]], filename: str):
    """Saves the metrics dictionary to a CSV file."""
    all_metrics_df = pd.DataFrame()
    # first column should be group name and then the rest of the columns should be the metric names
    all_metrics_df["group"] = metrics_dict.keys()
    for metric_name in metrics_dict[list(metrics_dict.keys())[0]].keys():
        all_metrics_df[metric_name] = [
            metrics_dict[group_name][metric_name] for group_name in metrics_dict.keys()
        ]
    print(all_metrics_df.head())
    all_metrics_df.to_csv(filename, index=False)
    print(f"\nMetrics saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get maximum metrics from W&B runs, grouped by run prefix before '_step'."
    )
    parser.add_argument(
        "-p", "--project_name", type=str, help="W&B project name under eai-ai2 entity"
    )
    parser.add_argument(
        "--run_prefix",
        type=str,
        default=None,
        help="Optional prefix to filter runs (e.g., 'my_experiment'). If not specified, processes all runs.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: {project_name}_eval_metrics.csv or {run_prefix}_eval_metrics.csv)",
    )
    parser.add_argument(
        "--per-partition",
        action="store_true",
        help="Aggregate metrics per dataset partition instead of grouping by '_step'",
    )

    args = parser.parse_args()

    if args.per_partition:
        if not args.run_prefix:
            parser.error("--per-partition requires run_prefix to be specified")
        print("Getting max metrics per dataset partition (excluding default)...")
        partition_metrics = get_max_metrics_per_partition(
            args.project_name, args.run_prefix
        )

        print("\nResults per partition:")
        rows = []  # for CSV: partition, metric, value
        for partition in PARTITIONS:
            if partition in partition_metrics:
                print(f"\n{partition}:")
                for metric in METRICS:
                    # Try original name
                    key = f"eval/{metric}"
                    val = partition_metrics[partition].get(key)
                    # Fallback with underscore variant
                    if val is None:
                        metric_alt = metric.replace("-", "_")
                        key_alt = f"eval/{metric_alt}"
                        val = partition_metrics[partition].get(key_alt)
                        name_for_print = metric_alt if val is not None else metric
                    else:
                        name_for_print = metric

                    if val is None:
                        print(f"  {metric}: not found")
                        rows.append((partition, metric, "not found"))
                    else:
                        print(f"  {name_for_print}: {val}")
                        rows.append((partition, name_for_print, val))
            else:
                print(f"\n{partition}: no runs found")

        with open(args.output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["partition", "metric", "value"])
            writer.writerows(rows)
        print(f"\nPer-partition metrics written to {args.output_file}")

    else:
        if args.run_prefix:
            print(
                f"Getting max metrics grouped by run prefix before '_step' (filtering by '{args.run_prefix}')..."
            )
        else:
            print(
                "Getting max metrics grouped by run prefix before '_step' (all runs in project)..."
            )

        group_metrics = get_max_metrics_grouped(args.project_name, args.run_prefix)

        print("\nFinal Results:")
        for group_name, metrics in group_metrics.items():
            print(f"\n{group_name}:")
            for metric in METRICS:
                try:
                    k = f"eval/{metric}"
                    print(f"  {metric}: {metrics[k]}")
                except KeyError:
                    try:
                        metric = metric.replace("-", "_")
                        k = f"eval/{metric}"
                        print(f"  {metric}: {metrics[k]}")
                    except KeyError:
                        print(f"  {metric}: not found")

        # Save to CSV
        if args.output:
            output_csv = args.output
        elif args.run_prefix:
            output_csv = f"{args.run_prefix}_eval_metrics.csv"
        else:
            output_csv = f"{args.project_name}_eval_metrics.csv"
        save_metrics_to_csv(group_metrics, output_csv)

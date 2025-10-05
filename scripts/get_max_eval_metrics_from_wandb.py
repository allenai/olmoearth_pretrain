"""Get metrics summary from W&B. See get_max_metrics."""

import argparse
import csv

import wandb

WANDB_ENTITY = "eai-ai2"
METRICS = [
    "m-eurosat",
    "m-forestnet",
    "m-so2sat",
    "m-brick-kiln",
    "m-bigearthnet",
    "m-sa-crop-type",
    "m-cashew-plant",
    "pastis_sentinel1",
    "pastis_sentinel2",
    "pastis_sentinel1_sentinel2",
    "mados",
    "sen1floods11",
    "breizhcrops",
]

# Dataset partitions to consider (excluding default)
PARTITIONS = [
    "0.01x_train",
    # "0.02x_train",
    "0.05x_train",
    "0.10x_train",
    "0.20x_train",
    "0.50x_train",
]


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


def main():
    """Parse args and call get_max_metrics or get_max_metrics_per_partition."""
    parser = argparse.ArgumentParser(
        description="Aggregate eval metrics and write CSV."
    )
    parser.add_argument("project_name", help="W&B project under eai-ai2")
    parser.add_argument("run_prefix", help="Run prefix (or exact run name if unique)")
    parser.add_argument(
        "-o",
        "--output-file",
        default="final_metrics.csv",
        help="Path to output CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--per-partition",
        action="store_true",
        help="Aggregate per dataset partition (excludes 'default')",
    )
    args = parser.parse_args()

    if args.per_partition:
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
        print("Getting max metrics across runs...")
        metrics = get_max_metrics(args.project_name, args.run_prefix)

        print("\nFinal Results:")
        rows = []
        for metric in METRICS:
            key = f"eval/{metric}"
            val = metrics.get(key)
            if val is None:
                metric_alt = metric.replace("-", "_")
                key_alt = f"eval/{metric_alt}"
                val = metrics.get(key_alt)
                name_for_print = metric_alt if val is not None else metric
            else:
                name_for_print = metric

            if val is None:
                print(f"Metric {metric} not found")
                rows.append((metric, "not found"))
            else:
                print(f"{name_for_print} {val}")
                rows.append((name_for_print, val))

        with open(args.output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerows(rows)

        print(f"\nMetrics written to {args.output_file}")


if __name__ == "__main__":
    main()

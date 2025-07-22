"""Get metrics summary from W&B. See get_max_metrics."""

import sys

import wandb

WANDB_ENTITY = "eai-ai2"
METRICS = [
    "m-eurosat",
    "m-so2sat",
    "m-brick-kiln",
    "m-bigearthnet",
    "m-sa-crop-type",
    "m-cashew-plant",
    "sickle-sentinel1",
    "sickle_landsat",
    "sickle_sentinel1_landsat",
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


if __name__ == "__main__":
    # project_name is the W&B project under eai-ai2
    project_name = sys.argv[1]
    # run_prefix is the prefix of the runs with different probe learning rates.
    # If you are not sweeping across probe learning rates, this can just be the name
    # of the run (as long as no other run shares the same prefix).
    run_prefix = sys.argv[2]

    # Check if user wants partition-based aggregation
    if len(sys.argv) > 3 and sys.argv[3] == "--per-partition":
        print("Getting max metrics per dataset partition (excluding default)...")
        partition_metrics = get_max_metrics_per_partition(project_name, run_prefix)

        print("\nResults per partition:")
        for partition in PARTITIONS:
            if partition in partition_metrics:
                print(f"\n{partition}:")
                for metric in METRICS:
                    try:
                        k = f"eval/{metric}"
                        print(f"  {metric}: {partition_metrics[partition][k]}")
                    except KeyError:
                        try:
                            metric = metric.replace("-", "_")
                            k = f"eval/{metric}"
                            print(f"  {metric}: {partition_metrics[partition][k]}")
                        except KeyError:
                            print(f"  {metric}: not found")
            else:
                print(f"\n{partition}: no runs found")
    else:
        print("Getting max metrics across runs...")
        metrics = get_max_metrics(project_name, run_prefix)

        print("\nFinal Results:")
        for metric in METRICS:
            try:
                k = f"eval/{metric}"
                print(f"{metric} {metrics[k]}")
            except KeyError:
                try:
                    metric = metric.replace("-", "_")
                    k = f"eval/{metric}"
                    print(f"{metric} {metrics[k]}")
                except KeyError:
                    print(f"Metric {metric} not found")

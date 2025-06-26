"""Get metrics summary from W&B. See get_max_metrics."""

import sys

import wandb

WANDB_ENTITY = "eai-ai2"
METRICS = [
    "m-eurosat",
    "m-so2sat",
    "m-brick-kiln",
    "m-bigearthnet",
    "m_sa_crop_type",
    "m_cashew_plant",
    "sickle_sentinel1",
    "sickle_landsat",
    "sickle_sentinel1_landsat",
    "pastis_sentinel1",
    "pastis_sentinel2",
    "pastis_sentinel1_sentinel2",
    "mados",
    "sen1floods11",
    "breizhcrops",
]


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

    metrics = get_max_metrics(project_name, run_prefix)
    for metric in METRICS:
        k = f"eval/{metric}"
        print(f"{metric} {metrics[k]}")

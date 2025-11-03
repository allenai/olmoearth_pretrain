"""Get metrics summary from W&B. See get_max_metrics."""

import argparse
import csv
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import wandb

from olmoearth_pretrain.evals.models import (
    MODELS_WITH_MULTIPLE_SIZES,
    BaselineModelName,
)
from olmoearth_pretrain.internal.all_evals import EVAL_TASKS, FT_EVAL_TASKS
from olmoearth_pretrain.train.callbacks.evaluator_callback import EvalMode

WANDB_ENTITY = "eai-ai2"

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


def get_run_groups(
    project_name: str,
    run_prefix: str | None = None,
    group_baseline_model_and_size: bool = False,
) -> dict[str, dict[str, float]]:
    """Get the maximum value for each metric grouped by run prefix before '_step'.

    Args:
        project_name: the W&B project for the run.
        run_prefix: optional prefix to filter runs. If None, processes all runs.
        group_baseline_model_and_size: if True, group by baseline model name and model size key instead of run prefix before '_step'.

    Returns:
        a dictionary mapping from group name to a dict of metric name to max value.
    """
    api = wandb.Api()
    wandb_path = f"{WANDB_ENTITY}/{project_name}"

    if not group_baseline_model_and_size:
        grouped_runs = group_runs_by_run_prefix_and_step(api, wandb_path, run_prefix)
    else:
        grouped_runs = group_runs_by_baseline_model_and_size(api, wandb_path)

    print(f"\nFound {len(grouped_runs)} groups")

    # print all the groups found and stop here
    print(f"\nGroups found: {grouped_runs.keys()}")
    return grouped_runs


def group_runs_by_run_prefix_and_step(
    api: wandb.Api, wandb_path: str, run_prefix: str | None = None
) -> dict[str, list[wandb.Run]]:
    """Group runs by their prefix before "_step".

    Args:
        api: the W&B API object.
        wandb_path: the W&B path for the run.
        run_prefix: optional prefix to filter runs. If None, processes all runs.

    Returns:
        a dictionary mapping from group name to a list of wandb.Run objects.
    """
    grouped_runs = defaultdict(list)
    for run in api.runs(wandb_path, lazy=False):
        if run_prefix and not run.name.startswith(run_prefix):
            continue
        group_name = get_run_group_name(run.name)
        grouped_runs[group_name].append(run)
        print(f"Found run {run.name} ({run.id}) -> group: {group_name}")
    return grouped_runs


def group_runs_by_baseline_model_and_size(
    api: wandb.Api, wandb_path: str
) -> dict[str, list[wandb.Run]]:
    """Group runs by their baseline model name and model size key."""

    def _find_model_name_and_size(run: wandb.Run) -> tuple[BaselineModelName, str]:
        """Find the baseline model name and size key in the run config."""
        for name in list(BaselineModelName):
            if name.value in run.name:
                model_config = run.config["model"]
                print(f"Model config: {model_config} type: {type(model_config)}")
                return name, model_config.get("size", None)
        raise ValueError(f"No baseline model name found in run {run.name}")

    def _get_group_name(model_name: BaselineModelName, size: str | None) -> str:
        """Get the group name for the run."""
        if size is None:
            return model_name.value
        return f"{model_name.value}_{size}"

    grouped_runs = defaultdict(list)
    for run in api.runs(wandb_path, lazy=False):
        print(f"Processing run {run.name} ({run.id})")
        model_name, size = _find_model_name_and_size(run)
        if model_name in MODELS_WITH_MULTIPLE_SIZES and size is None:
            print(
                f"Skipping run {run.name} ({run.id}) because it has no size specified and is a model with multiple sizes"
            )
            continue
        group_name = _get_group_name(model_name, size)
        grouped_runs[group_name].append(run)
        print(f"Found run {run.name} ({run.id}) -> group: {group_name}")
    return grouped_runs


def _get_corresponding_test_key(key: str) -> str:
    """Get the corresponding test key for a given metric key."""
    return key.replace("eval/", "eval/test/")


def get_max_metrics_grouped(
    grouped_runs: dict[str, list[wandb.Run]],
    get_test_metrics: bool = False,
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    dict[str, dict[str, wandb.Run]],
]:
    """Get max metrics for each group."""
    # Get max metrics for each group
    group_metrics = {}
    group_max_runs_per_metric = {}
    for group_name, runs in grouped_runs.items():
        print(f"\nProcessing group: {group_name} ({len(runs)} runs)")
        #  Get the run that has test metrics with the highest validation score for each metric
        metrics = {}
        max_runs_per_metric = {}
        for run in runs:
            for key, value in run.summary.items():
                # TODO: Make these metrics names constants
                if not key.startswith("eval/"):
                    continue
                if key.startswith("eval/test/"):
                    print(
                        f"Skipping test metric {key} for run {run.name} because it is a test metric"
                    )
                    # DO NOT select on test metrics
                    continue
                # Ensure the run has test metrics
                if run.summary.get(_get_corresponding_test_key(key), None) is None:
                    # DOn't select top val metrics if there is no corresponding test metric
                    print(
                        f"Skipping metric {key} for run {run.name} because it has no corresponding test metric"
                    )
                    continue

                # If for the given metric, it is a linear probe task skip if it was not done with early stop linear porbing
                task_name = key.split("/")[1]
                task_config = run.config["trainer"]["callbacks"][
                    "downstream_evaluator"
                ]["tasks"][task_name]

                eval_mode = task_config.get("eval_mode", None)
                is_linear_probe_task = (
                    EvalMode(eval_mode.lower()) == EvalMode.LINEAR_PROBE
                    if eval_mode is not None
                    else False
                )
                is_select_final_test_miou_based_on_epoch_of_max_val_miou = (
                    task_config.get(
                        "select_final_test_miou_based_on_epoch_of_max_val_miou", False
                    )
                )
                if (
                    is_linear_probe_task
                    and not is_select_final_test_miou_based_on_epoch_of_max_val_miou
                ):
                    print(
                        f"Skipping metric {key} for run {run.name} because it is a linear probe task but not done with early stop linear probing"
                    )
                    continue
                print(
                    f"Selecting metric {key} for run {run.name} because it matches criteria"
                )

                prev_max_val = metrics.get(key, float("-inf"))
                metrics[key] = max(prev_max_val, value)
                if value > prev_max_val:
                    max_runs_per_metric[key] = run

        group_metrics[group_name] = metrics
        group_max_runs_per_metric[group_name] = max_runs_per_metric

    grouped_test_metrics = {}
    if get_test_metrics:
        print("\nGetting test metrics...")
        # get the test set values for all the max runs per metric

        for group_name, max_runs_per_metric in group_max_runs_per_metric.items():
            test_metrics = {}
            for metric, run in max_runs_per_metric.items():
                test_metric_key = metric.replace("eval/", "eval/test/")
                value = run.summary.get(test_metric_key, None)
                if value is None:
                    print(
                        f"No test metric found for run {run.name} for metric {metric}"
                    )
                    continue
                print(
                    f"Found test metric {test_metric_key} for run {run.name} with value {value}"
                )
                test_metrics[test_metric_key] = value
            grouped_test_metrics[group_name] = test_metrics
    return group_metrics, grouped_test_metrics, group_max_runs_per_metric


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
    api = wandb.Api(timeout=10000)

    # Dictionary to store max metrics for each partition
    partition_metrics = {}

    # For each partition, find runs and get max metrics
    for partition in PARTITIONS:
        print(f"\nProcessing partition: {partition}")

        # List all the runs in the project and find the subset matching the prefix and partition
        run_ids: list[str] = []
        for run in api.runs(f"{WANDB_ENTITY}/{project_name}", lazy=False):
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
    for run in api.runs(f"{WANDB_ENTITY}/{project_name}", lazy=False):
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
    all_groups = list(metrics_dict.keys())
    # Collect all unique metric names across all groups
    all_metric_names = set()
    for group_metrics in metrics_dict.values():
        all_metric_names.update(group_metrics.keys())
    all_metric_names = sorted(all_metric_names)

    # Build rows, using np.nan if a metric is missing for a group
    rows = []
    for group in all_groups:
        row = {"group": group}
        for metric in all_metric_names:
            row[metric] = metrics_dict[group].get(metric, np.nan)
        rows.append(row)

    all_metrics_df = pd.DataFrame(rows)
    print(all_metrics_df.head())
    all_metrics_df.to_csv(filename, index=False)
    print(f"\nMetrics saved to {filename}")


def serialize_max_settings_per_group(
    json_filename: str, group_max_runs_per_metric: dict[str, dict[str, wandb.Run]]
) -> None:
    """Serialize the max settings per group."""
    output_dict = {}
    # I want  it to be group name -> metric name -> run settings
    # Run settings should include whether we are doing mean or max pooling
    # what lr we are using if it linear probing
    # whether or not we used pretrained normalizer
    for group_name, max_runs_per_metric in group_max_runs_per_metric.items():
        for metric, run in max_runs_per_metric.items():
            task_name = metric.replace("eval/", "")

            # Ensure nested structure exists
            output_dict.setdefault(group_name, {}).setdefault(task_name, {})

            run_settings = {}
            task_config = run.config["trainer"]["callbacks"]["downstream_evaluator"][
                "tasks"
            ][task_name]
            run_settings["pooling_type"] = task_config["pooling_type"]
            run_settings["probe_lr"] = task_config.get("probe_lr", None)
            run_settings["norm_stats_from_pretrained"] = task_config[
                "norm_stats_from_pretrained"
            ]
            output_dict[group_name][task_name]["settings"] = run_settings
            output_dict[group_name][task_name]["run_id"] = run.id

    # Save the output dict to a JSON file
    with open(json_filename, "w") as f:
        json.dump(output_dict, f)
    return output_dict


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
    # pull and group by baseline model name and model size key
    parser.add_argument(
        "--group_baseline_model_and_size",
        action="store_true",
        help="Group by baseline model name and model size key",
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
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Use finetune evaluation tasks when determining metrics",
    )
    parser.add_argument(
        "--get_test_metrics",
        action="store_true",
        help="Report test metrics based on the configuration of the validation results witht the highest score",
    )
    parser.add_argument(
        "--json_filename",
        type=str,
        default=None,
        help="Output JSON file path (default: {project_name}_eval_metrics.json)",
    )

    args = parser.parse_args()
    metrics = list(FT_EVAL_TASKS.keys()) if args.finetune else list(EVAL_TASKS.keys())

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
                for metric in metrics:
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
        print(f"Running with the following arguments: {args}")
        run_groups = get_run_groups(
            args.project_name, args.run_prefix, args.group_baseline_model_and_size
        )
        group_metrics, group_test_metrics, group_max_runs_per_metric = (
            get_max_metrics_grouped(run_groups, args.get_test_metrics)
        )
        print(group_max_runs_per_metric)
        if args.json_filename:
            serialize_max_settings_per_group(
                args.json_filename, group_max_runs_per_metric
            )
        print("\nFinal Results:")
        for group_name, metrics in group_metrics.items():
            print(f"\n{group_name}:")
            for metric in metrics:
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
        if args.get_test_metrics:
            print("\nFinal Test Results:")
            for group_name, metrics in group_test_metrics.items():
                print(f"\n{group_name}:")
                for metric in metrics:
                    try:
                        k = f"eval/test/{metric}"
                        print(f"  {metric}: {metrics[k]}")
                    except KeyError:
                        try:
                            metric = metric.replace("-", "_")
                            k = f"eval/test/{metric}"
                            print(f"  {metric}: {metrics[k]}")
                        except KeyError:
                            print(f"  {metric}: not found")

        # Save to CSV
        if args.output:
            output_csv = args.output
            test_output_csv = args.output.replace(".csv", "_test.csv")
        elif args.run_prefix:
            output_csv = f"{args.run_prefix}_eval_metrics.csv"
            test_output_csv = f"{args.run_prefix}_eval_metrics_test.csv"
        else:
            output_csv = f"{args.project_name}_eval_metrics.csv"
            test_output_csv = f"{args.project_name}_eval_metrics_test.csv"
        save_metrics_to_csv(group_metrics, output_csv)
        if args.get_test_metrics:
            save_metrics_to_csv(group_test_metrics, test_output_csv)

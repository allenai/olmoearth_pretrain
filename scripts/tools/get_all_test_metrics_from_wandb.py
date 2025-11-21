"""Get all test metrics from W&B for all runs and all tasks.

This script retrieves test metrics (eval/test/*) from all runs in a W&B project
and outputs them in a CSV with run names as rows and metrics as columns.
"""

import argparse

import numpy as np
import pandas as pd
import wandb

WANDB_ENTITY = "eai-ai2"


def get_run_group_name(run_name: str) -> str:
    """Extract the group name from a run name by splitting on '_from_json'.

    Args:
        run_name: the full run name.

    Returns:
        The group name (part before '_from_json'), or the full run name if '_from_json' not found.
    """
    if "_from_json" in run_name:
        return run_name.split("_from_json")[0]
    return run_name


def get_all_test_metrics(
    project_name: str,
    run_prefix: str | None = None,
    include_val_metrics: bool = False,
    include_bootstrap_metrics: bool = False,
) -> pd.DataFrame:
    """Get all test metrics for all runs in a project.

    Args:
        project_name: the W&B project name.
        run_prefix: optional prefix to filter runs. If None, processes all runs.
        include_val_metrics: if True, also include validation metrics (eval/*) in addition to test metrics.
        include_bootstrap_metrics: if True, also include bootstrap metrics (metrics with 'bootstrap' in the name).

    Returns:
        A pandas DataFrame with run names as the index and metrics as columns.
    """
    api = wandb.Api(timeout=10000)
    wandb_path = f"{WANDB_ENTITY}/{project_name}"

    print(f"Fetching runs from {wandb_path}...")
    runs = api.runs(wandb_path, lazy=False)

    # Collect all test metrics for each run
    run_metrics = {}
    all_metric_names = set()

    for run in runs:
        if run_prefix and not run.name.startswith(run_prefix):
            continue

        print(f"Processing run: {run.name} ({run.id})")
        metrics = {}

        for key, value in run.summary.items():
            # Skip bootstrap metrics unless explicitly requested
            is_bootstrap_metric = "bootstrap" in key.lower()
            if is_bootstrap_metric and not include_bootstrap_metrics:
                continue

            # Only consider test metrics (eval/test/*)
            if key.startswith("eval/test/"):
                metrics[key] = value
                all_metric_names.add(key)
            # Optionally include validation metrics
            elif (
                include_val_metrics
                and key.startswith("eval/")
                and not key.startswith("eval/test/")
            ):
                metrics[key] = value
                all_metric_names.add(key)

        if metrics:
            run_metrics[run.name] = metrics
            print(f"  Found {len(metrics)} metrics")
        else:
            print("  No test metrics found")

    if not run_metrics:
        print("\nNo test metrics found for any runs!")
        return pd.DataFrame()

    print(f"\nTotal runs with metrics: {len(run_metrics)}")
    print(f"Total unique metrics: {len(all_metric_names)}")

    # Build DataFrame
    # Sort metric names for consistent ordering
    all_metric_names = sorted(all_metric_names)

    rows = []
    for run_name, metrics in run_metrics.items():
        row = {
            "run_name": run_name,
            "group": get_run_group_name(run_name),
        }
        for metric in all_metric_names:
            row[metric] = metrics.get(metric, np.nan)
        rows.append(row)

    df = pd.DataFrame(rows)
    # Set run_name and group as the leftmost columns
    df = df[["run_name", "group"] + all_metric_names]

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get all test metrics from W&B runs for all tasks."
    )
    parser.add_argument(
        "-p",
        "--project_name",
        type=str,
        required=True,
        help="W&B project name under eai-ai2 entity",
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
        help="Output CSV file path (default: {project_name}_all_test_metrics.csv or {run_prefix}_all_test_metrics.csv)",
    )
    parser.add_argument(
        "--include_val_metrics",
        action="store_true",
        help="Also include validation metrics (eval/*) in addition to test metrics",
    )
    parser.add_argument(
        "--include_bootstrap_metrics",
        action="store_true",
        help="Also include bootstrap metrics (metrics with 'bootstrap' in the name)",
    )

    args = parser.parse_args()

    print(f"Running with the following arguments: {args}")

    # Get all test metrics
    df = get_all_test_metrics(
        args.project_name,
        args.run_prefix,
        args.include_val_metrics,
        args.include_bootstrap_metrics,
    )

    if df.empty:
        print("\nNo data to save.")
        exit(1)

    # Determine output file path
    if args.output:
        output_csv = args.output
    elif args.run_prefix:
        output_csv = f"{args.run_prefix}_all_test_metrics.csv"
    else:
        output_csv = f"{args.project_name}_all_test_metrics.csv"

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\nAll test metrics saved to {output_csv}")
    print(f"Shape: {df.shape[0]} runs × {df.shape[1] - 2} metrics")
    print("\nFirst few rows:")
    print(df.head())

"""Script to run locally that scrapes throughput run info from wandb."""

import argparse

import pandas as pd
import wandb

from helios.inference_benchmarking import constants
from helios.inference_benchmarking.constants import PARAM_KEYS
from helios.inference_benchmarking.data_models import RunParams


def main(
    prefix: str | None, suffix: str | None, group: str | None, output_file: str
) -> None:
    """Performs the scraping."""
    # Need to be able to select a group of runs to scrape
    wandb_api = wandb.Api()  # make sure env has WANDB_API_KEY

    all_runs = wandb_api.runs(f"{constants.ENTITY_NAME}/{constants.PROJECT_NAME}")
    all_history = []

    expected_metrics = [
        constants.PER_BATCH_TOKEN_RATE_METRIC,
        constants.MEAN_BATCH_TOKEN_RATE_METRIC,
        constants.MEAN_BATCH_TIME_METRIC,
        constants.NUM_TOKENS_PER_BATCH_METRIC,
        constants.SQUARE_KM_PER_SECOND_METRIC,
        constants.PIXELS_PER_SECOND_METRIC,
    ]

    for run in all_runs:
        if group is not None and run.group != group:
            continue
        if prefix is not None and not run.name.startswith(prefix):
            continue
        if suffix is not None and not run.name.endswith(suffix):
            continue
        print(run.name)
        history_df = run.history()
        if history_df.empty:
            continue
        if not all(metric in history_df.columns for metric in expected_metrics):
            continue
        print(history_df.head())
        name = run.name
        params = RunParams.from_run_name(name)

        # Initialize all metrics with NaN values
        # history_df[PARAM_KEYS["batch_size"]] = float("nan")
        # history_df[constants.PER_BATCH_TOKEN_RATE_METRIC] = float("nan")
        # history_df[constants.MEAN_BATCH_TOKEN_RATE_METRIC] = float("nan")
        # history_df[constants.MEAN_BATCH_TIME_METRIC] = float("nan")
        # history_df[constants.NUM_TOKENS_PER_BATCH_METRIC] = float("nan")
        # history_df[constants.SQUARE_KM_PER_SECOND_METRIC] = float("nan")
        # history_df[constants.PIXELS_PER_SECOND_METRIC] = float("nan")
        # history_df[constants.OOM_OCCURRED_METRIC] = float("nan")

        # Set the single batch size for all rows
        history_df[PARAM_KEYS["batch_size"]] = params.batch_size

        # Add all parameter columns
        history_df[PARAM_KEYS["model_size"]] = params.model_size
        history_df[PARAM_KEYS["model_size"]] = pd.Categorical(
            history_df[PARAM_KEYS["model_size"]],
            categories=["nano", "tiny", "base", "large", "giga"],
            ordered=True,
        )
        history_df[PARAM_KEYS["gpu_type"]] = params.gpu_type
        history_df[PARAM_KEYS["bf16"]] = params.bf16
        history_df[PARAM_KEYS["image_size"]] = params.image_size
        history_df[PARAM_KEYS["patch_size"]] = params.patch_size
        history_df[PARAM_KEYS["num_timesteps"]] = params.num_timesteps
        history_df[PARAM_KEYS["use_s1"]] = params.use_s1
        history_df[PARAM_KEYS["use_s2"]] = params.use_s2
        history_df[PARAM_KEYS["use_landsat"]] = params.use_landsat
        history_df["run_url"] = run.url

        # Calculate derived metrics
        history_df["tokens_per_instance"] = (
            history_df[constants.NUM_TOKENS_PER_BATCH_METRIC]
            / history_df[PARAM_KEYS["batch_size"]]
        )

        # Select and order columns for final output, add OOM_OCCURRED_METRIC if it exists
        columns = [
            PARAM_KEYS["gpu_type"],
            constants.GPU_NAME_METRIC,
            PARAM_KEYS["model_size"],
            PARAM_KEYS["bf16"],
            PARAM_KEYS["image_size"],
            PARAM_KEYS["patch_size"],
            PARAM_KEYS["num_timesteps"],
            PARAM_KEYS["batch_size"],
            "tokens_per_instance",
            constants.NUM_TOKENS_PER_BATCH_METRIC,
            constants.MEAN_BATCH_TIME_METRIC,
            constants.MEAN_BATCH_TOKEN_RATE_METRIC,
            constants.PER_BATCH_TOKEN_RATE_METRIC,
            constants.SQUARE_KM_PER_SECOND_METRIC,
            constants.PIXELS_PER_SECOND_METRIC,
            PARAM_KEYS["use_s1"],
            PARAM_KEYS["use_s2"],
            PARAM_KEYS["use_landsat"],
            "run_url",
        ]
        # Add OOM_OCCURRED_METRIC if it exists in the DataFrame
        if constants.OOM_OCCURRED_METRIC in history_df.columns:
            columns.insert(
                columns.index(constants.PIXELS_PER_SECOND_METRIC) + 1,
                constants.OOM_OCCURRED_METRIC,
            )
        history_df = history_df[columns]
        # remove per batch token rate metric as it is a chart
        history_df = history_df.drop(columns=[constants.PER_BATCH_TOKEN_RATE_METRIC])
        all_history.append(history_df)

    all_history_as_pd = pd.concat(all_history, axis=0)
    all_history_as_pd = all_history_as_pd.sort_values(
        by=[
            PARAM_KEYS["gpu_type"],
            PARAM_KEYS["model_size"],
            PARAM_KEYS["bf16"],
            PARAM_KEYS["image_size"],
            PARAM_KEYS["patch_size"],
            PARAM_KEYS["num_timesteps"],
            PARAM_KEYS["batch_size"],
        ]
    )
    all_history_as_pd.to_csv(output_file, index=False)
    print(f"Exported {len(all_history_as_pd)} rows to inference_throughput.csv")


if __name__ == "__main__":
    # parse a prefix from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, required=False, default=None)
    parser.add_argument("--suffix", type=str, required=False, default=None)
    parser.add_argument("--group", type=str, required=False, default=None)
    parser.add_argument(
        "--output_file", type=str, required=False, default="inference_throughput.csv"
    )
    args = parser.parse_args()
    main(args.prefix, args.suffix, args.group, args.output_file)

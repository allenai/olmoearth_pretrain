import pandas as pd
import wandb

from helios.inference_benchmarking import constants
from helios.inference_benchmarking.constants import PARAM_KEYS
from helios.inference_benchmarking.data_models import RunParams


def main() -> None:
    wandb_api = wandb.Api()  # make sure env has WANDB_API_KEY

    all_runs = wandb_api.runs(f"{constants.ENTITY_NAME}/{constants.PROJECT_NAME}")
    all_history = []

    for run in all_runs:
        history_df = run.history()
        name = run.name
        params = RunParams.from_run_name(name)

        history_df[PARAM_KEYS["batch_size"]] = float("nan")

        for index, batch_size in enumerate(params.batch_sizes):
            if len(history_df) <= index:
                history_df = pd.concat(
                    [
                        history_df,
                        pd.DataFrame({
                            constants.MEAN_BATCH_TOKEN_RATE_METRIC: [float("nan")],
                            constants.MEAN_BATCH_TIME_METRIC: [float("nan")],
                            constants.NUM_TOKENS_PER_BATCH_METRIC: [float("nan")],
                            PARAM_KEYS["batch_size"]: [batch_size],
                        })
                    ]
                )
            else:
                history_df.at[index, PARAM_KEYS["batch_size"]] = batch_size

        history_df[PARAM_KEYS["model_size"]] = params.model_size
        history_df[PARAM_KEYS["model_size"]] = pd.Categorical(
            history_df[PARAM_KEYS["model_size"]],
            categories=["NANO", "TINY", "BASE", "LARGE"],
            ordered=True
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
        history_df["tokens_per_instance"] = (
                history_df[constants.NUM_TOKENS_PER_BATCH_METRIC] / history_df[PARAM_KEYS["batch_size"]]
        )

        history_df = history_df[
            [
                PARAM_KEYS["gpu_type"],
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
                PARAM_KEYS["use_s1"],
                PARAM_KEYS["use_s2"],
                PARAM_KEYS["use_landsat"],
                "run_url",
            ]
        ]
        all_history.append(history_df)

    all_history_as_pd = pd.concat(all_history, axis=0)
    all_history_as_pd = all_history_as_pd.sort_values(by=[
        PARAM_KEYS["gpu_type"],
        PARAM_KEYS["model_size"],
        PARAM_KEYS["bf16"],
        PARAM_KEYS["image_size"],
        PARAM_KEYS["patch_size"],
        PARAM_KEYS["num_timesteps"],
        PARAM_KEYS["batch_size"],
    ])

    all_history_as_pd.to_csv("inference_throughput.csv", index=False)


if __name__ == "__main__":
    main()

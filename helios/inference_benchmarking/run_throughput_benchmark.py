import json
import os
import time
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
import wandb

from olmo_core.config import Config
from olmo_core.distributed.checkpoint import load_model_and_optim_state

from helios.nn.flexihelios import TokensAndMasks
from helios.train.masking import MaskedHeliosSample, MaskValue
from helios.inference_benchmarking import constants
from helios.inference_benchmarking.data_models import RunParams


NUM_S1_BANDS = 4
NUM_S2_BANDS = 12
NUM_LANDSAT_BANDS = 11


class Helios(torch.nn.Module):
    def __init__(self, checkpoint_path: str):
        super().__init__()

        # Load the model config and initialize it.
        # We avoid loading the train module here because it depends on running within
        # olmo_core.
        with open(f"{checkpoint_path}/config.json") as f:
            config_dict = json.load(f)
            model_config = Config.from_dict(config_dict["model"])

        # We only want the encoder, as the rest of the network will throw off
        # memory and latency estimates
        model = model_config.build()

        train_module_dir = f"{checkpoint_path}/model_and_optim"
        load_model_and_optim_state(train_module_dir, model)
        model = getattr(model, "encoder")

        self.model = model

    def forward(self, x: MaskedHeliosSample, patch_size: int):
        return self.model.forward(x, patch_size=patch_size)


def run_benchmarking(
    model: Helios,
    metrics: Any,
    run_params: RunParams
):
    device = next(model.parameters()).device

    if run_params.bf16:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    for idx, batch_size in enumerate(run_params.batch_sizes):
        if run_params.use_s1:
            # dims: (B, H, W, T, len(S1_BANDS)]
            s1_tensor = torch.rand(
                batch_size,
                run_params.image_size,
                run_params.image_size,
                run_params.num_timesteps,
                NUM_S1_BANDS,
                device=device,
                dtype=dtype,
            )
        else:
            s1_tensor = None

        if run_params.use_s2:
            # dims: (B, H, W, T, len(S2_BANDS)]
            s2_tensor = torch.rand(
                batch_size,
                run_params.image_size,
                run_params.image_size,
                run_params.num_timesteps,
                NUM_S2_BANDS,
                device=device,
                dtype=dtype,
            )
        else:
            s2_tensor = None

        if run_params.use_landsat:
            # dims: (B, H, W, T, len(LANDSAT_bands))
            landsat_tensor = torch.rand(
                batch_size,
                run_params.image_size,
                run_params.image_size,
                run_params.num_timesteps,
                NUM_LANDSAT_BANDS,
                device=device,
                dtype=dtype,
            )
        else:
            landsat_tensor = None

        latlon = torch.rand(batch_size, 2, device=device, dtype=dtype)  # dims: (B, 2)
        timestamps = torch.ones(
            batch_size,
            run_params.num_timesteps,
            3,
            dtype=torch.int32,
            device=device
        )  # dims: (B, T, D=3)

        def maybe_make_mask(maybe_t):
            if maybe_t is not None:
                return (torch.ones(maybe_t.shape, dtype=dtype, device=device,)
                        * MaskValue.ONLINE_ENCODER.value)

        masked_sample = MaskedHeliosSample(
            timestamps=timestamps,
            sentinel2_l2a=s2_tensor,
            sentinel2_l2a_mask=maybe_make_mask(s2_tensor),
            sentinel1=s1_tensor,
            sentinel1_mask=maybe_make_mask(s1_tensor),
            landsat=landsat_tensor,
            landsat_mask=maybe_make_mask(landsat_tensor),
            latlon=latlon,
            latlon_mask=maybe_make_mask(latlon),
        )

        interval_start_time = time.monotonic()
        tokens_processed_per_batch = []
        time_taken_per_batch = []

        while (time.monotonic() - interval_start_time) < run_params.benchmark_interval_s or len(tokens_processed_per_batch) < run_params.min_batches_per_interval:
            batch_start = time.monotonic()

            if run_params.bf16:
                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                    results: TokensAndMasks = model.forward(
                        masked_sample,
                        patch_size=run_params.patch_size
                    )
            else:
                results: TokensAndMasks = model.forward(
                    masked_sample,
                    patch_size=run_params.patch_size
                )

            num_s1_tokens = calculate_num_token_embeddings(results.sentinel1)
            num_s2_tokens = calculate_num_token_embeddings(results.sentinel2_l2a)
            num_landsat_tokens = calculate_num_token_embeddings(results.landsat)
            tokens_processed_per_batch.append(num_s1_tokens + num_s2_tokens + num_landsat_tokens)
            time_taken_per_batch.append(time.monotonic() - batch_start)

        metrics_to_submit = {
            constants.PER_BATCH_TOKEN_RATE_METRIC: wandb.Histogram(np.array(
                [
                    tokens_processed_per_batch,
                    time_taken_per_batch,
                ]
            )),
            constants.MEAN_BATCH_TOKEN_RATE_METRIC: sum(tokens_processed_per_batch) / sum(time_taken_per_batch),
            constants.MEAN_BATCH_TIME_METRIC: sum(time_taken_per_batch) / len(time_taken_per_batch),
            constants.NUM_TOKENS_PER_BATCH_METRIC: sum(tokens_processed_per_batch) / len(tokens_processed_per_batch)
        }

        print(f"Metrics for {batch_size} were: {metrics_to_submit}")
        metrics.log(metrics_to_submit, step=idx)


def calculate_num_token_embeddings(t: Optional[torch.Tensor]) -> int:
    if t is not None:
        batch_size, p_height, p_width, timestamps, bandsets, _ = tuple(t.shape)
        return batch_size * p_height * p_width * timestamps * bandsets

    return 0


if __name__ == "__main__":
    checkpoint_path = os.getenv(constants.PARAM_KEYS["checkpoint_path"], "/artifacts")
    run_params = RunParams.from_env_vars()
    project = os.getenv(constants.PARAM_KEYS["project"], constants.PROJECT_NAME)
    owner = os.getenv(constants.PARAM_KEYS["owner"], constants.ENTITY_NAME)
    name = os.getenv(constants.PARAM_KEYS["name"], "test")

    print("Initializing wandb...")
    wandb_dir = "/wandb"
    os.mkdir(wandb_dir)
    metrics = wandb.init(
        dir=wandb_dir,
        project=project,
        entity=owner,
        name=name,
    )

    try:
        print("Loading model...")
        model = Helios(checkpoint_path)
        print("helios loaded and on gpu")
        if torch.cuda.is_available():
            model.to("cuda:0")
        run_benchmarking(
            model,
            metrics,
            run_params
        )

    except Exception as e:
        wandb.finish(exit_code=1, quiet=True)
        raise e

    else:
        wandb.finish(exit_code=0, quiet=True)

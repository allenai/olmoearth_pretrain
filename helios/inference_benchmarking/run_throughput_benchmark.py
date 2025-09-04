"""Script for performing an inference throughput benchmarking run."""

import json
import os
import time
from typing import Any
import logging

import numpy as np
from logging import getLogger
import torch
import wandb
from olmo_core.config import Config
from olmo_core.distributed.checkpoint import load_model_and_optim_state

from helios.data.constants import BASE_GSD, Modality, BASE_RESOLUTION
from helios.inference_benchmarking import constants
from helios.inference_benchmarking.data_models import RunParams
from helios.nn.flexihelios import TokensAndMasks, Encoder
from helios.train.masking import MaskedHeliosSample, MaskValue

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

NUM_S1_BANDS = Modality.SENTINEL1.num_bands
NUM_S2_BANDS = Modality.SENTINEL2.num_bands
NUM_LANDSAT_BANDS = Modality.LANDSAT.num_bands

NUM_SQUARE_KM_LAND_IN_WORLD = 149_000_000

logger = getLogger(__name__)
class Helios(torch.nn.Module):
    """Thin wrapper around Helios checkpoint that loads just the encoder."""

    def __init__(self, checkpoint_path: str):
        """Loads the checkpoint, keeps only the encoder."""
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


        self.model: Encoder = model
        self.model.eval()
        self.model.apply_compile()

    def forward(self, x: MaskedHeliosSample, patch_size: int) -> TokensAndMasks:
        """Pass-through."""
        return self.model.forward(x, patch_size=patch_size, always_pass_none_mask_to_transformer=True)["tokens_and_masks"]


def run_benchmarking(model: Helios, metrics: Any, run_params: RunParams) -> None:
    """Runs the benchmarking code.

    Requires an instance of the Helios wrapper, a wandb metrics instance, and run params.
    """
    device = next(model.parameters()).device

    if run_params.bf16:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # track squarekm per second for every batch size and report the highest batch size
    squarekm_per_second_per_batch_size = {}
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
            batch_size, run_params.num_timesteps, 3, dtype=torch.int32, device=device
        )  # dims: (B, T, D=3)

        def maybe_make_mask(maybe_t: torch.Tensor | None) -> torch.Tensor | None:
            if maybe_t is not None:
                return (
                    torch.ones(
                        maybe_t.shape,
                        dtype=dtype,
                        device=device,
                    )
                    * MaskValue.ONLINE_ENCODER.value
                )
            return None
        # log the s2 tensor shape
        logger.info(f"S2 tensor shape: {s2_tensor.shape}")
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

        tokens_processed_per_batch: list[int] = []
        time_taken_per_batch: list[float] = []
        # log that the data is prepared
        logger.info("Data prepared, starting benchmark")
        # Run 5 forward passes as warmup
        for _ in range(5):
            with torch.inference_mode():
                if run_params.bf16:
                    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                        results = model.forward(
                            masked_sample, patch_size=run_params.patch_size
                        )
                else:
                    results = model.forward(masked_sample, patch_size=run_params.patch_size)

        num_forward_passes = 0
        if device.type == "cuda":
            torch.cuda.synchronize()
        overall_start_time = time.monotonic()
        interval_start_time = time.monotonic()
        while (
            time.monotonic() - interval_start_time
        ) < run_params.benchmark_interval_s or len(
            tokens_processed_per_batch
        ) < run_params.min_batches_per_interval:
            batch_start = time.monotonic()

            results: TokensAndMasks
            with torch.inference_mode():
                if run_params.bf16:
                    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                        results = model.forward(
                            masked_sample, patch_size=run_params.patch_size
                        )
                else:
                    results = model.forward(masked_sample, patch_size=run_params.patch_size)
            num_forward_passes += 1
            num_s1_tokens = calculate_num_token_embeddings(results.sentinel1)
            num_s2_tokens = calculate_num_token_embeddings(results.sentinel2_l2a)
            num_landsat_tokens = calculate_num_token_embeddings(results.landsat)
            tokens_processed_per_batch.append(
                num_s1_tokens + num_s2_tokens + num_landsat_tokens
            )
            time_taken_per_batch.append(time.monotonic() - batch_start)
        if device.type == "cuda":
            torch.cuda.synchronize()
        overall_time_taken = time.monotonic() - overall_start_time
        logger.info(f"Overall time taken: {overall_time_taken} sum of time taken per batch: {sum(time_taken_per_batch)}")
        metrics_to_submit = {
            constants.PER_BATCH_TOKEN_RATE_METRIC: wandb.Histogram(
                np.array(
                    [
                        tokens_processed_per_batch,
                        time_taken_per_batch,
                    ]
                )
            ),
            constants.MEAN_BATCH_TOKEN_RATE_METRIC: sum(tokens_processed_per_batch)
            / sum(time_taken_per_batch),
            constants.MEAN_BATCH_TIME_METRIC: sum(time_taken_per_batch)
            / len(time_taken_per_batch),
            constants.NUM_TOKENS_PER_BATCH_METRIC: sum(tokens_processed_per_batch)
            / len(tokens_processed_per_batch),
        }
        num_batches = len(time_taken_per_batch)  # or use num_forward_passes
        tile_km2 = (run_params.image_size * BASE_GSD / 1000.0) ** 2  # m -> km, then square
        area_processed_km2 = batch_size * tile_km2 * num_batches
        square_km_per_second = area_processed_km2 / sum(time_taken_per_batch)
        squarekm_per_second_per_batch_size[batch_size] = square_km_per_second
        metrics_to_submit[constants.SQUARE_KM_PER_SECOND_METRIC] = square_km_per_second
        metrics_to_submit[constants.HRS_TO_PROCESS_ALL_LAND_METRIC] = (
            NUM_SQUARE_KM_LAND_IN_WORLD / square_km_per_second / 3600.0
        )
        metrics_to_submit[constants.OVERALL_TIME_TAKEN_METRIC] = overall_time_taken
        # For N timesteps that number is what we care about
        print(f"Metrics for {batch_size} were: {metrics_to_submit}")
        metrics.log(metrics_to_submit, step=idx)
    logger.info(f"Square km per second per batch size: {squarekm_per_second_per_batch_size}")
    # which batch size has the highest square km per second
    highest_batch_size = max(squarekm_per_second_per_batch_size, key=squarekm_per_second_per_batch_size.get)
    logger.info(f"Highest batch size: {highest_batch_size}")
    logger.info(f"Highest square km per second: {squarekm_per_second_per_batch_size[highest_batch_size]}")
class Metrics:
    """Simple metrics logger that stores logs per step, but does not submit to wandb."""

    def __init__(self):
        self.logged = []

    def log(self, metrics: dict, step: int = None):
        entry = {"step": step, "metrics": metrics}
        self.logged.append(entry)

def calculate_num_token_embeddings(t: torch.Tensor | None) -> int:
    """Determines how many tokens are represented in the given tensor."""
    if t is not None:
        batch_size, p_height, p_width, timestamps, bandsets, _ = tuple(t.shape)
        return batch_size * p_height * p_width * timestamps * bandsets

    return 0


# Make this whole thing omega config based
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    os.environ[constants.PARAM_KEYS["checkpoint_path"]] = constants.BASE_PATH
    # set benchmark interval to 15 seconds
    os.environ[constants.PARAM_KEYS["benchmark_interval_s"]] = "15"
    # use S2 set to true
    os.environ[constants.PARAM_KEYS["use_s2"]] = "1"
    os.environ[constants.PARAM_KEYS["patch_size"]] = "2"
    os.environ[constants.PARAM_KEYS["image_size"]] = "4"
    os.environ[constants.PARAM_KEYS["num_timesteps"]] = "12"
    os.environ[constants.PARAM_KEYS["batch_sizes"]] = "8,64,128,256,512,1024,2048,4096,8192"
    # make sure bfloat16 is on
    os.environ[constants.PARAM_KEYS["bf16"]] = "1"
    checkpoint_path = os.getenv(constants.PARAM_KEYS["checkpoint_path"], "/artifacts")
    run_params = RunParams.from_env_vars()
    project = os.getenv(constants.PARAM_KEYS["project"], constants.PROJECT_NAME)
    owner = os.getenv(constants.PARAM_KEYS["owner"], constants.ENTITY_NAME)
    name = os.getenv(constants.PARAM_KEYS["name"], "test")

    # print("Initializing wandb...")
    # wandb_dir = "/wandb"
    # os.makedirs(wandb_dir, exist_ok=True)
    # metrics = wandb.init(
    #     dir=wandb_dir,
    #     project=project,
    #     entity=owner,
    #     name=name,
    # )
    metrics = Metrics()
    try:
        logger.info("Loading model...")
        model = Helios(checkpoint_path)
        if torch.cuda.is_available():
            model.to("cuda:0")
            logger.info("helios loaded and on gpu")
        run_benchmarking(model, metrics, run_params)

    except Exception as e:
        import traceback
        traceback.print_exc()
        wandb.finish(exit_code=1, quiet=True)
        raise e

    else:
        wandb.finish(exit_code=0, quiet=True)

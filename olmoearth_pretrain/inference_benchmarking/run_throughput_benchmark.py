"""Script for performing an inference throughput benchmarking run."""

import itertools
import os
import time
import uuid
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
from olmo_core.config import Config
from olmo_core.io import copy_file, file_exists, join_path
from olmo_core.train.callbacks import ProfilerCallback, WandBCallback
from olmo_core.train.trainer import PathOrStr

from olmoearth_pretrain._compat import deprecated_class_alias as _deprecated_class_alias
from olmoearth_pretrain.data.constants import BASE_GSD, Modality
from olmoearth_pretrain.inference_benchmarking import constants
from olmoearth_pretrain.inference_benchmarking.data_models import RunParams
from olmoearth_pretrain.internal.utils import MODEL_SIZE_ARGS
from olmoearth_pretrain.nn.flexi_vit import (
    Encoder,
    EncoderConfig,
    PredictorConfig,
    TokensAndMasks,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample, MaskValue

NUM_S1_BANDS = Modality.SENTINEL1.num_bands
NUM_S2_BANDS = Modality.SENTINEL2.num_bands
NUM_LANDSAT_BANDS = Modality.LANDSAT.num_bands

NUM_SQUARE_KM_LAND_IN_WORLD = 149_000_000

logger = getLogger(__name__)


class MinimalTrainer:
    """Minimal trainer that only has the persist_working_file method so we can use the callbacks without the full trainer."""

    def __init__(
        self, device: torch.device, work_dir: Path, save_folder: Path | None = None
    ):
        """Initializes the minimal trainer."""
        self.device = device
        self.work_dir = work_dir  # Will be set later
        if save_folder is None:
            self.save_folder = work_dir
        else:
            self.save_folder = save_folder

    def persist_working_file(self, name: PathOrStr) -> PathOrStr:
        """Persists a working file."""
        if Path(name).is_relative_to(self.work_dir):
            name = Path(name).relative_to(self.work_dir)
        source = join_path(self.work_dir, name)
        target = join_path(self.save_folder, name)
        if source != target:
            copy_file(source, target)
        elif not file_exists(source):
            raise FileNotFoundError(source)
        return target


class OlmoEarth(torch.nn.Module):
    """Thin wrapper around OlmoEarth Pretrain checkpoint that loads just the encoder."""

    def __init__(self, model_config: Config) -> None:
        """Loads the checkpoint, keeps only the encoder."""
        super().__init__()

        # We only want the encoder, as the rest of the network will throw off
        # memory and latency estimates
        model = model_config.build()

        model = getattr(model, "encoder")

        self.model: Encoder = model
        self.model.eval()
        # probably want to add a flag for this
        self.model.apply_compile()

    def forward(
        self,
        x: MaskedOlmoEarthSample,
        patch_size: int,
        fast_pass: bool = True,
    ) -> TokensAndMasks:
        """Pass-through."""
        return self.model.forward(
            x,
            patch_size=patch_size,
            fast_pass=fast_pass,
        )["tokens_and_masks"]


Helios = _deprecated_class_alias(
    OlmoEarth, "helios.inference_benchmarking.run_throughput_benchmark.Helios"
)


@dataclass
class ThroughputBenchmarkRunnerConfig(Config):
    """Defines the configuration for a throughput benchmarking run."""

    sweep_dict: dict[str, Any] | None = None  # dict of run params to sweep over
    sweep_keys: list[str] | None = None
    sweep_group_name: str | None = None
    training_modalities: list[str] = field(
        default_factory=lambda: [
            Modality.SENTINEL2_L2A.name,
            Modality.SENTINEL1.name,
            Modality.LANDSAT.name,
        ]
    )
    work_dir: Path = Path("./test_work_dir")
    default_run_params: RunParams | None = None
    save_folder: Path | None = None
    cross_product_sweep: bool = False

    def build(self) -> "ThroughputBenchmarkRunner":
        """Builds a throughput benchmarking runner."""
        if self.default_run_params is None:
            self.default_run_params = RunParams()

        if self.sweep_dict is None and self.sweep_keys is None:
            raise ValueError("Either sweep_dict or sweep_keys must be set")
        if self.sweep_dict is not None and self.sweep_keys is not None:
            raise ValueError("Only one of sweep_dict or sweep_keys can be set")
        if self.sweep_dict is None and self.sweep_keys is not None:
            sweep_dict: dict[str, Any] = {}
            for sweep_key in self.sweep_keys:
                sweep_dict[sweep_key] = constants.SWEEPS[sweep_key]
            sweep_dict = sweep_dict

        return ThroughputBenchmarkRunner(
            default_run_params=self.default_run_params,
            sweep_group_name=self.sweep_group_name,
            training_modalities=self.training_modalities,
            work_dir=self.work_dir,
            save_folder=self.save_folder,
            sweep_dict=sweep_dict,
            cross_product_sweep=self.cross_product_sweep,
        )


def calculate_num_token_embeddings(t: torch.Tensor | None) -> int:
    """Determines how many tokens are represented in the given tensor."""
    if t is not None:
        batch_size, p_height, p_width, timestamps, bandsets, _ = tuple(t.shape)
        return batch_size * p_height * p_width * timestamps * bandsets

    return 0


class ThroughputBenchmarkRunner:
    """Runner for a throughput benchmarking run."""

    def __init__(
        self,
        default_run_params: RunParams,
        sweep_group_name: str | None,
        training_modalities: list[str],
        work_dir: Path,
        save_folder: Path | None = None,
        sweep_dict: dict[str, Any] = {},
        cross_product_sweep: bool = False,
    ):
        """Initializes the throughput benchmarking runner."""
        self.default_run_params = default_run_params
        self.sweep_group_name = sweep_group_name
        self.training_modalities = training_modalities
        self.work_dir = work_dir
        self.work_dir.mkdir(exist_ok=True)
        self.save_folder = save_folder
        self.sweep_dict = sweep_dict
        self.cross_product_sweep = cross_product_sweep
        uuid_str = str(uuid.uuid4())[:6]
        self.sweep_name = "_".join(self.sweep_dict.keys()) + "-" + uuid_str

    def build_model(self, run_params: RunParams) -> OlmoEarth:
        """Builds a model based on the run parameters."""
        model_size = MODEL_SIZE_ARGS[run_params.model_size]
        training_modalities = self.training_modalities
        encoder_config = EncoderConfig(
            embedding_size=int(model_size["encoder_embedding_size"]),
            num_heads=int(model_size["encoder_num_heads"]),
            depth=int(model_size["encoder_depth"]),
            mlp_ratio=float(model_size["mlp_ratio"]),
            supported_modality_names=training_modalities,
        )
        decoder_config = PredictorConfig(
            encoder_embedding_size=int(model_size["encoder_embedding_size"]),
            decoder_embedding_size=int(model_size["decoder_embedding_size"]),
            depth=int(model_size["decoder_depth"]),
            mlp_ratio=float(model_size["mlp_ratio"]),
            num_heads=int(model_size["decoder_num_heads"]),
            supported_modality_names=training_modalities,
            max_sequence_length=12,
        )
        model_config = LatentMIMConfig(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
        )
        model = OlmoEarth(model_config=model_config)
        return model

    def build_sweep_run_params(self) -> list[RunParams]:
        """Builds a list of run parameters based on the sweep dictionary."""
        run_params_list: list[RunParams] = []
        if self.cross_product_sweep:
            # take a cross product of the sweep dictionary
            sweep_dict_keys = list(self.sweep_dict.keys())
            # for every different combination of the sweep dictionary, build a run params
            for combination in itertools.product(
                *[self.sweep_dict[key] for key in sweep_dict_keys]
            ):
                run_params_list.append(
                    self.default_run_params.replace(
                        **dict(zip(sweep_dict_keys, combination))
                    )
                )
        else:
            for key, value in self.sweep_dict.items():
                for v in value:
                    # Merge the sweep parameter with default_run_params
                    run_params_list.append(self.default_run_params.replace(**{key: v}))
        # Add the default run params
        run_params_list.append(self.default_run_params)
        return run_params_list

    def run_benchmarking_sweep(self, run_params_list: list[RunParams]) -> None:
        """Runs the benchmarking code for a list of run parameters."""
        for run_params in run_params_list:
            try:
                logger.info(f"Running benchmarking for {run_params}")
                self.run_benchmarking(run_params)
            except Exception as e:
                logger.error(f"Error running benchmarking for {run_params}: {e}")
                continue

    def run_benchmarking(self, run_params: RunParams) -> None:
        """Runs the benchmarking code.

        Requires an instance of the OlmoEarth Pretrain wrapper, a wandb metrics instance, and run params.
        """
        model = self.build_model(run_params)
        if torch.cuda.is_available() and run_params.gpu_type == "cuda":
            logger.info("helios loaded and on gpu")
            model.to(run_params.gpu_type)
        device = next(model.parameters()).device
        batch_size = run_params.batch_size
        idx = 0

        if run_params.bf16:
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        callbacks = []
        # insantiate callbacks
        if run_params.profiler_enabled:
            profiler = ProfilerCallback(
                skip_first=0,  # Don't skip any steps
                wait=0,  # Start profiling immediately
                warmup=5,  # Warmup for 5 steps (matches your current warmup)
                active=5,  # Profile for 5 steps
                repeat=1,  # Only one cycle
            )

            profiler.trainer = MinimalTrainer(device, self.work_dir)

            callbacks.append(profiler)

        if run_params.wandb_enabled:
            project = os.getenv(constants.PARAM_KEYS["project"], constants.PROJECT_NAME)
            owner = os.getenv(constants.PARAM_KEYS["owner"], constants.ENTITY_NAME)
            name = run_params.run_name
            name = f"{run_params.run_name}-{self.sweep_name}"
            group = self.sweep_group_name
            wandb_callback = WandBCallback(
                project=project,
                entity=owner,
                name=name,
                group=group,
                config=run_params.as_dict(),
            )
            wandb_callback.trainer = MinimalTrainer(device, self.work_dir)
            callbacks.append(wandb_callback)

        for callback in callbacks:
            callback.pre_train()

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

        masked_sample = MaskedOlmoEarthSample(
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
        logger.info("Data prepared, starting warmup")
        torch.cuda.set_sync_debug_mode("warn")
        # Run 5 forward passes as warmup
        oom_occurred = False
        for _ in range(5):
            try:
                with torch.inference_mode():
                    if run_params.bf16:
                        with torch.amp.autocast(
                            device_type=device.type, dtype=torch.bfloat16
                        ):
                            results = model.forward(
                                masked_sample, patch_size=run_params.patch_size
                            )
                    else:
                        results = model.forward(
                            masked_sample, patch_size=run_params.patch_size
                        )
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"CUDA OOM during warmup: {e}")
                oom_occurred = True
                break

        if oom_occurred:
            logger.info("CUDA OOM occurred during warmup, skipping benchmark")
            # Log OOM status to wandb
            metrics_oom_occurred: dict[str, Any] = {
                constants.OOM_OCCURRED_METRIC: 1,
            }
            for callback in callbacks:
                callback.log_metrics(step=0, metrics=metrics_oom_occurred)
            for callback in callbacks:
                callback.post_train()
            return

        logger.info("Warmup complete, starting benchmark")
        # TODO: Do cuda event timing
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

            with torch.inference_mode():
                if run_params.bf16:
                    with torch.amp.autocast(
                        device_type=device.type, dtype=torch.bfloat16
                    ):
                        results = model.forward(
                            masked_sample,
                            patch_size=run_params.patch_size,
                        )
                else:
                    results = model.forward(
                        masked_sample, patch_size=run_params.patch_size
                    )
            # seperately time batches outside the larger loop
            time_taken_per_batch.append(time.monotonic() - batch_start)

            # Call profiler step for each forward pass
            for callback in callbacks:
                callback.pre_load_batch()

            num_s1_tokens = calculate_num_token_embeddings(results.sentinel1)
            num_s2_tokens = calculate_num_token_embeddings(results.sentinel2_l2a)
            num_landsat_tokens = calculate_num_token_embeddings(results.landsat)
            tokens_processed_per_batch.append(
                num_s1_tokens + num_s2_tokens + num_landsat_tokens
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        overall_time_taken = time.monotonic() - overall_start_time
        logger.info(
            f"Overall time taken: {overall_time_taken} sum of time taken per batch: {sum(time_taken_per_batch)} num batches: {len(time_taken_per_batch)}"
        )
        metrics_to_submit: dict[str, Any] = {
            constants.PER_BATCH_TOKEN_RATE_METRIC: wandb.Histogram(
                np.array(
                    [
                        tokens_processed_per_batch,
                        time_taken_per_batch,
                    ]
                )
            ),
            constants.MEAN_BATCH_TOKEN_RATE_METRIC: sum(tokens_processed_per_batch)
            / overall_time_taken,
            constants.MEAN_BATCH_TIME_METRIC: overall_time_taken
            / len(time_taken_per_batch),
            constants.NUM_TOKENS_PER_BATCH_METRIC: sum(tokens_processed_per_batch)
            / len(tokens_processed_per_batch),
        }
        num_batches = len(time_taken_per_batch)
        num_centroids = num_batches * batch_size
        centroids_per_second = num_centroids / overall_time_taken
        tile_km2 = (
            run_params.image_size * BASE_GSD / 1000.0
        ) ** 2  # m -> km, then square
        area_processed_km2 = batch_size * tile_km2 * num_batches
        square_km_per_second = area_processed_km2 / overall_time_taken
        metrics_to_submit[constants.SQUARE_KM_PER_SECOND_METRIC] = square_km_per_second
        metrics_to_submit[constants.PIXELS_PER_SECOND_METRIC] = centroids_per_second
        try:
            gpu_name = torch.cuda.get_device_name(device)
            metrics_to_submit[constants.GPU_NAME_METRIC] = gpu_name
        except Exception as e:
            logger.error(f"Error getting GPU name: {e}")

        logger.info(f"Metrics for {batch_size} were: {metrics_to_submit}")
        # TODO: If different configurations are different runs how can we do them back to back?
        for callback in callbacks:
            callback.log_metrics(step=idx, metrics=metrics_to_submit)
        for callback in callbacks:
            callback.post_train()

    def run(self) -> None:
        """Runs the throughput benchmarking."""
        run_params_list = self.build_sweep_run_params()
        logger.info(
            f"Running {len(run_params_list)} benchmarking runs sweeping over {self.sweep_dict}"
        )
        self.run_benchmarking_sweep(run_params_list)

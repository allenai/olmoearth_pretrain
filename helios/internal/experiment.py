"""Code for configuring and running Helios experiments."""

import logging
import sys
from dataclasses import dataclass
from os import environ
from typing import Callable, Dict, List, Optional, cast

import numpy as np
import torch
from olmo_core.config import Config, StrEnum
from olmo_core.data import (DataMix, NumpyDataLoaderConfig, NumpyDatasetConfig,
                            NumpyDatasetType, TokenizerConfig,
                            VSLCurriculumConfig, VSLCurriculumType)
from olmo_core.distributed.parallel.data_parallel import (DataParallelConfig,
                                                          DataParallelType)
from olmo_core.distributed.utils import (get_fs_local_rank, get_local_rank,
                                         get_rank, get_world_size)
from olmo_core.internal.common import (build_launch_config,
                                       get_beaker_username, get_root_dir,
                                       get_work_dir)
from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim.adamw import AdamWConfig
from olmo_core.optim.scheduler import ConstantWithWarmup
from olmo_core.train import (TrainerConfig, prepare_training_environment,
                             teardown_training_environment)
from olmo_core.train.callbacks import (Callback, CometCallback,
                                       ConfigSaverCallback,
                                       DownstreamEvaluatorCallbackConfig,
                                       GarbageCollectorCallback,
                                       GPUMemoryMonitorCallback,
                                       LMEvaluatorCallbackConfig,
                                       ProfilerCallback, SlackNotifierCallback,
                                       WandBCallback)
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.common import Duration, LoadStrategy
from olmo_core.train.train_module import TransformerTrainModuleConfig
from olmo_core.utils import prepare_cli_environment, seed_all
from rich import print
from upath import UPath

from helios.data.constants import Modality
from helios.data.dataloader import HeliosDataLoaderConfig
from helios.data.dataset import HeliosDatasetConfig, collate_helios
from helios.nn.flexihelios import EncoderConfig, PredictorConfig
from helios.nn.latent_mim import LatentMIMConfig
from helios.train.callbacks.speed_monitor import HeliosSpeedMonitorCallback
from helios.train.loss import LossConfig
from helios.train.masking import MaskingConfig
from helios.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)
# Variables to be changed per user
workdir = UPath("/temp/helios/workdir")  # nosec
# This allows pre-emptible jobs to save their workdir in the output folder
if environ.get("USE_OUTPUT_FOLDER"):
    workdir = UPath(environ["USE_OUTPUT_FOLDER"]) / "helios" / "workdir"

WANDB_USERNAME = "eai-ai2"  # nosec
WANDB_PROJECT = "helios-debug"
# PLEASE CHANGE IF THIS IS A NEW EXPERIMENT
run_name = "helios-test-new"
# PER EXPERIMENT Variables
LR = 0.0001
GLOBAL_BATCH_SIZE = 32
RANK_BATCH_SIZE = 32
MAX_DURATION = Duration.epochs(50)
NUM_WORKERS = 16
NUM_THREADS = 0
METRICS_COLLECT_INTERVAL = 1
CANCEL_CHECK_INTERVAL = 1
SAVE_FOLDER = workdir / "save_folder"
LOAD_STRATEGY = LoadStrategy.if_available

TILE_PATH = UPath("/weka/dfive-default/helios/dataset/20250212/")
DTYPE = np.dtype("float32")
SUPPORTED_MODALITIES = [
    Modality.SENTINEL2,
    Modality.LATLON,
    Modality.SENTINEL1,
    Modality.WORLDCOVER,
]
MAX_PATCH_SIZE = 8  # NOTE: actual patch_size <= max_patch_size
ENCODE_RATIO = 0.5
DECODE_RATIO = 0.5
TOKEN_BUDGET = 1500
H_W_TO_SAMPLE_MIN = 2
H_W_TO_SAMPLE_MAX = 13
WARMUP_STEPS = 2
ENCODER_EMBEDDING_SIZE = 256
DECODER_EMBEDDING_SIZE = 256
ENCODER_DEPTH = 4
DECODER_DEPTH = 4
ENCODER_NUM_HEADS = 8
DECODER_NUM_HEADS = 8
MLP_RATIO = 4.0
# First layout all the configs


@dataclass
class CommonComponents(Config):
    """Any configurable items that are common to all experiments."""

    run_name: str
    save_folder: str
    launch: BeakerLaunchConfig
    dataset: HeliosDatasetConfig
    data_loader: HeliosDataLoaderConfig
    callbacks: dict[str, Callback]


# I want a model config that is more agnostic to the specific architecture
def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model config for an experiment."""
    logger.info("Building model config")
    logger.info(f"Common components: {common} not set up yet")
    encoder_config = EncoderConfig(
        supported_modalities=SUPPORTED_MODALITIES,
        embedding_size=ENCODER_EMBEDDING_SIZE,
        max_patch_size=MAX_PATCH_SIZE,
        num_heads=ENCODER_NUM_HEADS,
        depth=ENCODER_DEPTH,
        mlp_ratio=MLP_RATIO,
        drop_path=0.1,
        max_sequence_length=12,
        use_channel_embs=True,
    )
    decoder_config = PredictorConfig(
        encoder_embedding_size=ENCODER_EMBEDDING_SIZE,
        decoder_embedding_size=DECODER_EMBEDDING_SIZE,
        depth=DECODER_DEPTH,
        mlp_ratio=MLP_RATIO,
        num_heads=DECODER_NUM_HEADS,
        max_sequence_length=12,
        supported_modalities=SUPPORTED_MODALITIES,
        learnable_channel_embeddings=True,
    )
    model_config = LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        token_budget=TOKEN_BUDGET,
        h_w_to_sample_min=H_W_TO_SAMPLE_MIN,
        h_w_to_sample_max=H_W_TO_SAMPLE_MAX,
    )
    return model_config


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """Build the train module config for an experiment."""
    logger.info("Building train module config")
    logger.info(f"Common components: {common} not set up yet")
    optim_config = AdamWConfig(lr=LR)
    masking_config = MaskingConfig(
        strategy_config={
            "type": "random",
            "encode_ratio": ENCODE_RATIO,
            "decode_ratio": DECODE_RATIO,
        }
    )
    loss_config = LossConfig(
        loss_config={
            "type": "patch_discrimination",
        }
    )
    dp_config = DataParallelConfig(name=DataParallelType.ddp)
    scheduler = ConstantWithWarmup(warmup_steps=WARMUP_STEPS)
    train_module_config = LatentMIMTrainModuleConfig(
        optim=optim_config,
        masking_config=masking_config,
        loss_config=loss_config,
        rank_batch_size=RANK_BATCH_SIZE,
        max_grad_norm=1.0,
        dp_config=dp_config,
        scheduler=scheduler,
    )
    return train_module_config


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
checkpointer_config = CheckpointerConfig(work_dir=workdir)
dataset_config = HeliosDatasetConfig(
    tile_path=TILE_PATH,
    supported_modalities=SUPPORTED_MODALITIES,
    dtype=DTYPE,
)
# things should be set during building
dataloader_config = HeliosDataLoaderConfig(
    global_batch_size=GLOBAL_BATCH_SIZE,
    dp_world_size=get_world_size(dp_process_group),
    dp_rank=get_rank(dp_process_group),
    fs_local_rank=get_fs_local_rank(),
    work_dir=workdir,
    num_threads=NUM_THREADS,
    num_workers=NUM_WORKERS,
)
dataloader = dataloader_config.build(
    dataset=dataset,
    collator=collate_helios,
)
wandb_callback = WandBCallback(
    name=run_name,
    project=WANDB_PROJECT,
    entity=WANDB_USERNAME,
    enabled=True,  # set to False to avoid wandb errors
)

dp_config = DataParallelConfig(name=DataParallelType.ddp)
# Let us not use garbage collector fallback
trainer_config = (
    TrainerConfig(
        work_dir=workdir,
        load_strategy=LOAD_STRATEGY,
        save_folder=SAVE_FOLDER,
        cancel_check_interval=CANCEL_CHECK_INTERVAL,
        metrics_collect_interval=METRICS_COLLECT_INTERVAL,
        max_duration=MAX_DURATION,
        checkpointer=checkpointer_config,
    )
    .with_callback("wandb", wandb_callback)
    .with_callback("speed_monitor", HeliosSpeedMonitorCallback())
    .with_callback("gpu_memory_monitor", GPUMemoryMonitorCallback())
    # .with_callback("profiler", ProfilerCallback())
)


@dataclass
class HeliosExperimentConfig(Config):
    run_name: str
    # launch: BeakerLaunchConfig # we should use this as well
    model: LatentMIMConfig  # TODO: make this agnostic to training setup
    dataset: HeliosDatasetConfig  # will likely be fixed for us
    data_loader: HeliosDataLoaderConfig  # will likely be fixed for us
    train_module: LatentMIMTrainModuleConfig  # we will want to support different train module model combinations
    trainer: TrainerConfig
    init_seed: int = 12536


helios_experiment_config = HeliosExperimentConfig(
    run_name=run_name,
    # launch=launch_config,
    model=model_config,
    dataset=dataset_config,
    data_loader=dataloader_config,
)

# Build config


# def build_common_components(
#     run_name: str,
#     cluster: str,
#     overrides: List[str],
#     *,
#     global_batch_size: int,
# ) -> CommonComponents:
#     """Build the common components for an experiment."""


def build_config(
    run_name: str,
    model_config: Callable[[CommonComponents], LatentMIMConfig],
    dataset_config: Callable[[CommonComponents], HeliosDatasetConfig],
    dataloader_config: Callable[[CommonComponents], HeliosDataLoaderConfig],
) -> HeliosExperimentConfig:
    pass


# prep

# train

# a run method that does all this


# actually make this config
# set up the logic for each different experiment


# ask pete to make it more agnostic from language modeling
# Support logging all configs to WandB and to have an experiment config


# Extra features
# Parameter overrides
# debug modes

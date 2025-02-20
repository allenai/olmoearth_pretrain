"""Trying to prototype fitting everything into olmo core."""

import logging
import uuid
from os import environ

import numpy as np
from olmo_core.distributed.utils import get_fs_local_rank, get_rank, get_world_size
from olmo_core.optim import AdamWConfig
from olmo_core.optim.scheduler import ConstantWithWarmup
from olmo_core.train import prepare_training_environment, teardown_training_environment
from olmo_core.train.callbacks import (
    GPUMemoryMonitorCallback,
    WandBCallback,
)
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.common import Duration, LoadStrategy
from olmo_core.train.config import TrainerConfig
from olmo_core.utils import get_default_device
from upath import UPath

from helios.data.constants import Modality
from helios.data.dataloader import HeliosDataLoaderConfig
from helios.data.dataset import HeliosDatasetConfig, collate_helios
from helios.nn.flexihelios import EncoderConfig, PredictorConfig
from helios.nn.latent_mim import LatentMIMConfig
from helios.train.callbacks.evaluator_callback import DownstreamEvaluatorCallbackConfig
from helios.train.callbacks.speed_monitor import HeliosSpeedMonitorCallback
from helios.train.loss import LossConfig
from helios.train.masking import MaskingConfig
from helios.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # Variables to be changed per user
    workdir = UPath("/temp/helios/workdir")  # nosec
    # This allows pre-emptible jobs to save their workdir in the output folder
    if environ.get("USE_OUTPUT_FOLDER"):
        workdir = UPath(environ["USE_OUTPUT_FOLDER"]) / "helios" / "workdir"

    WANDB_USERNAME = "eai-ai2"  # nosec
    WANDB_PROJECT = "helios-debug"
    run_name = f"helios-test-no-worldcover-2M-{str(uuid.uuid4())[:8]}"
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

    # Downstream evaluation
    EVAL_INTERVAL = 20
    EVAL_TASKS = ["m-eurosat"]

    #################### Setup environment ####################
    dp_config = None
    # for distributed training use torchrun
    # Uncomment this line to use distributed training
    # dp_config = DataParallelConfig(name=DataParallelType.ddp)
    # for distributed training use torchrun
    if dp_config is not None:
        prepare_training_environment(seed=42)
    else:
        prepare_training_environment(seed=42, backend=None)
    logger.setLevel(logging.DEBUG)
    logger.info("Starting Helios training")

    #################### Configs for model ####################
    # TODO: build encoder_small, encoder_base, encoder_large, etc. Same for decoder
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
    model = model_config.build()

    device = get_default_device()
    logger.info(f"Using device: {device}")
    # Ideally though this should be handled by the Model COnfig and build
    model = model.to(device)

    #################### Configs for train module ####################
    checkpointer_config = CheckpointerConfig(work_dir=workdir)
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
    scheduler = ConstantWithWarmup(warmup_steps=WARMUP_STEPS)
    train_module_config = LatentMIMTrainModuleConfig(
        optim=optim_config,
        masking_config=masking_config,
        loss_config=loss_config,
        rank_batch_size=RANK_BATCH_SIZE,
        max_grad_norm=1.0,
        scheduler=scheduler,
    )
    train_module = train_module_config.build(model=model)
    dp_process_group = train_module.dp_process_group

    #################### Configs for dataloader ####################
    dataset_config = HeliosDatasetConfig(
        tile_path=TILE_PATH,
        supported_modalities=SUPPORTED_MODALITIES,
        dtype=DTYPE,
    )
    dataset = dataset_config.build()
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

    #################### Configs for trainer ####################
    wandb_callback = WandBCallback(
        name=run_name,
        project=WANDB_PROJECT,
        entity=WANDB_USERNAME,
        enabled=True,
    )
    # Let us not use garbage collector fallback
    trainer_config = (
        TrainerConfig(
            work_dir=workdir,
            load_strategy=LOAD_STRATEGY,
            device=device,
            save_folder=SAVE_FOLDER,
            cancel_check_interval=CANCEL_CHECK_INTERVAL,
            metrics_collect_interval=METRICS_COLLECT_INTERVAL,
            max_duration=MAX_DURATION,
            checkpointer=checkpointer_config,
        )
        .with_callback("wandb", wandb_callback)
        .with_callback("speed_monitor", HeliosSpeedMonitorCallback())
        .with_callback("gpu_memory_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=EVAL_TASKS,
                eval_interval=EVAL_INTERVAL,
            ),
        )
    )
    trainer = trainer_config.build(
        train_module=train_module,
        data_loader=dataloader,
    )
    trainer.fit()

    teardown_training_environment()

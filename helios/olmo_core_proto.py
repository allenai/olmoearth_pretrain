"""Trying to prototype fitting everything into olmo core."""

import logging
import uuid

import numpy as np
from olmo_core.distributed.parallel import DataParallelConfig, DataParallelType
from olmo_core.distributed.utils import get_fs_local_rank, get_rank, get_world_size
from olmo_core.optim import AdamWConfig
from olmo_core.train import prepare_training_environment, teardown_training_environment
from olmo_core.train.callbacks.wandb import WandBCallback
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.common import Duration, LoadStrategy
from olmo_core.train.config import TrainerConfig
from olmo_core.utils import get_default_device
from upath import UPath

from helios.data.collator import per_modality_collate_fn
from helios.data.dataloader import HeliosDataLoader
from helios.data.dataset import HeliosDataset
from helios.dataset.index import DatasetIndexParser
from helios.latent_predictor import LatentMIMStyle
from helios.train.callbacks.speed_monitor import HeliosSpeedMonitorCallback
from helios.train.decoder import SimpleLatentDecoder
from helios.train.encoder import PatchEncoder
from helios.train.loss import patch_disc_loss
from helios.train.train_module import HeliosTrainModuleConfig

logger = logging.getLogger(__name__)

# THings that need a config
# Data Loader
# Model

if __name__ == "__main__":
    # Variables to be changed per user
    workdir = UPath("/Users/henryh/Desktop/eai-repos/helios-repos/helios/workdir")
    WANDB_USERNAME = "henryhzog"
    WANDB_PROJECT = "helios-test"
    # PER EXPERIMENT Variables
    GLOBAL_BATCH_SIZE = 8
    RANK_BATCH_SIZE = 4
    MAX_DURATION = Duration.epochs(4)
    NUM_WORKERS = 0
    NUM_THREADS = 0
    METRICS_COLLECT_INTERVAL = 1
    CANCEL_CHECK_INTERVAL = 1
    SAVE_FOLDER = workdir / "save_folder"
    LOAD_STRATEGY = LoadStrategy.if_available

    dp_config = None
    # for distributed training use torchrun
    # Uncomment this line to use distributed training
    dp_config = DataParallelConfig(name=DataParallelType.ddp)

    # for distributed training use torchrun
    if dp_config is not None:
        prepare_training_environment(seed=42)
    else:
        prepare_training_environment(seed=42, backend=None)
    # set log level to debug
    logger.setLevel(logging.DEBUG)

    index_path = "/weka/dfive-default/helios_sample_data/20250115-sample-dataset-helios/index.csv"
    index_parser = DatasetIndexParser(index_path)
    samples = index_parser.samples

    # Variable masking is not used
    encoder = PatchEncoder(
        in_channels=13,
        embed_dim=64,
        patch_size=16,
        depth=1,
        num_heads=1,
        mlp_ratio=1.0,
    )
    decoder = SimpleLatentDecoder(
        embed_dim=64,
        mlp_ratio=1.0,
        dropout=0.1,
    )
    model = LatentMIMStyle(encoder, decoder)

    device = get_default_device()
    # Ideally though this should be handled by the Model COnfig and build
    model = model.to(device)
    checkpointer_config = CheckpointerConfig(work_dir=workdir)
    optim_config = AdamWConfig()

    train_module_config = HeliosTrainModuleConfig(
        optim=optim_config,
        rank_batch_size=RANK_BATCH_SIZE,
        loss_fn=patch_disc_loss,
    )
    train_module = train_module_config.build(model=model)
    dp_process_group = train_module.dp_process_group
    dataloader = HeliosDataLoader.wrap_numpy_dataset(
        dataset=HeliosDataset(
            *samples,
            ignore_data_sources=["openstreetmap"],
            filter_samples_with_missing_inputs=True,
            dtype=np.dtype("float32"),
        ),
        global_batch_size=GLOBAL_BATCH_SIZE,
        dp_world_size=get_world_size(dp_process_group),
        dp_rank=get_rank(dp_process_group),
        fs_local_rank=get_fs_local_rank(),
        collator=per_modality_collate_fn,
        work_dir=workdir,
        num_threads=NUM_THREADS,
        num_workers=NUM_WORKERS,
    )

    run_name = f"test-debug-{str(uuid.uuid4())[:8]}"
    wandb_callback = WandBCallback(
        name=run_name,
        project=WANDB_PROJECT,
        entity=WANDB_USERNAME,
    )
    callbacks = {
        "speed_monitor": HeliosSpeedMonitorCallback(),
        "wandb": wandb_callback,
    }
    trainer_config = TrainerConfig(
        work_dir=workdir,
        load_strategy=LOAD_STRATEGY,
        device=device,
        save_folder=SAVE_FOLDER,
        callbacks=callbacks,
        cancel_check_interval=CANCEL_CHECK_INTERVAL,
        metrics_collect_interval=METRICS_COLLECT_INTERVAL,
        max_duration=MAX_DURATION,
        checkpointer=checkpointer_config,
    )
    trainer = trainer_config.build(
        train_module=train_module,
        data_loader=dataloader,
    )
    trainer.fit()
    teardown_training_environment()

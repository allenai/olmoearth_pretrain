"""Trying to prototype fitting everything into olmo core."""

import logging
import uuid

import numpy as np
from olmo_core.distributed.parallel import DataParallelConfig, DataParallelType
from olmo_core.distributed.utils import get_fs_local_rank, get_rank, get_world_size
from olmo_core.optim import AdamWConfig
from olmo_core.train import prepare_training_environment, teardown_training_environment
from olmo_core.train.callbacks import (
    GPUMemoryMonitorCallback,
    ProfilerCallback,
    WandBCallback,
)
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.common import Duration, LoadStrategy
from olmo_core.train.config import TrainerConfig
from olmo_core.utils import get_default_device
from upath import UPath

from helios.data.constants import Modality
from helios.data.dataloader import HeliosDataLoader
from helios.data.dataset import HeliosDataset, collate_helios
from helios.dataset.parse import parse_helios_dataset
from helios.dataset.sample import image_tiles_to_samples
from helios.latent_predictor import LatentMIMStyle
from helios.nn.flexihelios import Encoder, Predictor
from helios.train.callbacks.speed_monitor import HeliosSpeedMonitorCallback
from helios.train.loss import LossConfig
from helios.train.masking import MaskingConfig
from helios.train.train_module import HeliosTrainModuleConfig

logger = logging.getLogger(__name__)

# THings that need a config
# Data Loader
# Model
## OLD LOSS FUNCTION Keeping so pipeline runs until we have new integration


if __name__ == "__main__":
    # Variables to be changed per user
    workdir = UPath("/temp/helios/workdir")  # nosec
    WANDB_USERNAME = "eai-ai2"  # nosec
    WANDB_PROJECT = "helios-debug"
    # PER EXPERIMENT Variables
    GLOBAL_BATCH_SIZE = 1
    RANK_BATCH_SIZE = 1
    MAX_DURATION = Duration.epochs(10)
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
    logger.info("Starting Helios training")

    supported_modalities = [
        Modality.SENTINEL2,
        Modality.LATLON,
        Modality.SENTINEL1,
        # Modality.WORLDCOVER,
    ]
    encoder = Encoder(
        embedding_size=16,
        max_patch_size=8,
        num_heads=2,
        depth=2,
        mlp_ratio=1.0,
        drop_path=0.1,
        max_sequence_length=12,
        base_patch_size=8,
        use_channel_embs=True,
        supported_modalities=supported_modalities,
    )
    decoder = Predictor(
        encoder_embedding_size=16,
        decoder_embedding_size=16,
        depth=2,
        mlp_ratio=1.0,
        num_heads=2,
        max_sequence_length=12,
        max_patch_size=8,
        supported_modalities=supported_modalities,
    )
    model = LatentMIMStyle(encoder, decoder)

    device = get_default_device()
    logger.info(f"Using device: {device}")
    # Ideally though this should be handled by the Model COnfig and build
    model = model.to(device)
    checkpointer_config = CheckpointerConfig(work_dir=workdir)
    optim_config = AdamWConfig()
    masking_config = MaskingConfig(
        strategy_config={
            "type": "random",
        }
    )
    loss_config = LossConfig(
        loss_config={
            "type": "patch_discrimination",
        }
    )
    train_module_config = HeliosTrainModuleConfig(
        optim=optim_config,
        masking_config=masking_config,
        loss_config=loss_config,
        rank_batch_size=RANK_BATCH_SIZE,
    )
    train_module = train_module_config.build(model=model)
    dp_process_group = train_module.dp_process_group

    # Prepare samples from Helios dataset
    tile_path = UPath("/weka/dfive-default/helios/dataset/20250212/")

    tiles = parse_helios_dataset(tile_path, supported_modalities=supported_modalities)

    logger.info(f"Tiles: {len(tiles)}")
    samples = image_tiles_to_samples(tiles, supported_modalities=supported_modalities)
    logger.info(f"Samples: {len(samples)}")
    # Create HeliosDataLoader
    dataloader = HeliosDataLoader(
        dataset=HeliosDataset(
            *samples,
            path=tile_path,
            supported_modalities=supported_modalities,
            dtype=np.dtype("float32"),
        ),
        collator=collate_helios,
        global_batch_size=GLOBAL_BATCH_SIZE,
        dp_world_size=get_world_size(dp_process_group),
        dp_rank=get_rank(dp_process_group),
        fs_local_rank=get_fs_local_rank(),
        work_dir=workdir,
        num_threads=NUM_THREADS,
        num_workers=NUM_WORKERS,
    )

    run_name = f"test-debug-{str(uuid.uuid4())[:8]}"
    wandb_callback = WandBCallback(
        name=run_name,
        project=WANDB_PROJECT,
        entity=WANDB_USERNAME,
        enabled=False,  # set to False to avoid wandb errors
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
        .with_callback("profiler", ProfilerCallback())
    )
    trainer = trainer_config.build(
        train_module=train_module,
        data_loader=dataloader,
    )
    trainer.fit()

    # eval. Currently this will fail because by default our model ingests 4 timesteps.
    # we should update the model architecture to ingest variable numbers of timesteps
    from torch.utils.data import DataLoader

    from helios.evals.datasets import GeobenchDataset
    from helios.evals.embeddings import get_embeddings
    from helios.evals.knn import run_knn

    geobench_dir = UPath("/weka/skylight-default/presto-geobench/dataset/geobench")

    common_args = {"geobench_dir": geobench_dir, "dataset": "m-eurosat"}
    train_ds = GeobenchDataset(geobench_dir, "m-eurosat", "train", "default")
    train_loader = DataLoader(train_ds, collate_fn=GeobenchDataset.collate_fn)
    val_loader = DataLoader(
        GeobenchDataset(geobench_dir, "m-eurosat", "valid", "default"),
        collate_fn=GeobenchDataset.collate_fn,
    )
    # TODO: this should use target encoder
    train_embeddings, train_labels = get_embeddings(
        data_loader=train_loader, model=encoder
    )
    val_embeddings, test_labels = get_embeddings(data_loader=val_loader, model=encoder)
    val_result = run_knn(
        eval_type="KNN-20",
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        test_embeddings=val_embeddings,
        test_labels=test_labels,
        num_classes=train_ds.num_classes,
        is_multilabel=train_ds.is_multilabel,
        device=device,
    )
    logger.info(val_result)
    teardown_training_environment()

"""Trying to prototype fitting everything into olmo core.

TO run this script please clone the olmo-core repo and pip install the latest version of olmo-core.

The released version on pypi is behind what is used here.

"""

import logging

import numpy as np
from olmo_core.utils import setup_logging
from upath import UPath

from helios.data.collator import per_modality_collate_fn
from helios.data.dataloader import HeliosDataLoader
from helios.data.dataset import HeliosDataset
from helios.dataset.index import DatasetIndexParser
from helios.train.decoder import SimpleLatentDecoder
from helios.train.loss import patch_disc_loss

logger = logging.getLogger(__name__)


## Config does not yet support our new dataset type so we will construct manually for now


if __name__ == "__main__":
    setup_logging()
    # set log level to debug
    logger.setLevel(logging.DEBUG)

    index_path = "/weka/dfive-default/helios_sample_data/20250115-sample-dataset-helios/index.csv"
    index_parser = DatasetIndexParser(index_path)
    samples = index_parser.samples
    workdir = UPath("/Users/henryh/Desktop/eai-repos/helios-repos/helios/workdir")
    dataloader = HeliosDataLoader.wrap_numpy_dataset(
        dataset=HeliosDataset(
            *samples,
            ignore_data_sources=["openstreetmap"],
            filter_samples_with_missing_inputs=True,
            dtype=np.dtype("float32"),
        ),
        global_batch_size=4,
        dp_world_size=1,
        collator=per_modality_collate_fn,
        work_dir=workdir,
        num_threads=0,
        num_workers=2,
    )

    from helios.latent_predictor import LatentMIMStyle
    from helios.train.encoder import PatchEncoder
    from helios.train.trainer import HeliosTrainer

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

    from olmo_core.optim import AdamWConfig
    from olmo_core.train.callbacks.wandb import WandBCallback
    from olmo_core.train.checkpoint import CheckpointerConfig
    from olmo_core.train.common import Duration, LoadStrategy
    from olmo_core.utils import get_default_device

    from helios.train.callbacks.speed_monitor import HeliosSpeedMonitorCallback
    max_duration = Duration.epochs(4)

    checkpointer_config = CheckpointerConfig(work_dir=workdir)
    checkpointer = checkpointer_config.build()
    DEVICE = get_default_device()
    model = model.to(DEVICE)
    optim_config = AdamWConfig()
    from helios.train.train_module import HeliosTrainModule

    train_module = HeliosTrainModule(
        model=model,
        optim=optim_config,
        rank_batch_size=4,
        loss_fn=patch_disc_loss,
    )
    import uuid
    run_name = f"test-debug-{uuid.uuid4()[:8]}"
    wandb_callback = WandBCallback(
        name=run_name,
        project="helios-test",
        entity="henryhzog",#PLEASE CHANGE
    )
    trainer = HeliosTrainer(
        work_dir=workdir,
        train_module=train_module,
        data_loader=dataloader,
        load_strategy=LoadStrategy.if_available,
        device=DEVICE,
        save_folder=workdir / "save_folder",
        callbacks={"speed_monitor": HeliosSpeedMonitorCallback(), "wandb": wandb_callback},
        cancel_check_interval=1,
        metrics_collect_interval=1,
        max_duration=max_duration,
        checkpointer=checkpointer,
    )

    trainer.fit()

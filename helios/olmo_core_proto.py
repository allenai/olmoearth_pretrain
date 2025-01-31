"""Trying to prototype fitting everything into olmo core.

TO run this script please clone the olmo-core repo and pip install the latest version of olmo-core.

The released version on pypi is behind what is used here.

"""

import logging

import numpy as np
import torch
from olmo_core.utils import setup_logging
from upath import UPath

from helios.data.collator import per_modality_collate_fn
from helios.data.dataloader import HeliosDataLoader
from helios.data.dataset import HeliosDataset
from helios.dataset.index import DatasetIndexParser
from helios.train.decoder import SimpleLatentDecoder

logger = logging.getLogger(__name__)


## Config does not yet support our new dataset type so we will construct manually for now


if __name__ == "__main__":
    setup_logging()
    # set log level to debug
    logger.setLevel(logging.DEBUG)

    index_path_old = "gs://ai2-helios/data/20250113-sample-dataset-helios/index.csv"
    index_path = "gs://ai2-helios/data/20250115-sample-dataset-helios/index.csv"
    index_parser = DatasetIndexParser(index_path)
    samples = index_parser.samples
    workdir = UPath("/weka/dfive-default/henryh/helios/workdir")
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
        num_workers=0,
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
    # target_encoder = deepcopy(encoder)
    # for p in target_encoder.parameters():
    #     p.requires_grad = False

    # # we will need to keep the state of momentum so that we can resume training
    # # Add EMA decay rate and optimizer
    # ema_decay = 0.99
    # optimizer = torch.optim.AdamW(
    #     list(encoder.parameters()) + list(decoder.parameters()),
    #     lr=1e-4,
    #     weight_decay=0.01
    # )

    # for epoch in range(1, 3):
    #     dataloader.reshuffle(epoch=epoch)
    #     batch_iterator = dataloader._iter_batches()
    #     batches_found = 0
    #     for batch in batch_iterator:
    #         optimizer.zero_grad()

    #         with torch.no_grad():
    #             input = rearrange(batch.sentinel2, "b h w t c -> b c t h w")
    #             target_output = target_encoder.forward(input)

    #         # Run Encoder and decoder on the augmented input
    #         latent = encoder.forward(input, apply_aug=True)
    #         decoded = decoder.forward(latent["encoded"])
    #         loss = patch_disc_loss(
    #             pred=decoded,
    #             target=target_output["encoded"],
    #             pred2unit=True,
    #         )

    #         # Backpropagate and optimize
    #         loss.backward()
    #         optimizer.step()

    #         print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    #         # Update target encoder with EMA
    #         with torch.no_grad():
    #             for param, target_param in zip(
    #                 encoder.parameters(), target_encoder.parameters()
    #             ):
    #                 target_param.data = (
    #                     ema_decay * target_param.data + (1 - ema_decay) * param.data
    #                 )

    #     dataloader.reset()

    # Need an optimizer
    # Need a checkpointer
    # Need a module
    # first lets grab the anysat model already in there repo

    from olmo_core.optim import AdamWConfig
    from olmo_core.train.checkpoint import CheckpointerConfig
    from olmo_core.train.common import Duration, LoadStrategy

    from helios.train.callbacks.speed_monitor import HeliosSpeedMonitorCallback

    max_duration = Duration.epochs(4)

    checkpointer_config = CheckpointerConfig(work_dir=workdir)
    checkpointer = checkpointer_config.build()
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(DEVICE)
    optim_config = AdamWConfig()
    optim = optim_config.build(model)
    trainer = HeliosTrainer(
        work_dir=workdir,
        model=model,
        optim=optim,
        data_loader=dataloader,
        load_strategy=LoadStrategy.never,
        device=DEVICE,
        save_folder=workdir / "save_folder",
        callbacks={"speed_monitor": HeliosSpeedMonitorCallback()},
        rank_microbatch_size=4,
        cancel_check_interval=1,
        max_duration=max_duration,
        checkpointer=checkpointer,
    )

    trainer.fit()

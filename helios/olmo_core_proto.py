"""Trying to prototype fitting everything into olmo core."""

import logging
import time

import numpy as np
from olmo_core.utils import setup_logging
from upath import UPath

from helios.data.collator import variable_time_collate_fn
from helios.data.dataloader import HeliosDataLoader
from helios.data.dataset import HeliosDataset
from helios.helios.dataset.index import DatasetIndexParser

logger = logging.getLogger(__name__)


## Config does not yet support our new dataset type so we will construct manually for now
if __name__ == "__main__":
    setup_logging()

    index_path = "gs://ai2-helios/data/20250113-sample-dataset-helios/index.csv"
    index_parser = DatasetIndexParser(index_path)
    samples = index_parser.samples
    workdir = UPath("/Users/henryh/Desktop/eai-repos/helios-repos/helios/workdir")
    dataloader = HeliosDataLoader.wrap_numpy_dataset(
        dataset=HeliosDataset(*samples, dtype=np.dtype("float32")),
        global_batch_size=4,
        collator=variable_time_collate_fn,
        work_dir=workdir,
        num_threads=4,
    )

    # potentially missing dataset prepare
    for epoch in range(1, 3):
        dataloader.reshuffle(epoch=epoch)
        batch_iterator = dataloader._iter_batches()
        # Need to call reshuffle
        batches_found = 0
        batch_start = time.time()
        for batch in batch_iterator:
            batch_end = time.time()
            if batches_found > 0:
                logger.info(f"batch time {batch_end - batch_start}")
            batches_found += 1
            time.sleep(10)
            batch_start = time.time()
            logger.info("batch found")
        dataloader.reset()

    # need to call reset after the epich

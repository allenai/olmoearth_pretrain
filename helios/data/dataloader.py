"""Helios DataLoader."""

import logging
import math
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from olmo_core.config import Config
from olmo_core.data.data_loader import DataLoaderBase
from olmo_core.data.utils import get_rng, memmap_to_write
from olmo_core.distributed.utils import (
    barrier,
    get_fs_local_rank,
    get_rank,
    get_world_size,
)
from olmo_core.utils import roundrobin, threaded_generator
from torch.utils.data import default_collate
from upath import UPath

from helios.data.constants import Modality
from helios.data.dataset import HeliosDataset, HeliosSample

logger = logging.getLogger(__name__)


class HeliosDataLoader(DataLoaderBase):
    """Helios dataloader.

    This dataloader is adapted from OLMo-core's TextDataLoaderBase and NumpyDataLoaderBase,
    incorporating their core functionality for DDP, multi-threading, and multi-processing.
    """

    def __init__(
        self,
        dataset: HeliosDataset,
        work_dir: UPath,
        global_batch_size: int,
        dp_world_size: int = 1,
        dp_rank: int = 0,
        fs_local_rank: int = 0,
        seed: int = 0,
        shuffle: bool = True,
        num_threads: int | None = None,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        collator: Callable = default_collate,
        target_device_type: str = "cpu",
        drop_last: bool = True,
        persistent_workers: bool = True,
    ):
        """Initialize the HeliosDataLoader."""
        super().__init__(
            work_dir=work_dir,
            global_batch_size=global_batch_size,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            fs_local_rank=fs_local_rank,
        )
        self.dataset = dataset
        assert isinstance(self.dataset, HeliosDataset)  # type: ignore
        self.collator = collator
        self.seed = seed
        self.shuffle = shuffle
        self.num_threads = num_threads
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.target_device_type = target_device_type
        self.drop_last = drop_last
        self._global_indices: np.ndarray | None = None
        self.persistent_workers = persistent_workers

    @property
    def total_batches(self) -> int:
        """The total number of batches in an epoch."""
        return len(self.dataset) // (self.global_batch_size)

    @property
    def total_size(self) -> int:
        """The total number of instances in an epoch."""
        return self.total_batches * self.global_batch_size

    @property
    def _global_indices_file(self) -> UPath:
        """Global indices file."""
        global_indices_fname = self._format_fname_from_fields(
            "global_indices",
            seed=self.seed if self.shuffle else None,
            epoch=self.epoch if self.shuffle else None,  # type: ignore
            size=self.total_size,
        )
        return (
            Path(self.work_dir)
            / f"dataset-{self.dataset.fingerprint}"
            / f"{global_indices_fname}.npy"
        )

    def _build_global_indices(self) -> np.ndarray:
        """Build global indices."""
        assert len(self.dataset) < np.iinfo(np.uint32).max

        rng: np.random.Generator | None = None
        if self.shuffle:
            # Deterministically shuffle based on epoch and seed
            rng = get_rng(self.seed + self.epoch)  # type: ignore
        indices: np.ndarray
        indices = np.arange(len(self.dataset), dtype=np.uint32)
        if rng is not None:
            rng.shuffle(indices)
        # Remove tail of data to make it evenly divisible
        indices = indices[: self.total_size]
        return indices

    def build_and_save_global_indices(self, in_memory: bool = False) -> None:
        """Build and save global indices."""
        if in_memory:
            self._global_indices = self._build_global_indices()
        else:
            self._global_indices = None
            if self.fs_local_rank == 0:
                # Either load from file or build and save to file
                if self._global_indices_file.is_file():
                    logger.info(
                        f"Using existing global indices file for seed {self.seed} and epoch {self.epoch}"  # type: ignore
                        f"at:\n'{self._global_indices_file}'"
                    )
                else:
                    global_indices = self._build_global_indices()
                    assert (
                        len(global_indices) < np.iinfo(np.int32).max
                    )  # Note: OLMo uses uint32
                    with memmap_to_write(
                        self._global_indices_file,
                        shape=global_indices.shape,
                        dtype=np.int32,
                    ) as global_indices_mmap:
                        global_indices_mmap[:] = global_indices
                    logger.info(
                        f"Global data order indices saved to:\n'{self._global_indices_file}'"
                    )
        barrier()

    def reshuffle(
        self, epoch: int | None = None, in_memory: bool = False, **kwargs: Any
    ) -> None:
        """Reshuffle the data."""
        del kwargs
        if epoch is None:
            epoch = 1 if self._epoch is None else self._epoch + 1  # type: ignore
        if epoch <= 0:
            raise ValueError(f"'epoch' must be at least 1, got {epoch}")
        self._epoch = epoch
        # Since epoch has been updated, we need to create new global indices
        self.build_and_save_global_indices(in_memory=in_memory)

    def get_global_indices(self) -> np.ndarray:
        """Get global indices."""
        # Either load from memory or file
        if self._global_indices is not None:
            return self._global_indices
        if not self._global_indices_file.is_file():
            raise RuntimeError(
                "Missing global indices file, did you forget to call 'reshuffle()'?"
            )
        return np.memmap(self._global_indices_file, mode="r", dtype=np.uint32)

    def _iter_batches(self) -> Iterable[HeliosSample]:
        """Iterate over the dataset in batches."""
        return torch.utils.data.DataLoader(
            _IterableDatasetWrapper(self),
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.target_device_type == "cuda" and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers
            if self.num_workers > 0
            else False,
            timeout=0,
        )

    @property
    def worker_info(self):  # type: ignore
        """Get worker info."""
        return torch.utils.data.get_worker_info()

    def _get_local_instance_indices(self, indices: np.ndarray) -> Iterable[int]:
        """Get local instance indices."""
        # NOTE:'indices' are global instance indices.
        instances_per_batch = self.global_batch_size
        indices = indices.reshape(-1, instances_per_batch)

        # Offset by the number of batches already processed.
        if self.batches_processed > 0:  # type: ignore
            indices = indices[self.batches_processed :]  # type: ignore

        # Slice batches by data loader worker rank to avoid duplicates.
        if (worker_info := self.worker_info) is not None:
            indices = indices[worker_info.id :: worker_info.num_workers]

        # Finally slice batches into micro batches for the local DP rank.
        indices = indices[:, self.dp_rank :: self.dp_world_size].reshape((-1,))
        return indices

    def _get_dataset_item(self, idx: int) -> HeliosSample:
        """Get a dataset item."""
        item = self.dataset[idx]
        return item

    def state_dict(self) -> dict[str, Any]:
        """Get the state dict."""
        return {
            "dataset_fingerprint_version": self.dataset.fingerprint_version,
            "dataset_fingerprint": self.dataset.fingerprint,
            "batches_processed": self.batches_processed,  # type: ignore
            "seed": self.seed,
            "epoch": self._epoch,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the state dict."""
        if (
            state_dict["dataset_fingerprint_version"]
            != self.dataset.fingerprint_version
        ):
            logger.warning(
                "Dataset fingerprint version does not match the version in the checkpoint, "
                "this could mean the data has changed"
            )
        elif state_dict["dataset_fingerprint"] != self.dataset.fingerprint:
            raise RuntimeError(
                "Restoring state from a different dataset is not supported! (fingerprint doesn't match)"
            )

        if state_dict["seed"] != self.seed:
            logger.warning(
                "Restoring data loading state with a different data seed, "
                "will use data seed from state dict for data order consistency."
            )
            self.seed = state_dict["seed"]

        self.batches_processed = state_dict["batches_processed"]
        self._epoch = state_dict["epoch"] or self._epoch  # type: ignore

    def _format_fname_from_fields(self, prefix: str, **fields: Any) -> str:
        parts = [prefix]
        for key in sorted(fields):
            value = fields[key]
            if value is not None:
                parts.append(f"{key}{value}")
        return "_".join(parts)

    def get_mock_batch(self) -> HeliosSample:
        """Get a mock batch, for dry-run of forward and backward pass."""
        logger.info("Getting mock batch NOT FROM DATASET")
        # TODO: This should be a feature of the modality spec
        output_dict = {}
        if Modality.SENTINEL2_L2A in self.dataset.supported_modalities:
            mock_sentinel2_l2a = torch.rand(1, 256, 256, 12, 12)
            output_dict["sentinel2_l2a"] = mock_sentinel2_l2a
        if Modality.SENTINEL1 in self.dataset.supported_modalities:
            mock_sentinel1 = torch.rand(1, 256, 256, 12, 2)
            output_dict["sentinel1"] = mock_sentinel1
        if Modality.WORLDCOVER in self.dataset.supported_modalities:
            mock_worldcover = torch.rand(1, 256, 256, 1, 1)
            output_dict["worldcover"] = mock_worldcover
        if Modality.LATLON in self.dataset.supported_modalities:
            mock_latlon = torch.rand(1, 2)
            output_dict["latlon"] = mock_latlon
        days = torch.randint(0, 25, (1, 1, 12), dtype=torch.long)
        months = torch.randint(0, 12, (1, 1, 12), dtype=torch.long)
        years = torch.randint(2018, 2020, (1, 1, 12), dtype=torch.long)
        timestamps = torch.cat([days, months, years], dim=1)
        timestamps = rearrange(timestamps, "b t c -> b c t")
        output_dict["timestamps"] = timestamps
        return HeliosSample(**output_dict)

    def fast_forward(self, global_step: int) -> np.ndarray:
        """Fast forward the data loader to a specific global step and return the batch_indices."""
        logger.warning(
            "Fast forward does not yet support returning to indices for multiple GPUs"
        )
        if get_world_size() > 1:
            raise NotImplementedError("Fast forward is not supported in DDP")
        # If the model was trained with multiple GPUS, this logic must be updated so that we grab from where all the ranks started
        self.batches_processed = global_step
        epoch = math.ceil(global_step / self.total_batches)
        step_in_epoch = global_step % self.total_batches
        logger.info(f"epoch: {epoch}, step in epoch: {step_in_epoch}")
        for i in range(1, epoch + 1):
            self.reshuffle(epoch=i)
        batch_start = int(self.get_global_indices()[step_in_epoch])
        batch_end = batch_start + self.global_batch_size
        sample_indices = np.arange(batch_start, batch_end)
        return sample_indices


def iter_batched(
    iterable: Iterable[HeliosSample], batch_size: int, drop_last: bool = True
) -> Iterable[tuple[HeliosSample, ...]]:
    """Iterate over the dataset in batches.

    This is a modified version of olmo_core.data.data_loader.iter_batched that creates batches
    of size local_batch_size for the local rank from an iterator of items.


    Args:
        iterable: The iterator of items to batch.
        batch_size: The size of the batches to create for the local rank.
        drop_last: Whether to drop the last batch if it's not full.

    Returns:
        An iterator of batches of items.
    """
    assert batch_size > 0
    batch: list[HeliosSample] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield tuple(batch)
            batch.clear()

    # If there's a partial batch left over, yield it if `drop_last` is False
    if not drop_last and batch:
        yield tuple(batch)


class _IterableDatasetWrapper(torch.utils.data.IterableDataset[HeliosSample]):
    """Iterable dataset wrapper.

    This is a modified version of olmo_core.data.data_loader._IterableDatasetWrapper
    """

    def __init__(self, data_loader: HeliosDataLoader):
        """Initialize the IterableDatasetWrapper."""
        self.data_loader = data_loader

    @property
    def dataset(self) -> HeliosDataset:
        """Get the dataset."""
        return self.data_loader.dataset

    @property
    def worker_info(self):  # type: ignore
        """Get worker info."""
        return torch.utils.data.get_worker_info()

    def __iter__(self) -> Iterator[HeliosSample]:
        """Iterate over the dataset."""
        global_indices = self.data_loader.get_global_indices()

        num_threads = self.data_loader.num_threads
        if self.worker_info is None and self.data_loader.num_threads is None:
            # If `num_threads` hasn't been specified and we're not using multiprocessing we'll
            # try to guess a good number of threads.
            num_threads = 4

        # Potentially slice by threads.
        instance_iterator: Iterator[HeliosSample]
        if num_threads:
            # In order to stay ahead of training the total queue size (sum across all threads)
            # should be bigger than the batch size per rank.
            queue_size = math.ceil(self.data_loader.rank_batch_size * 2 / num_threads)
            thread_generators = []
            for i in range(num_threads):
                indices = self.data_loader._get_local_instance_indices(global_indices)
                generator = (
                    self.data_loader._get_dataset_item(int(idx))
                    for idx in islice(indices, i, None, num_threads)
                )
                thread_generators.append(
                    threaded_generator(
                        generator, maxsize=queue_size, thread_name=f"data thread {i}"
                    )
                )
            instance_iterator = roundrobin(*thread_generators)
        else:
            indices = self.data_loader._get_local_instance_indices(global_indices)
            instance_iterator = (
                self.data_loader._get_dataset_item(int(idx)) for idx in indices
            )

        return (
            self.data_loader.collator(batch)
            for batch in iter_batched(
                instance_iterator,
                self.data_loader.rank_batch_size,
                self.data_loader.drop_last,
            )
        )


@dataclass
class HeliosDataLoaderConfig(Config):
    """Configuration for the HeliosDataLoader."""

    work_dir: str
    global_batch_size: int
    seed: int
    shuffle: bool = True
    num_threads: int | None = None
    num_workers: int = 0
    prefetch_factor: int | None = None
    target_device_type: str = "cpu"
    drop_last: bool = True

    def validate(self) -> None:
        """Validate the configuration."""
        if self.work_dir is None:
            raise ValueError("Work directory is not set")

    @property
    def work_dir_upath(self) -> UPath:
        """Get the work directory."""
        return UPath(self.work_dir)

    def build(
        self,
        dataset: HeliosDataset,
        collator: Callable,
        dp_process_group: dist.ProcessGroup | None = None,
    ) -> "HeliosDataLoader":
        """Build the HeliosDataLoader."""
        self.validate()
        if not isinstance(dataset, HeliosDataset):
            raise ValueError("Dataset must be a HeliosDataset")
        dataset.prepare()

        return HeliosDataLoader(
            dataset=dataset,
            work_dir=self.work_dir_upath,
            global_batch_size=self.global_batch_size,
            dp_world_size=get_world_size(dp_process_group),
            dp_rank=get_rank(dp_process_group),
            fs_local_rank=get_fs_local_rank(),
            seed=self.seed,
            shuffle=self.shuffle,
            num_threads=self.num_threads,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            target_device_type=self.target_device_type,
            collator=collator,
            drop_last=self.drop_last,
        )

"""Helios DataLoader."""

import logging
import math
from collections.abc import Callable, Iterable, Iterator
from itertools import islice
from typing import Any

import numpy as np
import torch
from olmo_core.data.data_loader import DataLoaderBase
from olmo_core.data.utils import get_rng, memmap_to_write
from olmo_core.utils import roundrobin, threaded_generator
from olmo_core.distributed.utils import get_fs_local_rank, barrier, get_rank, get_world_size
from upath import UPath

from helios.data.dataset import HeliosDataset

logger = logging.getLogger(__name__)


class HeliosDataLoader(DataLoaderBase):
    """Helios data loader."""
    def __init__(
        self,
        dataset: HeliosDataset,
        collator: Callable,
        work_dir: UPath,
        global_batch_size: int,
        dp_world_size: int = 1,
        dp_rank: int = 0,
        fs_local_rank: int = 0,
        seed: int = 0,
        shuffle: bool = True,
        num_threads: Optional[int] = None,
        num_workers: int = 0,
        prefetch_factor: Optional[int] = None,
        target_device_type: str = "cpu",
    ):
        super().__init__(
            work_dir=work_dir,
            global_batch_size=global_batch_size,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            fs_local_rank=fs_local_rank
        )
        assert isinstance(self.dataset, HeliosDataset)
        self.dataset = dataset
        self.collator = collator
        self.seed = seed
        self.shuffle = shuffle
        self.num_threads = num_threads
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.target_device_type = target_device_type

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
            seed = self.seed if self.shuffle else None,
            epoch = self.epoch if self.shuffle else None,
            size = self.total_size
        )
        return (
            Path(self.work_dir)
            / f"dataset-{self.dataset.fingerprint}"
            / f"{global_indices_fname}.npy"
        )

    def _build_global_indices(self) -> np.ndarray:
        """Build global indices."""
        assert len(dataset) < np.iinfo(np.uint32).max

        rng: Optional[np.random.Generator] = None
        if self.shuffle:
            # Deterministically shuffle based on epoch and seed
            rng = get_rng(self.seed + self.epoch)
        indices: np.ndarray
        indices = np.arange(len(self.dataset), dtype=np.uint32)
        if rng is not None:
            rng.shuffle(indices)
        # Remove tail of data to make it evenly divisible
        indices = indices[: self.total_size]
        return indices

    def build_and_save_global_indices(self, in_memory: bool = False):
        """Build and save global indices."""
        if in_memory:
            self._global_indices = self._build_global_indices()
        else:
            self._global_indices = None
            if self.fs_local_rank == 0:
                # Either load from file or build and save to file
                if self._global_indices_file.is_file():
                    logger.info(
                        f"Using existing global indices file for seed {self.seed} and epoch {self.epoch}"
                        f"at:\n'{self._global_indices_file}'"
                    )
                else:
                    global_indices = self._build_global_indices()
                    assert len(global_indices) < np.iinfo(np.unit32).max
                    with memmap_to_write(
                        self._global_indices_file, shape=global_indices.shape, dtype=np.uint32
                    ) as global_indices_mmap:
                        global_indices_mmap[:] = global_indices
                    logger.info(f"Global data order indices saved to:\n'{self._global_indices_file}'")
        barrier()

    def reshuffle(self, epoch: Optional[int] = None, in_memory: bool = False, **kwargs):
        """Reshuffle the data."""
        del kwargs
        if epoch is None:
            epoch = 1 if self.epoch is None else self.epoch + 1
        if epoch <= 0:
            raise ValueError(f"'epoch' must be at least 1, got {epoch}")
        self.epoch = epoch
        # Since epoch has been updated, we need to create new global indices
        self.build_and_save_global_indices(in_memory=in_memory)

    def get_global_indices(self) -> np.ndarray:
        """Get global indices."""
        # Either load from memory or file
        if self._global_indices is not None:
            return self._global_indices
        if not self._global_indices_file.is_file():
            raise RuntimeError("Missing global indices file, did you forget to call 'reshuffle()'?")
        return np.memmap(self._global_indices_file, mode="r", dtype=np.uint32)

    def _iter_batches(self) -> Iterable[Dict[str, Any]]:
        """Iterate over the dataset in batches."""
        return torch.utils.data.DataLoader(
            _IterableDatasetWrapper(self),
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.target_device_type == "cuda" and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=False,
            timeout=0,
        )

    @property
    def worker_info(self):
        """Get worker info."""
        return torch.utils.data.get_worker_info()

    def _get_local_instance_indices(self, indices: np.ndarray) -> Iterable[int]:
        """Get local instance indices."""
        # NOTE:'indices' are global instance indices.
        instances_per_batch = self.global_batch_size
        indices = indices.reshape(-1, instances_per_batch) 

        # Offset by the number of batches already processed.
        if self.batches_processed > 0:
            indices = indices[self.batches_processed:]
        
        # Slice batches by data loader worker rank to avoid duplicates.
        if (worker_info := self.worker_info) is not None:
            indices = indices[worker_info.id::worker_info.num_workers]
        
        # Finally slice batches into micro batches for the local DP rank.
        indices = indices[:, self.dp_rank::self.dp_world_size].reshape((-1,))
        return indices

    def _get_dataset_item(self, idx: int) -> Dict[str, Any]:
        """Get a dataset item."""
        item = self.dataset[idx]
        if isinstance(item, dict):
            return dict(**item, index=idx)
        else:
            # TODO: Check if this is correct
            return {"index": idx}

    def state_dict(self) -> Dict[str, Any]:
        """Get the state dict."""
        return {
            "dataset_fingerprint_version": self.dataset.fingerprint_version,
            "dataset_fingerprint": self.dataset.fingerprint,
            "batches_processed": self.batches_processed,
            "seed": self.seed,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load the state dict."""
        if state_dict["dataset_fingerprint_version"] != self.dataset.fingerprint_version:
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
        self.epoch = state_dict["epoch"] or self.epoch
    
    def _format_fname_from_fields(self, prefix: str, **fields) -> str:
        parts = [prefix]
        for key in sorted(fields):
            value = fields[key]
            if value is not None:
                parts.append(f"{key}{value}")
        return "_".join(parts)

    def get_mock_batch(self) -> Dict[str, Any]:
        """Get a mock batch, for dry-run of forward and backward pass."""
        # TODO: add a batch of arbitrary data
        pass


class _IterableDatasetWrapper(torch.utils.data.IterableDataset[Dict[str, Any]]):
    """Iterable dataset wrapper.
    
    This is a modified version of olmo_core.data.data_loader._IterableDatasetWrapper
    """
    def __init__(self, data_loader: HeliosDataLoader):
        self.data_loader = data_loader

    @property
    def dataset(self) -> HeliosDataset:
        return self.data_loader.dataset
    
    @property
    def worker_info(self):
        return torch.utils.data.get_worker_info()
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over the dataset."""
        global_indices = self.data_loader.get_global_indices()
       
        num_threads = self.data_loader.num_threads
        if self.worker_info is None and self.data_loader.num_threads is None:
            # If `num_threads` hasn't been specified and we're not using multiprocessing we'll
            # try to guess a good number of threads.
            num_threads = 4
        
        # Potentially slice by threads.
        instance_iterator: Iterator[Dict[str, Any]]
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
            instance_iterator = (self.data_loader._get_dataset_item(int(idx)) for idx in indices)
        
        return (
            self.data_loader.collator(batch)
            for batch in iter_batched(instance_iterator, self.data_loader.rank_batch_size)
        )

# TODO: we also need a configuration class for HeliosDataLoader
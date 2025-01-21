"""Helios DataLoader."""

import logging
import math
from collections.abc import Callable, Iterable, Iterator
from itertools import islice
from typing import Any

import numpy as np
import torch
from olmo_core.data.data_loader import NumpyDataLoaderBase, _IterableDatasetWrapper
from olmo_core.data.numpy_dataset import NumpyDatasetBase
from olmo_core.data.utils import get_rng
from olmo_core.utils import roundrobin, threaded_generator
from upath import UPath

from helios.data.dataset import HeliosDataset

logger = logging.getLogger(__name__)


def iter_batched(
    iterable: Iterable[dict[str, Any]], local_batch_size: int
) -> Iterable[tuple[dict[str, Any], ...]]:
    """Iterate over the dataset in batches.

    This is a modified version of olmo_core.data.data_loader.iter_batched that creates batches
    of size local_batch_size for the local rank from an iterator of items.

    Args:
        iterable: The iterator of items to batch.
        local_batch_size: The size of the batches to create for the local rank.

    Returns:
        An iterator of batches of items.
    """
    batch: list[dict[str, Any]] = []
    instances = 0
    # shape: Optional[tuple[int, ...]] = None
    for x in iterable:
        if instances > local_batch_size:
            yield tuple(batch)
            batch.clear()
            instances = 0
            # shape = None

        batch.append(x)
        instances += 1

        # TODO:  Our shape checking is more complex we likely should do this later
        # if shape is not None and shape != x["input_ids"].shape:
        #     raise RuntimeError(
        #         f"Items in batch don't have the same shape! Expected {shape}, "
        #         f"got {tuple(x['input_ids'].shape)}"
        #     )
        # shape = tuple(x["input_ids"].shape)

    if batch:
        yield tuple(batch)


class HeliosIterableDatasetWrapper(_IterableDatasetWrapper):
    """Helios iterable dataset wrapper.

    This is a modified version of olmo_core.data.data_loader._IterableDatasetWrapper that
    creates batches of size local_batch_size for the local rank from an iterator of items using
    multi-threading.
    """

    def __iter__(self) -> Iterator[dict]:
        """Iterate over the dataset.

        Customized iteratormethod based on olmo_core.data.data_loader._IterableDatasetWrapper
        """
        global_indices = self.data_loader.get_global_indices()

        num_threads = self.data_loader.num_threads
        if self.worker_info is None and self.data_loader.num_threads is None:
            # If `num_threads` hasn't been specified and we're not using multiprocessing we'll
            # try to guess a good number of threads.
            num_threads = 4

        # Potentially slice by threads.
        instance_iterator: Iterator[dict[str, Any]]
        if num_threads:
            # In order to stay ahead of training the total queue size (sum across all threads)
            # should be bigger than the maximum number of instances per batch locally.
            max_instances_per_rank: int
            if isinstance(self.dataset, HeliosDataset):
                max_instances_per_rank = self.data_loader.rank_batch_size
            else:
                raise NotImplementedError

            queue_size = math.ceil(max_instances_per_rank * 2 / num_threads)

            thread_generators = []
            for i in range(num_threads):
                # NOTE: `_get_local_instance_indices` might return an iterator, so we have to
                # create a unique one for each thread otherwise it would be exhausted prematurely
                # and give the wrong order.
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
        print(f"data loader rank batch size {self.data_loader.rank_batch_size}")
        return (
            self.data_loader.collator(batch)
            for batch in iter_batched(
                instance_iterator, self.data_loader.rank_batch_size
            )
        )


class HeliosDataLoader(NumpyDataLoaderBase):
    """Helios data loader."""

    def __init__(
        self,
        dataset: NumpyDatasetBase,
        *,  # type: ignore
        collator: Callable,
        **kwargs: Any,  # type: ignore
    ):
        """Initialize the data loader."""
        super().__init__(
            dataset=dataset,
            collator=collator,
            **kwargs,
        )

    @classmethod
    def wrap_numpy_dataset(
        cls,
        dataset: NumpyDatasetBase,
        *,
        global_batch_size: int,
        collator: Callable,
        work_dir: UPath | None = None,
        seed: int = 0,
        dp_world_size: int = 1,
        dp_rank: int = 0,
        fs_local_rank: int = 0,
        num_threads: int | None = None,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        target_device_type: str = "cpu",
    ) -> "NumpyDataLoaderBase":
        """Construct the corresponding :class:`NumpyDataLoaderBase` instance for the given :class:`NumpyDatasetBase`.

        This is a modified version of olmo_core.data.data_loader.wrap_numpy_dataset

        Args:
            dataset: The dataset to wrap.
            global_batch_size: The global batch size.
            collator: The collator to use.
            work_dir: The work directory.
            dp_world_size: The number of data parallel workers.
            dp_rank: The data parallel rank.
            fs_local_rank: The file system local rank.
            seed: The seed to use.
            num_threads: The number of threads to use.
            num_workers: The number of workers to use.
            prefetch_factor: The prefetch factor.
            target_device_type: The target device type.

        Returns:
            The wrapped data loader.

        Raises:
            NotImplementedError: If the dataset is not a HeliosDataset.
        """
        kwargs = dict(
            global_batch_size=global_batch_size,
            collator=collator,
            work_dir=work_dir or dataset.work_dir,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            fs_local_rank=fs_local_rank,
            seed=seed,
            num_threads=num_threads,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            target_device_type=target_device_type,
        )
        if isinstance(dataset, HeliosDataset):
            data_loader = HeliosDataLoader(
                dataset,
                **kwargs,  # type: ignore
            )
        else:
            raise NotImplementedError

        return data_loader

    @property
    def _global_indices_file(self) -> UPath:
        """Global indices file."""
        global_indices_fname = self._format_fname_from_fields(
            "global_indices",
            seed=self.seed if self.shuffle else None,
            epoch=self.epoch if self.shuffle else None,
            size=self.total_size,
        )
        return UPath(self.work_dir) / f"{global_indices_fname}.npy"

    def _build_global_indices(self) -> np.ndarray:
        """Build global indices.

        This an array of all the training indices that wil be used globally in the distributed
        setup.
        """
        rng: np.random.Generator | None = None
        if self.shuffle:
            # Deterministically shuffle based on epoch and seed
            rng = get_rng(self.seed + self.epoch)
        indices = np.arange(len(self.dataset), dtype=np.uint32)
        if rng is not None:
            rng.shuffle(indices)
        # what shape should this be?
        # TODO:Remove tail of data to make it evenly divisible, not sure yet if we need this
        logger.debug(f"indices shape before removing tail {indices.shape}")
        indices = indices[: self.total_size]
        logger.debug(f"indices shape after removing tail {indices.shape}")
        return indices

    def _get_local_instance_indices(self, indices: np.ndarray) -> Iterable[int]:
        """Get local instance indices."""
        assert isinstance(self.dataset, HeliosDataset)
        # Slice up by batch.
        instances_per_batch = self.global_batch_size
        # shape: (global num batches, global num instances per batch)
        logger.debug(f"indices shape before reshape {indices.shape}")
        indices = indices.reshape(-1, instances_per_batch)
        logger.debug(
            f"indices shape after reshape to (global num batches, global num instances per batch) {indices.shape}"
        )

        # Offset by the number of batches already processed.
        if self.batches_processed > 0:
            indices = indices[self.batches_processed :]
        logger.debug(f"indices shape after offset {indices.shape}")

        # Slice batches by data loader worker rank to avoid duplicates.
        if (worker_info := self.worker_info) is not None:
            indices = indices[worker_info.id :: worker_info.num_workers]
            logger.debug(f"indices shape after slicing by worker rank {indices.shape}")
        # Finally slice batches into micro batches for the local DP rank.
        indices = indices[:, self.dp_rank :: self.dp_world_size].reshape((-1,))
        logger.debug(f"indices shape after slicing by local DP rank {indices.shape}")
        return indices

    @property
    def total_size(self) -> int:
        """Total size."""
        return self.total_batches * self.global_batch_size

    @property
    def total_batches(self) -> int:
        """Total batches."""
        return len(self.dataset) // (self.global_batch_size)

    def _iter_batches(self) -> Iterable[dict[str, Any]]:
        return torch.utils.data.DataLoader(
            HeliosIterableDatasetWrapper(self),
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.target_device_type == "cuda" and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=False,
            timeout=0,
        )

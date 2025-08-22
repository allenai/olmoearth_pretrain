"""Geo-aware data loader."""

import logging
import math
import multiprocessing as mp
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from olmo_core.config import Config
from olmo_core.data.data_loader import DataLoaderBase
from olmo_core.data.utils import get_rng, memmap_to_write
from olmo_core.distributed.utils import (
    barrier,
    get_fs_local_rank,
    get_rank,
    get_world_size,
)
from olmo_core.utils import get_default_device
from torch.utils.data import default_collate
from upath import UPath

from helios.data.concat import HeliosConcatDataset
from helios.data.constants import IMAGE_TILE_SIZE, Modality
from helios.data.dataset import GetItemArgs, HeliosDataset, HeliosSample
from helios.data.dataloader import HeliosDataLoader, _IterableDatasetWrapper, HeliosDataLoaderConfig, iter_batched
from helios.data.utils import nearest_haversine, haversine_distance_radians

logger = logging.getLogger(__name__)


class GeoAwareDataLoader(HeliosDataLoader):
    """Geo-aware data loader."""

    def __init__(self, neighbor_percentage: float = 0.5, min_neighbor_radius: float = 1000.0, max_neighbor_radius: float = 100_000.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_neighbor_radius = min_neighbor_radius
        self.max_neighbor_radius = max_neighbor_radius
        self.neighbor_percentage = neighbor_percentage
        self.ring_batch_group_size = int(self.rank_batch_size * self.neighbor_percentage)

        # the neighbor percentage doesn't work yet beyond 0.5 or 0.25
        if self.neighbor_percentage > 0.5:
            raise ValueError("neighbor_percentage must be less than 0.5")
        if self.neighbor_percentage < 0.25:
            raise ValueError("neighbor_percentage must be greater than 0.25")

    def _iter_batches(self) -> Iterable[HeliosSample]:
        """Iterate over the dataset in batches."""
        multiprocessing_context = (
            self.multiprocessing_context if self.num_workers > 0 else None
        )
        logger.warning(f"Multiprocessing context: {multiprocessing_context}")
        return torch.utils.data.DataLoader(
            _GeoAwareIterableDatasetWrapper(self),
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.target_device_type == "cuda" and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=(
                self.persistent_workers if self.num_workers > 0 else False
            ),
            multiprocessing_context=multiprocessing_context,
            timeout=0,
        )

    # I think no way we can materialize this array even with chunking
    # So we can do 1 to many computations on the fly or only do local shard to all computations
    def get_latlons(self, indices: np.ndarray) -> np.ndarray:
        """Get the latlons for the given indices."""
        return self.dataset.latlon_distribution[indices]

    # .11 to get all ring indices does this matter
    def get_per_instance_donut_indices(self, anchor_index: np.ndarray, global_indices: np.ndarray) -> np.ndarray:
        """Get the per instance donut indices."""
        latlons = self.get_latlons(global_indices)
        anchor_latlon = self.get_latlons(anchor_index)
        # add new axis to anchor_latlon to make it 1, 2
        anchor_latlon = anchor_latlon[np.newaxis, :]
        # maybe we will need chunking maybe not
        neighbor_distances = haversine_distance_radians(anchor_latlon, latlons)
        # squueze the first axis
        neighbor_distances = neighbor_distances.squeeze(axis=0)
        # Filter the indices to the min and max neighbor radius
        donut_mask = (neighbor_distances >= self.min_neighbor_radius) & (neighbor_distances <= self.max_neighbor_radius)
        donut_indices = global_indices[donut_mask]
        return donut_indices







class _GeoAwareIterableDatasetWrapper(_IterableDatasetWrapper):
    """Iterable dataset wrapper.

    This is a modified version of olmo_core.data.data_loader._IterableDatasetWrapper
    """

    def __iter__(self) -> Iterator[HeliosSample]:
        """Iterate over the dataset."""
        global_indices = self.data_loader.get_global_indices()
        indices = self.data_loader._get_local_instance_indices(global_indices)
        # donut_indices = self.data_loader.get_local_donut_indices(indices)
        # TODO: now that this is self we may not need to pass as much in
        instance_iterator = (
            self.data_loader._get_dataset_item(int(idx), patch_size, sampled_hw_p)
            for idx, patch_size, sampled_hw_p in self._get_batch_item_params_iterator(
                indices,
                global_indices,
                self.data_loader.patch_sizes,
                self.data_loader.sampled_hw_p_list,
                self.data_loader.rank_batch_size,
            )
        )

        return (
            self.data_loader.collator(batch)
            for batch in iter_batched(
                instance_iterator,
                self.data_loader.rank_batch_size,
                self.data_loader.drop_last,
            )
        )

    def _get_batch_item_params_iterator(
        self,
        indices: np.ndarray,
        global_indices: np.ndarray,
        patch_size_list: list[int],
        hw_p_to_sample: list[int],
        rank_batch_size: int,
    ) -> Iterator[tuple[int, int, int]]:
        """Get a generator that yields a tuple of (idx, patch_size, sampled_hw_p).

        Changes patch_size and sampled_hw_p every rank_batch_size.
        """
        patch_size_array = np.array(patch_size_list)
        hw_p_to_sample_array = np.array(hw_p_to_sample)
        instances_processed = 0

        # TODO: We need to maintain state and reproducibility here
        worker_id = self.worker_info.id if self.worker_info is not None else 0
        rng = self.rngs[worker_id]
        # select an anchor index
        # select the rbs // 2  ring neighbors
        # select the rbs // 2 - 1 random points
        local_random_batch_group_size = rank_batch_size - self.data_loader.ring_batch_group_size - 1
        num_anchor_points = len(indices) // (rank_batch_size - self.data_loader.ring_batch_group_size)
        for i in range(num_anchor_points):
            if instances_processed % rank_batch_size == 0:
                patch_size = rng.choice(patch_size_array)
                max_height_width_tokens = int(IMAGE_TILE_SIZE / patch_size)
                filtered_hw_p_to_sample_array = hw_p_to_sample_array[
                    hw_p_to_sample_array <= max_height_width_tokens
                ]
                filtered_hw_p_to_sample_array = filtered_hw_p_to_sample_array[
                    filtered_hw_p_to_sample_array > 0
                ]
                sampled_hw_p = rng.choice(filtered_hw_p_to_sample_array)
            batch_indices = []
            local_idx_of_anchor = i * self.data_loader.ring_batch_group_size
            anchor_idx = indices[local_idx_of_anchor]
            batch_indices.append(anchor_idx)
            # Use the next rbs // 2 - 1 locals indexes
            start_idx = 1 + local_idx_of_anchor
            end_idx = start_idx + local_random_batch_group_size
            random_points = indices[start_idx:end_idx]
            # logger.info(f"num random points: {len(random_points)}")
            batch_indices.extend(random_points)
            # select the rbs // 2  ring neighbors randomly
            # THIS ACTUALLY SHOULD BE THE GLOBAL INDICES
            # For now we pull the ring neighbors from the global indices
            ring_neighbors = self.data_loader.get_per_instance_donut_indices(anchor_idx, global_indices)
            # Remove any ring neighbors that are already in the batch
            # TODO: do thi with np.intersect1d
            ring_neighbors = [n for n in ring_neighbors if n not in batch_indices]
            # what if there are not enough ring neighbors # Then maybe we just fill out batch
            # with global indices
            ring_neighbors = rng.choice(ring_neighbors, size=self.data_loader.ring_batch_group_size)
            # logger.info(f"num ring neighbors: {len(ring_neighbors)}")
            batch_indices.extend(ring_neighbors)
            # assert len(batch_indices) == rank_batch_size, f"Batch indices length is not equal to rank batch size got {len(batch_indices)} expected {rank_batch_size}"
            if len(batch_indices) < rank_batch_size:
                logger.warning(f"Not enough ring neighbors, filling out batch with global indices")
                # fill out the rest of the batch with global indices
                remaining_indices = np.setdiff1d(global_indices, batch_indices)
                batch_indices.extend(rng.choice(remaining_indices, size=rank_batch_size - len(batch_indices)))
            assert len(batch_indices) == rank_batch_size, f"Batch indices length is not equal to rank batch size got {len(batch_indices)} expected {rank_batch_size}"
            for idx in batch_indices:
                yield idx, int(patch_size), int(sampled_hw_p)
                # instances_processed may not be needed
                instances_processed += 1


@dataclass
class GeoAwareDataLoaderConfig(HeliosDataLoaderConfig):
    """Configuration for the HeliosDataLoader."""

    neighbor_percentage: float = 0.5
    min_neighbor_radius: float = 1000.0
    max_neighbor_radius: float = 100_000.0

    def validate(self) -> None:
        """Validate the configuration."""
        super().validate()
        if self.min_neighbor_radius < 0:
            raise ValueError("min_neighbor_radius must be greater than 0")
        if self.max_neighbor_radius < self.min_neighbor_radius:
            raise ValueError("max_neighbor_radius must be greater than min_neighbor_radius")

    def build(
        self,
        dataset: HeliosDataset,
        collator: Callable,
        dp_process_group: dist.ProcessGroup | None = None,
    ) -> "HeliosDataLoader":
        """Build the HeliosDataLoader."""
        self.validate()
        dataset.prepare()
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs["dp_world_size"] = get_world_size(dp_process_group)
        kwargs["dp_rank"] = get_rank(dp_process_group)
        kwargs["fs_local_rank"] = get_fs_local_rank()
        kwargs["target_device_type"] = self.target_device_type or get_default_device().type
        kwargs["collator"] = collator
        kwargs["work_dir"] = self.work_dir_upath # replacing the work_dir with the upath
        kwargs["dataset"] = dataset
        return GeoAwareDataLoader(**kwargs)

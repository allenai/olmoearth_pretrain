"""Helios DataLoader."""

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

logger = logging.getLogger(__name__)


class HeliosDataLoader(DataLoaderBase):
    """Helios dataloader.

    This dataloader is adapted from OLMo-core's TextDataLoaderBase and NumpyDataLoaderBase,
    incorporating their core functionality for DDP, multi-threading, and multi-processing.
    """

    def __init__(
        self,
        dataset: HeliosDataset | HeliosConcatDataset,
        work_dir: UPath,
        global_batch_size: int,
        min_patch_size: int,
        max_patch_size: int,
        sampled_hw_p_list: list[int],
        token_budget: int | None = None,
        dp_world_size: int = 1,
        dp_rank: int = 0,
        fs_local_rank: int = 0,
        seed: int = 0,
        shuffle: bool = True,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        collator: Callable = default_collate,
        target_device_type: str = "cpu",
        drop_last: bool = True,
        persistent_workers: bool = True,
        multiprocessing_context: str = "spawn",
        num_dataset_repeats_per_epoch: int = 1,
        dataset_percentage: float = 1.0,
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
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size
        if token_budget is None:
            logger.warning("No token budget provided ALL PIXELS WILL BE USED")
        self.token_budget = token_budget
        self.patch_sizes = np.arange(min_patch_size, max_patch_size + 1)
        self.sampled_hw_p_list = sampled_hw_p_list
        self.collator = collator
        self.seed = seed
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.target_device_type = target_device_type
        self.drop_last = drop_last
        self._global_indices: np.ndarray | None = None
        self.persistent_workers = persistent_workers
        self.multiprocessing_context = multiprocessing_context
        self.num_dataset_repeats_per_epoch = num_dataset_repeats_per_epoch
        self.dataset_percentage = dataset_percentage
        if self.num_workers > 0 and self.multiprocessing_context == "forkserver":
            # Overhead of loading modules on import by preloading them
            mp.set_forkserver_preload(["torch", "rasterio"])

    @property
    def total_unique_batches(self) -> int:
        """The total number of unique batches in an epoch."""
        return int(len(self.dataset) * self.dataset_percentage) // (
            self.global_batch_size
        )

    @property
    def total_unique_size(self) -> int:
        """The total number of unique instances in an epoch."""
        return self.total_unique_batches * self.global_batch_size

    @property
    def total_batches(self) -> int:
        """The total number of batches in an epoch."""
        return self.total_unique_batches * self.num_dataset_repeats_per_epoch

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
        indices = np.arange(len(self.dataset), dtype=np.uint32)
        if self.dataset_percentage < 1.0:
            # Dataset subset selected should be deterministic across random seeds and restarts
            data_percentage_rng = get_rng(self.seed)
            indices = data_percentage_rng.choice(
                indices,
                size=int(len(indices) * self.dataset_percentage),
                replace=False,
            )
            logger.info(
                f"Picked {len(indices)} samples from {len(self.dataset)} samples"
            )
        indices_list = []
        for _ in range(self.num_dataset_repeats_per_epoch):
            if rng is not None:
                rng.shuffle(indices)
            # Remove tail of data to make it evenly divisible
            cropped_indices = indices[: self.total_unique_size]
            indices_list.append(cropped_indices)
        indices = np.concatenate(indices_list)
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
                f"Missing global indices file {self._global_indices_file}, did you forget to call 'reshuffle()'?"
            )
        return np.memmap(self._global_indices_file, mode="r", dtype=np.uint32)

    def _iter_batches(self) -> Iterable[HeliosSample]:
        """Iterate over the dataset in batches."""
        return torch.utils.data.DataLoader(
            _IterableDatasetWrapper(self),
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.target_device_type == "cuda" and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=(
                self.persistent_workers if self.num_workers > 0 else False
            ),
            multiprocessing_context=(
                self.multiprocessing_context if self.num_workers > 0 else None
            ),
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

        if self.batches_processed > 0:  # type: ignore
            indices = indices[self.batches_processed :]  # type: ignore

        # Slice batches by data loader worker rank to avoid duplicates.
        if (worker_info := self.worker_info) is not None:
            indices = indices[worker_info.id :: worker_info.num_workers]

        # Finally step batches into micro batches for the local DP rank.
        indices = indices[:, self.dp_rank :: self.dp_world_size].reshape((-1,))
        return indices

    def _get_dataset_item(
        self, idx: int, patch_size: int, sampled_hw_p: int
    ) -> tuple[int, HeliosSample]:
        """Get a dataset item."""
        args = GetItemArgs(
            idx=idx,
            patch_size=patch_size,
            sampled_hw_p=sampled_hw_p,
            token_budget=self.token_budget,
        )
        item = self.dataset[args]
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
            raise ValueError(
                "Restoring data loading state from a different dataset is not supported! (fingerprint doesn't match)"
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

    def _get_mock_sample(self, rng: np.random.Generator) -> HeliosSample:
        output_dict = {}
        standard_hw = 64
        if Modality.SENTINEL2_L2A.name in self.dataset.training_modalities:
            mock_sentinel2_l2a = rng.random(
                (standard_hw, standard_hw, 12, 12), dtype=np.float32
            )
            output_dict["sentinel2_l2a"] = mock_sentinel2_l2a
        if Modality.NAIP_10.name in self.dataset.training_modalities:
            mock_naip_10 = rng.random((1024, 1024, 1, 4), dtype=np.float32)
            output_dict["naip_10"] = mock_naip_10
        if Modality.SENTINEL1.name in self.dataset.training_modalities:
            mock_sentinel1 = rng.random(
                (standard_hw, standard_hw, 12, 2), dtype=np.float32
            )
            output_dict[Modality.SENTINEL1.name] = mock_sentinel1
        if Modality.WORLDCOVER.name in self.dataset.training_modalities:
            mock_worldcover = rng.random(
                (standard_hw, standard_hw, 1, 1), dtype=np.float32
            )
            output_dict["worldcover"] = mock_worldcover
        if Modality.LATLON.name in self.dataset.training_modalities:
            mock_latlon = rng.random((2,), dtype=np.float32)
            output_dict["latlon"] = mock_latlon
        if Modality.OPENSTREETMAP_RASTER.name in self.dataset.training_modalities:
            mock_openstreetmap_raster = rng.random(
                (standard_hw, standard_hw, 1, 30), dtype=np.float32
            )
            output_dict["openstreetmap_raster"] = mock_openstreetmap_raster
        if Modality.SRTM.name in self.dataset.training_modalities:
            mock_srtm = rng.random((standard_hw, standard_hw, 1, 1), dtype=np.float32)
            output_dict["srtm"] = mock_srtm
        if Modality.LANDSAT.name in self.dataset.training_modalities:
            mock_landsat = rng.random(
                (standard_hw, standard_hw, 12, Modality.LANDSAT.num_bands),
                dtype=np.float32,
            )
            output_dict["landsat"] = mock_landsat
        if Modality.GSE.name in self.dataset.training_modalities:
            mock_gse = rng.random(
                (standard_hw, standard_hw, 1, Modality.GSE.num_bands), dtype=np.float32
            )
            output_dict["gse"] = mock_gse
        if Modality.CDL.name in self.dataset.training_modalities:
            mock_cdl = rng.random(
                (standard_hw, standard_hw, 1, Modality.CDL.num_bands), dtype=np.float32
            )
            output_dict["cdl"] = mock_cdl
        if Modality.WORLDPOP.name in self.dataset.training_modalities:
            mock_worldpop = rng.random(
                (standard_hw, standard_hw, 1, Modality.WORLDPOP.num_bands),
                dtype=np.float32,
            )
            output_dict["worldpop"] = mock_worldpop
        if Modality.WRI_CANOPY_HEIGHT_MAP.name in self.dataset.training_modalities:
            mock_wri_canopy_height_map = rng.random(
                (standard_hw, standard_hw, 1, Modality.WRI_CANOPY_HEIGHT_MAP.num_bands),
                dtype=np.float32,
            )
            output_dict["wri_canopy_height_map"] = mock_wri_canopy_height_map
        if Modality.ERA5_10.name in self.dataset.training_modalities:
            mock_era5_10 = rng.random(
                (12, Modality.ERA5_10.num_bands), dtype=np.float32
            )
            output_dict["era5_10"] = mock_era5_10

        days = rng.integers(0, 25, (12, 1))
        months = rng.integers(0, 12, (12, 1))
        years = rng.integers(2018, 2020, (12, 1))
        timestamps = np.concatenate([days, months, years], axis=1)  # shape: (12, 3)

        output_dict["timestamps"] = timestamps
        return HeliosSample(**output_dict)

    def get_mock_batch(self) -> HeliosSample:
        """Get a mock batch, for dry-run of forward and backward pass."""
        logger.info("Getting mock batch NOT FROM DATASET")
        logger.info(f"Training modalities: {self.dataset.training_modalities}")
        rng = get_rng(42)
        batch_size = self.global_batch_size // self.dp_world_size
        patch_size = 1
        collated_sample = self.collator(
            [
                (
                    patch_size,
                    self._get_mock_sample(rng).subset_default(
                        patch_size,
                        max_tokens_per_instance=1500,
                        sampled_hw_p=6,
                        current_length=12,
                    ),
                )
                for num in range(batch_size)
            ]
        )
        return collated_sample

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
        self.reshuffle(epoch=epoch)
        batch_start = int(self.get_global_indices()[step_in_epoch])
        batch_end = batch_start + self.global_batch_size
        sample_indices = np.arange(batch_start, batch_end)
        return sample_indices


def iter_batched(
    iterable: Iterable[tuple[int, HeliosSample]],
    batch_size: int,
    drop_last: bool = True,
) -> Iterable[tuple[tuple[int, HeliosSample], ...]]:
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
    batch: list[tuple[int, HeliosSample]] = []
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
        workers = data_loader.num_workers or 1
        self.rngs = [
            get_rng(
                data_loader.seed + data_loader.epoch + data_loader.dp_rank * workers + i
            )
            for i in range(workers)
        ]

    def _get_batch_item_params_iterator(
        self,
        indices: np.ndarray,
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

        for idx in indices:
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
            yield idx, int(patch_size), int(sampled_hw_p)
            instances_processed += 1

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
        indices = self.data_loader._get_local_instance_indices(global_indices)
        instance_iterator = (
            self.data_loader._get_dataset_item(int(idx), patch_size, sampled_hw_p)
            for idx, patch_size, sampled_hw_p in self._get_batch_item_params_iterator(
                indices,
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


@dataclass
class HeliosDataLoaderConfig(Config):
    """Configuration for the HeliosDataLoader."""

    work_dir: str
    global_batch_size: int
    min_patch_size: int
    max_patch_size: int
    sampled_hw_p_list: list[int]
    seed: int
    token_budget: int | None = None  # No subsetting if None
    shuffle: bool = True
    num_workers: int = 0
    prefetch_factor: int | None = None
    target_device_type: str | None = None
    drop_last: bool = True
    num_dataset_repeats_per_epoch: int = 1
    dataset_percentage: float = 1.0

    def validate(self) -> None:
        """Validate the configuration."""
        if self.work_dir is None:
            raise ValueError("Work directory is not set")
        if self.min_patch_size > self.max_patch_size:
            raise ValueError("min_patch_size must be less than max_patch_size")

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
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            target_device_type=self.target_device_type or get_default_device().type,
            collator=collator,
            drop_last=self.drop_last,
            min_patch_size=self.min_patch_size,
            max_patch_size=self.max_patch_size,
            sampled_hw_p_list=self.sampled_hw_p_list,
            token_budget=self.token_budget,
            num_dataset_repeats_per_epoch=self.num_dataset_repeats_per_epoch,
            dataset_percentage=self.dataset_percentage,
        )

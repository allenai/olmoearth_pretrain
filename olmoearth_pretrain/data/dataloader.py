"""OlmoEarth Pretrain DataLoader."""

import functools
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

from olmoearth_pretrain._compat import deprecated_class_alias as _deprecated_class_alias
from olmoearth_pretrain.config import Config
from olmoearth_pretrain.data.collate import (
    collate_double_masked_batched,
    collate_single_masked_batched,
)
from olmoearth_pretrain.data.concat import OlmoEarthConcatDataset
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataset import (
    GetItemArgs,
    OlmoEarthDataset,
    OlmoEarthSample,
    compute_bandset_rates,
    subset_sample_default,
)
from olmoearth_pretrain.data.transform import Transform, TransformConfig
from olmoearth_pretrain.nn.tokenization import TokenizationConfig
from olmoearth_pretrain.train.masking import MaskingConfig, MaskingStrategy

logger = logging.getLogger(__name__)


class OlmoEarthDataLoader(DataLoaderBase):
    """OlmoEarth Pretrain dataloader.

    This dataloader is adapted from OLMo-core's TextDataLoaderBase and NumpyDataLoaderBase,
    incorporating their core functionality for DDP, multi-threading, and multi-processing.
    """

    def __init__(
        self,
        dataset: OlmoEarthDataset | OlmoEarthConcatDataset,
        work_dir: UPath,
        global_batch_size: int,
        min_patch_size: int,
        max_patch_size: int,
        sampled_hw_p_list: list[int],
        token_budget: int | None = None,
        patch_size_probs: list[float] | None = None,
        time_priority_prob: float = 0.0,
        temporal_bias: float = 0.0,
        min_tokens_per_instance: int = 0,
        max_timesteps: int = 12,
        tile_size: int = 128,
        exclude_only_decode_from_budget: bool = False,
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
        # Dataloader-side masking
        transform: Transform | None = None,
        masking_strategy: MaskingStrategy | None = None,
        masking_strategy_b: MaskingStrategy | None = None,
        num_masked_views: int = 1,
        tokenization_config: TokenizationConfig | None = None,
        token_budget_excluded_modalities: list[str] | None = None,
    ):
        """Initialize the OlmoEarthDataLoader.

        Args:
            dataset: The dataset to load from.
            work_dir: The working directory for storing indices.
            global_batch_size: The global batch size across all workers.
            min_patch_size: Minimum patch size for training.
            max_patch_size: Maximum patch size for training.
            sampled_hw_p_list: List of possible height/width in patches to sample.
            token_budget: Optional token budget per instance.
            patch_size_probs: Optional per-patch-size sampling probabilities aligned
                with ``range(min_patch_size, max_patch_size + 1)``. If None, patch
                sizes are sampled uniformly (the historical behaviour). Use this to
                oversample small patch sizes for models deployed at that resolution.
            time_priority_prob: Probability that a batch samples its number of
                timesteps first (biased toward the full sequence) and then a spatial
                grid that fits, rather than sampling the grid first. 0.0 reproduces
                the historical space-first behaviour; >0 decorrelates grid size from
                sequence length so that large-grid x full-year shapes occur.
            temporal_bias: Skews the timestep draw toward the maximum of its feasible
                window; timesteps are sampled with weight ``t ** temporal_bias``. 0.0
                is uniform (the historical behaviour); larger values favour fuller
                sequences, restoring full-season exposure that uniform sampling
                dilutes.
            min_tokens_per_instance: Minimum token count a sampled shape must cost.
                Shapes below the floor are excluded, so tiny grids are forced to pair
                with long sequences (and vice versa) instead of collapsing to the
                ``hw=1, t=1`` corner. 0 disables the floor.
            max_timesteps: Maximum number of timesteps a sample can contribute.
            tile_size: Spatial extent (in base-resolution pixels) of a training tile.
                Used to bound the sampled grid so ``sampled_hw_p * patch_size`` fits.
            exclude_only_decode_from_budget: If True, modalities the masking strategy
                marks decode-only are not counted against the token budget (they are
                never encoded), freeing budget for more timesteps.
            dp_world_size: Data parallel world size.
            dp_rank: Data parallel rank.
            fs_local_rank: File system local rank.
            seed: Random seed.
            shuffle: Whether to shuffle the data.
            num_workers: Number of dataloader workers.
            prefetch_factor: Prefetch factor for dataloader.
            collator: Collation function.
            target_device_type: Target device type ("cpu" or "cuda").
            drop_last: Whether to drop the last incomplete batch.
            persistent_workers: Whether to keep workers alive between epochs.
            multiprocessing_context: Multiprocessing context ("spawn" or "forkserver").
            num_dataset_repeats_per_epoch: Number of times to repeat the dataset per epoch.
            transform: Optional transform to apply in the dataloader workers.
            masking_strategy: Masking strategy to apply in the dataloader workers.
            masking_strategy_b: Optional second masking strategy for Galileo-style training.
            num_masked_views: Number of masked views to return (1=single, 2=double).
            tokenization_config: Optional tokenization config for custom band groupings.
            token_budget_excluded_modalities: Loaded modalities that are not tokenized
                by the model and should not consume the token budget.
        """
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
        self.time_priority_prob = time_priority_prob
        self.temporal_bias = temporal_bias
        self.min_tokens_per_instance = min_tokens_per_instance
        self.max_timesteps = max_timesteps
        self.tile_size = tile_size
        if patch_size_probs is not None:
            if len(patch_size_probs) != len(self.patch_sizes):
                raise ValueError(
                    f"patch_size_probs must have {len(self.patch_sizes)} entries "
                    f"(one per patch size in [{min_patch_size}, {max_patch_size}]), "
                    f"got {len(patch_size_probs)}"
                )
            probs = np.asarray(patch_size_probs, dtype=np.float64)
            if not np.isclose(probs.sum(), 1.0):
                raise ValueError(f"patch_size_probs must sum to 1.0, got {probs.sum()}")
            self.patch_size_probs: np.ndarray | None = probs
        else:
            self.patch_size_probs = None
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

        # Dataloader-side masking configuration
        self.transform = transform
        self.masking_strategy = masking_strategy
        self.masking_strategy_b = masking_strategy_b
        self.num_masked_views = num_masked_views
        self.tokenization_config = tokenization_config
        self.token_budget_excluded_modalities = frozenset(
            token_budget_excluded_modalities or []
        )

        # Validate configuration
        if masking_strategy is None:
            raise ValueError("masking_strategy must be provided")
        if num_masked_views not in (1, 2):
            raise ValueError(f"num_masked_views must be 1 or 2, got {num_masked_views}")

        if self.num_workers > 0 and self.multiprocessing_context == "forkserver":
            # Overhead of loading modules on import by preloading them
            mp.set_forkserver_preload(["torch", "rasterio"])

        # Modalities kept out of the encoder token budget (decode-only targets are
        # never encoded, so they should not consume budget meant for the encoder).
        # Starts from the explicitly configured exclusions (e.g. open-set label
        # layers) and optionally adds the masking strategy's decode-only maps.
        self.budget_exclude_modalities: frozenset[str] = (
            self.token_budget_excluded_modalities
        )
        if exclude_only_decode_from_budget:
            self.budget_exclude_modalities |= frozenset(
                getattr(self.masking_strategy, "only_decode_modalities", []) or []
            )

        # Precompute per-instance band-set token rates so the shape sampler can
        # invert the budget without a concrete sample in hand. Uses the full
        # training modality set (a superset of any single sample), so the derived
        # max_t is conservative and always fits the per-sample budget downstream.
        training_modalities = getattr(self.dataset, "training_modalities", None)
        if training_modalities is not None:
            (
                self._st_bandsets,
                self._so_bandsets,
                self._static_bandsets,
                self._time_bandsets,
            ) = compute_bandset_rates(
                training_modalities,
                self.tokenization_config,
                exclude_modalities=self.budget_exclude_modalities,
            )
        else:
            # No modality metadata (e.g. mock datasets): the sampler falls back to
            # budget-unaware timestep sampling (max_t = max_timesteps).
            self._st_bandsets = self._so_bandsets = 0
            self._static_bandsets = self._time_bandsets = 0

    @property
    def total_unique_batches(self) -> int:
        """The total number of unique batches in an epoch."""
        return len(self.dataset) // (self.global_batch_size)

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
        indices_list = []
        for _ in range(self.num_dataset_repeats_per_epoch):
            indices = np.arange(len(self.dataset), dtype=np.uint32)
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

    def reshuffle(self, epoch: int | None = None, in_memory: bool = False) -> None:
        """Reshuffle the data."""
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

    def _iter_batches(self) -> Iterable[OlmoEarthSample]:
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
        self, idx: int, patch_size: int, sampled_hw_p: int, target_t: int | None = None
    ) -> tuple[int, OlmoEarthSample]:
        """Get a dataset item."""
        args = GetItemArgs(
            idx=idx,
            patch_size=patch_size,
            sampled_hw_p=sampled_hw_p,
            token_budget=self.token_budget,
            tokenization_config=self.tokenization_config,
            target_t=target_t,
            budget_exclude_modalities=self.budget_exclude_modalities,
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
            logger.warning(
                "Restoring state from a different dataset! If this is not expected, please check the dataset fingerprint(fingerprint doesn't match)"
                f"old fingerprint: {state_dict['dataset_fingerprint']}, new fingerprint: {self.dataset.fingerprint}"
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

    def _get_mock_sample(self, rng: np.random.Generator) -> OlmoEarthSample:
        output_dict = {}
        standard_hw = 64
        if Modality.SENTINEL2_L2A.name in self.dataset.training_modalities:
            mock_sentinel2_l2a = rng.random(
                (standard_hw, standard_hw, 12, 12), dtype=np.float32
            )
            output_dict[Modality.SENTINEL2_L2A.name] = mock_sentinel2_l2a
        if Modality.NAIP_10.name in self.dataset.training_modalities:
            mock_naip_10 = rng.random((1024, 1024, 1, 4), dtype=np.float32)
            output_dict[Modality.NAIP_10.name] = mock_naip_10
        if Modality.SENTINEL1.name in self.dataset.training_modalities:
            mock_sentinel1 = rng.random(
                (standard_hw, standard_hw, 12, 2), dtype=np.float32
            )
            output_dict[Modality.SENTINEL1.name] = mock_sentinel1
        if Modality.WORLDCOVER.name in self.dataset.training_modalities:
            mock_worldcover = rng.random(
                (standard_hw, standard_hw, 1, 1), dtype=np.float32
            )
            output_dict[Modality.WORLDCOVER.name] = mock_worldcover
        if Modality.WORLDCOVER_ONEHOT.name in self.dataset.training_modalities:
            num_classes = Modality.WORLDCOVER_ONEHOT.num_bands
            class_ids = rng.integers(0, num_classes, size=(standard_hw, standard_hw, 1))
            mock_worldcover_onehot = np.eye(num_classes, dtype=np.float32)[class_ids]
            output_dict[Modality.WORLDCOVER_ONEHOT.name] = mock_worldcover_onehot
        if Modality.LATLON.name in self.dataset.training_modalities:
            mock_latlon = rng.random((2,), dtype=np.float32)
            output_dict[Modality.LATLON.name] = mock_latlon
        if Modality.OPENSTREETMAP_RASTER.name in self.dataset.training_modalities:
            mock_openstreetmap_raster = rng.random(
                (standard_hw, standard_hw, 1, 30), dtype=np.float32
            )
            output_dict[Modality.OPENSTREETMAP_RASTER.name] = mock_openstreetmap_raster
        if Modality.SRTM.name in self.dataset.training_modalities:
            mock_srtm = rng.random((standard_hw, standard_hw, 1, 1), dtype=np.float32)
            output_dict["srtm"] = mock_srtm
        if Modality.LANDSAT.name in self.dataset.training_modalities:
            mock_landsat = rng.random(
                (standard_hw, standard_hw, 12, Modality.LANDSAT.num_bands),
                dtype=np.float32,
            )
            output_dict[Modality.LANDSAT.name] = mock_landsat
        if Modality.GSE.name in self.dataset.training_modalities:
            mock_gse = rng.random(
                (standard_hw, standard_hw, 1, Modality.GSE.num_bands), dtype=np.float32
            )
            output_dict[Modality.GSE.name] = mock_gse
        if Modality.TESSERA.name in self.dataset.training_modalities:
            mock_tessera = rng.random(
                (standard_hw, standard_hw, 1, Modality.TESSERA.num_bands),
                dtype=np.float32,
            )
            output_dict[Modality.TESSERA.name] = mock_tessera
        if Modality.CDL.name in self.dataset.training_modalities:
            mock_cdl = rng.random(
                (standard_hw, standard_hw, 1, Modality.CDL.num_bands), dtype=np.float32
            )
            output_dict[Modality.CDL.name] = mock_cdl
        if Modality.WORLDPOP.name in self.dataset.training_modalities:
            mock_worldpop = rng.random(
                (standard_hw, standard_hw, 1, Modality.WORLDPOP.num_bands),
                dtype=np.float32,
            )
            output_dict[Modality.WORLDPOP.name] = mock_worldpop
        if Modality.WRI_CANOPY_HEIGHT_MAP.name in self.dataset.training_modalities:
            mock_wri_canopy_height_map = rng.random(
                (standard_hw, standard_hw, 1, Modality.WRI_CANOPY_HEIGHT_MAP.num_bands),
                dtype=np.float32,
            )
            output_dict[Modality.WRI_CANOPY_HEIGHT_MAP.name] = (
                mock_wri_canopy_height_map
            )
        if Modality.ERA5_10.name in self.dataset.training_modalities:
            mock_era5_10 = rng.random(
                (12, Modality.ERA5_10.num_bands), dtype=np.float32
            )
            output_dict[Modality.ERA5_10.name] = mock_era5_10
        if Modality.EUROCROPS.name in self.dataset.training_modalities:
            mock_eurocrops = rng.random(
                (standard_hw, standard_hw, 1, Modality.EUROCROPS.num_bands),
                dtype=np.float32,
            )
            output_dict[Modality.EUROCROPS.name] = mock_eurocrops

        days = rng.integers(0, 25, (12, 1))
        months = rng.integers(0, 12, (12, 1))
        years = rng.integers(2018, 2020, (12, 1))
        timestamps = np.concatenate([days, months, years], axis=1)  # shape: (12, 3)

        output_dict["timestamps"] = timestamps
        return OlmoEarthSample(**output_dict)

    def get_mock_batch(self) -> Any:
        """Get a mock batch, for dry-run of forward and backward pass.

        Returns the appropriate batch format based on num_masked_views:
        - 1: (patch_size, MaskedOlmoEarthSample) - single masked view
        - 2: (patch_size, MaskedOlmoEarthSample, MaskedOlmoEarthSample) - double masked
        """
        logger.info("Getting mock batch NOT FROM DATASET")
        logger.info(f"Training modalities: {self.dataset.training_modalities}")
        logger.info(f"num_masked_views: {self.num_masked_views}")
        rng = get_rng(42)
        batch_size = self.global_batch_size // self.dp_world_size
        patch_size = 1

        # Generate mock samples
        mock_samples = [
            subset_sample_default(
                self._get_mock_sample(rng),
                patch_size=patch_size,
                max_tokens_per_instance=1500,
                sampled_hw_p=6,
                current_length=12,
            )
            for _ in range(batch_size)
        ]

        # Pass raw samples to the collator - the batched collators handle
        # transform + masking internally when num_masked_views > 0
        collated_sample = self.collator(
            [(patch_size, sample) for sample in mock_samples]
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
    iterable: Iterable[tuple[int, OlmoEarthSample]],
    batch_size: int,
    drop_last: bool = True,
) -> Iterable[tuple[tuple[int, OlmoEarthSample], ...]]:
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
    batch: list[tuple[int, OlmoEarthSample]] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield tuple(batch)
            batch.clear()

    # If there's a partial batch left over, yield it if `drop_last` is False
    if not drop_last and batch:
        yield tuple(batch)


class _IterableDatasetWrapper(torch.utils.data.IterableDataset[OlmoEarthSample]):
    """Iterable dataset wrapper.

    This is a modified version of olmo_core.data.data_loader._IterableDatasetWrapper
    """

    def __init__(self, data_loader: OlmoEarthDataLoader):
        """Initialize the IterableDatasetWrapper."""
        self.data_loader = data_loader
        workers = data_loader.num_workers or 1
        self.rngs = [
            get_rng(
                data_loader.seed + data_loader.epoch + data_loader.dp_rank * workers + i
            )
            for i in range(workers)
        ]
        # Dataloader-side masking configuration
        self.transform = data_loader.transform
        self.masking_strategy = data_loader.masking_strategy
        self.masking_strategy_b = data_loader.masking_strategy_b
        self.num_masked_views = data_loader.num_masked_views

    def _get_batch_item_params_iterator(
        self,
        indices: np.ndarray,
        patch_size_list: list[int],
        hw_p_to_sample: list[int],
        rank_batch_size: int,
    ) -> Iterator[tuple[int, int, int, int]]:
        """Yield ``(idx, patch_size, sampled_hw_p, target_t)`` per instance.

        The shape ``(patch_size, sampled_hw_p, target_t)`` is resampled every
        ``rank_batch_size`` instances.

        Historically ``target_t`` was derived downstream as the maximum number of
        timesteps that fit the token budget for the sampled grid, which perfectly
        anti-correlates grid size and sequence length. Here ``target_t`` is sampled
        as an independent axis so that large-grid x full-year shapes occur:

        - With probability ``time_priority_prob`` the number of timesteps is sampled
          first (biased toward the full sequence via ``temporal_bias``) and then a
          grid that fits it.
        - Otherwise a grid is sampled first (uniformly over the feasible sizes,
          which may exceed the old <=12 range) and then ``target_t`` over what its
          budget allows, again biased by ``temporal_bias``.

        Two floors keep degenerate shapes out. ``min_tokens_per_instance`` requires
        every shape to cost at least that many tokens, so tiny grids are forced to
        pair with long sequences (and vice versa) rather than collapsing to the
        ``hw=1, t=1`` corner. ``temporal_bias`` skews the timestep draw toward the
        maximum (0 = uniform; larger = fuller sequences), restoring the full-season
        exposure that pure uniform sampling dilutes.

        The budget remains a hard cap: ``subset_sample_*`` clamps to the per-sample
        budget, so the sampler's conservative estimate can never overshoot.
        """
        dl = self.data_loader
        patch_size_array = np.array(patch_size_list)
        hw_p_to_sample_array = np.array(hw_p_to_sample)
        instances_processed = 0

        budget = dl.token_budget
        max_t_data = dl.max_timesteps
        st_bs = dl._st_bandsets
        so_bs = dl._so_bandsets
        static_bs = dl._static_bandsets
        time_bs = dl._time_bandsets
        time_priority_prob = dl.time_priority_prob
        temporal_bias = dl.temporal_bias
        min_tokens = dl.min_tokens_per_instance
        ps_probs = dl.patch_size_probs

        def max_t_for(hw: int) -> int:
            """Largest number of timesteps that fits the budget for this grid."""
            if budget is None:
                return max_t_data
            fixed = so_bs * hw * hw + static_bs
            per_t = st_bs * hw * hw + time_bs
            if per_t <= 0:  # no time-varying modalities
                return max_t_data if fixed <= budget else 0
            remaining = budget - fixed
            if remaining < per_t:
                return 0
            return int(min(max_t_data, remaining // per_t))

        def min_t_for(hw: int) -> int:
            """Fewest timesteps whose (grid, t) shape clears the token floor."""
            if min_tokens <= 0:
                return 1
            fixed = so_bs * hw * hw + static_bs
            per_t = st_bs * hw * hw + time_bs
            if fixed >= min_tokens:  # floor already met by the fixed spatial tokens
                return 1
            if per_t <= 0:  # no time-varying modalities and floor unmet
                return max_t_data + 1  # unsatisfiable -> grid dropped below
            return int(max(1, -(-(min_tokens - fixed) // per_t)))  # ceil division

        def max_hw_for(t: int, grid_cap: int) -> int:
            """Largest grid side whose (grid, t) shape fits the budget."""
            if budget is None:
                return grid_cap
            denom = so_bs + st_bs * t
            if denom <= 0:  # no spatial modalities
                return grid_cap
            remaining = budget - static_bs - time_bs * t
            if remaining < denom:
                return 0
            return max(1, min(grid_cap, int(math.isqrt(remaining // denom))))

        def sample_t(lo: int, hi: int) -> int:
            """Sample a timestep in [lo, hi], biased toward hi by temporal_bias."""
            if hi <= lo:
                return lo
            ts = np.arange(lo, hi + 1)
            if temporal_bias == 0:
                return int(rng.choice(ts))
            weights = ts.astype(np.float64) ** temporal_bias
            weights /= weights.sum()
            return int(rng.choice(ts, p=weights))

        # TODO: We need to maintain state and reproducibility here
        worker_id = self.worker_info.id if self.worker_info is not None else 0
        rng = self.rngs[worker_id]

        for idx in indices:
            if instances_processed % rank_batch_size == 0:
                patch_size = int(rng.choice(patch_size_array, p=ps_probs))
                # Bound the grid so sampled_hw_p * patch_size fits the tile extent.
                grid_cap = dl.tile_size // patch_size
                grids = hw_p_to_sample_array[
                    (hw_p_to_sample_array > 0) & (hw_p_to_sample_array <= grid_cap)
                ]
                # A grid is a candidate only if some timestep count satisfies both
                # the budget ceiling and the token floor: min_t_for <= max_t_for.
                windows = {
                    int(h): (min_t_for(int(h)), max_t_for(int(h))) for h in grids
                }
                candidates = np.array(
                    [h for h in grids if windows[int(h)][0] <= windows[int(h)][1]],
                    dtype=grids.dtype,
                )
                if len(candidates) == 0:
                    raise ValueError(
                        f"No feasible sampled_hw_p for patch_size={patch_size}, "
                        f"token_budget={budget}, min_tokens={min_tokens}, "
                        f"tile_size={dl.tile_size}. Check sampled_hw_p_list, "
                        f"token_budget and min_tokens_per_instance."
                    )

                if rng.random() < time_priority_prob:
                    # Time-first: draw t (biased to the full sequence) among the
                    # timesteps some candidate grid supports, then a grid for it.
                    achievable = [
                        t
                        for t in range(1, max_t_data + 1)
                        if any(lo <= t <= hi for lo, hi in windows.values())
                    ]
                    target_t = sample_t(achievable[0], achievable[-1])
                    while not any(
                        windows[int(h)][0] <= target_t <= windows[int(h)][1]
                        for h in candidates
                    ):
                        # target_t landed in a gap between per-grid windows; redraw.
                        target_t = int(rng.choice(achievable))
                    feasible = np.array(
                        [
                            h
                            for h in candidates
                            if windows[int(h)][0] <= target_t <= windows[int(h)][1]
                        ],
                        dtype=candidates.dtype,
                    )
                    sampled_hw_p = int(rng.choice(feasible))
                else:
                    # Space-first: pick a (possibly large) grid, then t within its
                    # feasible [min_t, max_t] window.
                    sampled_hw_p = int(rng.choice(candidates))
                    lo, hi = windows[sampled_hw_p]
                    target_t = sample_t(lo, hi)
            yield idx, int(patch_size), int(sampled_hw_p), int(target_t)
            instances_processed += 1

    @property
    def dataset(self) -> OlmoEarthDataset:
        """Get the dataset."""
        return self.data_loader.dataset

    @property
    def worker_info(self):  # type: ignore
        """Get worker info."""
        return torch.utils.data.get_worker_info()

    def __iter__(self) -> Iterator[Any]:
        """Iterate over the dataset.

        Yields batches in one of two formats depending on num_masked_views:
        - 1: (patch_size, MaskedOlmoEarthSample) - single masked view
        - 2: (patch_size, MaskedOlmoEarthSample, MaskedOlmoEarthSample) - double masked views

        Transform and masking are applied in the batched collator for better vectorization.
        """
        global_indices = self.data_loader.get_global_indices()
        indices = self.data_loader._get_local_instance_indices(global_indices)

        # Create iterator that fetches samples from the dataset
        instance_iterator = (
            self.data_loader._get_dataset_item(
                int(idx), patch_size, sampled_hw_p, target_t
            )
            for idx, patch_size, sampled_hw_p, target_t in (
                self._get_batch_item_params_iterator(
                    indices,
                    self.data_loader.patch_sizes,
                    self.data_loader.sampled_hw_p_list,
                    self.data_loader.rank_batch_size,
                )
            )
        )

        return (
            self.data_loader.collator(batch)  # type: ignore[arg-type]
            for batch in iter_batched(
                instance_iterator,  # type: ignore[arg-type]
                self.data_loader.rank_batch_size,
                self.data_loader.drop_last,
            )
        )


@dataclass
class OlmoEarthDataLoaderConfig(Config):
    """Configuration for the OlmoEarthDataLoader."""

    work_dir: str
    global_batch_size: int
    min_patch_size: int
    max_patch_size: int
    sampled_hw_p_list: list[int]
    seed: int
    token_budget: int | None = None  # No subsetting if None
    patch_size_probs: list[float] | None = None
    time_priority_prob: float = 0.0
    temporal_bias: float = 0.0
    min_tokens_per_instance: int = 0
    max_timesteps: int = 12
    tile_size: int = 128
    exclude_only_decode_from_budget: bool = False
    shuffle: bool = True
    num_workers: int = 0
    prefetch_factor: int | None = None
    target_device_type: str | None = None
    drop_last: bool = True
    num_dataset_repeats_per_epoch: int = 1
    # New fields for dataloader-side masking
    transform_config: TransformConfig | None = None
    masking_config: MaskingConfig | None = None
    masking_config_b: MaskingConfig | None = None
    num_masked_views: int = 1  # 1 = single, 2 = double
    tokenization_config: TokenizationConfig | None = None
    token_budget_excluded_modalities: list[str] | None = None

    def validate(self) -> None:
        """Validate the configuration."""
        if self.work_dir is None:
            raise ValueError("Work directory is not set")
        if self.min_patch_size > self.max_patch_size:
            raise ValueError("min_patch_size must be less than max_patch_size")
        if self.masking_config is None:
            raise ValueError("masking_config must be provided")
        if self.num_masked_views not in (1, 2):
            raise ValueError(
                f"num_masked_views must be 1 or 2, got {self.num_masked_views}"
            )

    @property
    def work_dir_upath(self) -> UPath:
        """Get the work directory."""
        return UPath(self.work_dir)

    def build(
        self,
        dataset: OlmoEarthDataset,
        dp_process_group: dist.ProcessGroup | None = None,
    ) -> "OlmoEarthDataLoader":
        """Build the OlmoEarthDataLoader."""
        self.validate()
        dataset.prepare()

        # Build transform and masking strategies
        transform = (
            self.transform_config.build() if self.transform_config is not None else None
        )
        # masking_config is required (validated above)
        assert self.masking_config is not None
        masking_strategy = self.masking_config.build()
        masking_strategy_b = (
            self.masking_config_b.build() if self.masking_config_b is not None else None
        )

        # Select appropriate collator based on num_masked_views
        # Use batched collators that apply transform + masking to the entire batch
        # at once for better vectorization
        collator: Callable
        if self.num_masked_views == 1:
            collator = functools.partial(
                collate_single_masked_batched,
                transform=transform,
                masking_strategy=masking_strategy,
            )
        else:  # num_masked_views == 2
            collator = functools.partial(
                collate_double_masked_batched,
                transform=transform,
                masking_strategy=masking_strategy,
                masking_strategy_b=masking_strategy_b,
            )

        return OlmoEarthDataLoader(
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
            patch_size_probs=self.patch_size_probs,
            time_priority_prob=self.time_priority_prob,
            temporal_bias=self.temporal_bias,
            min_tokens_per_instance=self.min_tokens_per_instance,
            max_timesteps=self.max_timesteps,
            tile_size=self.tile_size,
            exclude_only_decode_from_budget=self.exclude_only_decode_from_budget,
            num_dataset_repeats_per_epoch=self.num_dataset_repeats_per_epoch,
            transform=transform,
            masking_strategy=masking_strategy,
            masking_strategy_b=masking_strategy_b,
            num_masked_views=self.num_masked_views,
            tokenization_config=self.tokenization_config,
            token_budget_excluded_modalities=self.token_budget_excluded_modalities,
        )


HeliosDataLoader = _deprecated_class_alias(
    OlmoEarthDataLoader, "helios.data.dataloader.HeliosDataLoader"
)
HeliosDataLoaderConfig = _deprecated_class_alias(
    OlmoEarthDataLoaderConfig, "helios.data.dataloader.HeliosDataLoaderConfig"
)

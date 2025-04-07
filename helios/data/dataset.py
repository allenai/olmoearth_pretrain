"""Dataset module for helios."""

import hashlib
import logging
import multiprocessing as mp
import os
import random
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from math import floor
from pathlib import Path
from typing import Any, NamedTuple, cast

import h5py
import numpy as np
import pandas as pd
import torch
from einops import rearrange
from olmo_core.aliases import PathOrStr
from olmo_core.config import Config, DType
from olmo_core.distributed.utils import get_fs_local_rank
from pyproj import Transformer
from torch.distributed import DeviceMesh
from torch.distributed.tensor import distribute_tensor
from torch.utils.data import Dataset
from tqdm import tqdm
from upath import UPath

from helios.data.constants import (
    BASE_RESOLUTION,
    IMAGE_TILE_SIZE,
    MISSING_VALUE,
    PROJECTION_CRS,
    TIMESTAMPS,
    Modality,
    ModalitySpec,
    TimeSpan,
)
from helios.data.normalize import Normalizer, Strategy
from helios.data.utils import convert_to_db
from helios.dataset.parse import ModalityTile, parse_helios_dataset
from helios.dataset.sample import (
    SampleInformation,
    image_tiles_to_samples,
    load_image_for_sample,
)
from helios.dataset.utils import get_modality_specs_from_names
from helios.types import ArrayTensor

logger = logging.getLogger(__name__)


class HeliosSample(NamedTuple):
    """A sample of the data from the Helios dataset.

    This is a namedtuple that contains the data of a single sample or a batch of samples from the Helios dataset.
    For each modality, we have an ArrayTensor named by the modality, along with the latlon and timestamps.
    """

    sentinel2_l2a: ArrayTensor | None = None  # [B, H, W, T, len(S2_bands)]
    latlon: ArrayTensor | None = None  # [B, 2]
    timestamps: ArrayTensor | None = None  # [B, T, D=3], where D=[day, month, year]
    sentinel1: ArrayTensor | None = None  # [B, H, W, T, len(S1_bands)]
    worldcover: ArrayTensor | None = None  # [B, H, W, 1, len(WC_bands)]
    openstreetmap_raster: ArrayTensor | None = None  # [B, H, W, 1, len(OSM_bands)]

    # TODO: Add unit tests for this
    def shape(self, attribute: str, mask: bool = False) -> Sequence[int]:
        """Returns the expected shape of an attribute.

        This is useful if you want to know what the shape of a
        missing attribute would have been for this sample.

        Args:
            attribute: The attribute to get the shape of, e.g., "sentinel2", "timestamps", etc.
            mask: Whether to get the shape of the mask.

        Returns:
            The shape of the attribute.
        """
        # It is safe to assume we always have Sentinel2, timestamps, and latlon
        # If other attributes are missing, we use Sentinel2 to get its partial shape (B, H, W, T)
        # For static modality like worldcover, we specify the T dimension as 1
        if attribute == "timestamps":
            if not mask:
                if self.timestamps is None:
                    raise ValueError("Timestamps are not present in the sample")
                return self.timestamps.shape
            else:
                # timestamps is a special case which is not in Modality
                raise ValueError("Timestamps are not maskable")
        else:
            attribute_shape = []
            if Modality.get(attribute).get_tile_resolution() > 0:
                # Add batch size (if has), height, width
                attribute_shape += [self.height, self.width]
            if Modality.get(attribute).is_multitemporal:
                # Add number of timesteps
                attribute_shape += [self.time]
            if not mask:
                # Add number of bands
                attribute_shape += [Modality.get(attribute).num_bands]
            else:
                # Add number of band sets
                attribute_shape += [Modality.get(attribute).num_band_sets]
            return attribute_shape

    @staticmethod
    def num_bands(attribute: str) -> int:
        """Get the number of channels for a given attribute."""
        if attribute == "timestamps":
            return len(TIMESTAMPS)
        else:
            return Modality.get(attribute).num_bands

    def as_dict(self, ignore_nones: bool = True) -> dict[str, ArrayTensor | None]:
        """Convert the namedtuple to a dictionary.

        Args:
            ignore_nones: Whether to ignore None values.

        Returns:
            Dictionary representation of the namedtuple.
        """
        return_dict = {}
        for field in self._fields:
            val = getattr(self, field)
            if ignore_nones and (val is None):
                continue
            else:
                return_dict[field] = val
        return return_dict

    @property
    def modalities(self) -> list[str]:
        """Get the modalities present in the sample.

        Includes timestamps and latlon
        """
        return [modality for modality in self.as_dict(ignore_nones=True).keys()]

    @property
    def missing_modalities(self) -> list[str]:
        """Get the modalities missing from the sample."""
        return [
            modality
            for modality in self.as_dict(ignore_nones=True).keys()
            if self.as_dict(ignore_nones=True)[modality] is None
        ]

    def to_device(self, device: torch.device) -> "HeliosSample":
        """Move all tensors to the specified device.

        Args:
            device: The device to move the tensors to.

        Returns:
            A new HeliosSample with all tensors moved to the specified device.
        """
        return HeliosSample(
            **{
                key: val.to(device)
                for key, val in self.as_dict(ignore_nones=True).items()
                if val is not None
            }
        )

    def distribute_tensors(self, device_mesh: DeviceMesh) -> "HeliosSample":
        """Distribute the tensors to the specified device mesh."""
        return HeliosSample(
            **{
                key: distribute_tensor(val, device_mesh)
                for key, val in self.as_dict(ignore_nones=True).items()
            }
        )

    @property
    def batch_size(self) -> int:
        """Get the batch size of the data."""
        vals = [
            cast(ArrayTensor, x).shape[0]
            for x in self.as_dict(ignore_nones=True).values()
        ]
        if len(set(vals)) == 1:
            return vals[0]
        else:
            return 1

    @property
    def height(self) -> int:
        """Get the height of the data."""
        height_width_time_modalities = ["sentinel2_l2a", "sentinel1", "worldcover"]
        for modality in height_width_time_modalities:
            x = getattr(self, modality)
            if x is not None:
                if len(x.shape) == 5:
                    return x.shape[1]
                else:
                    # no batch dimension
                    if len(x.shape) != 4:
                        raise ValueError(f"Unexpected shape {x.shape} for {modality}")
                    return x.shape[0]
        raise ValueError("No modality with height or width present")

    @property
    def width(self) -> int:
        """Get the height of the data."""
        height_width_time_modalities = ["sentinel2_l2a", "sentinel1", "worldcover"]
        for modality in height_width_time_modalities:
            x = getattr(self, modality)
            if x is not None:
                if len(x.shape) == 5:
                    return x.shape[2]
                else:
                    # no batch dimension
                    if len(x.shape) != 4:
                        raise ValueError(f"Unexpected shape {x.shape} for {modality}")
                    return x.shape[1]
        raise ValueError("No modality with height or width present")

    @property
    def time(self) -> int:
        """Get the number of time steps in the data."""
        if self.timestamps is None:
            raise ValueError("Timestamps are not present in the sample")
        return self.timestamps.shape[-2]

    def get_expected_shape(self, attribute: str) -> tuple[int, ...]:
        """Get the expected shape of an attribute."""
        modality_spec = Modality.get(attribute)
        if modality_spec.is_spacetime_varying:
            return (self.height, self.width, self.time, modality_spec.num_bands)
        elif modality_spec.is_space_only_varying:
            return (self.height, self.width, 1, modality_spec.num_bands)
        elif modality_spec.is_time_only_varying:
            return (1, 1, self.time, modality_spec.num_bands)
        else:
            return (1, 1, 1, modality_spec.num_bands)

    def _get_max_t_within_token_budget(
        self, h_w_p: int, max_tokens_per_instance: int
    ) -> int:
        """Find max t possible when subsetting.

        Given a sampled h_w_p (the number of tokens along the h and w dimensions)
        return the maximum t allowed within the
        max_tokens budget so that the patchified
        HeliosSample will have fewer than max_tokens tokens.

        This function assumes we apply (H, W, T=1 patchifying)
        """
        used_tokens = 0
        time_multiply_tokens = 0
        for attribute in self.as_dict(ignore_nones=True).keys():
            if attribute == "timestamps":
                continue
            modality_spec = Modality.get(attribute)
            if modality_spec.is_spacetime_varying:
                # for now, lets assume fixed resolution
                time_multiply_tokens += (h_w_p**2) * modality_spec.num_band_sets
            elif modality_spec.is_space_only_varying:
                # for now, lets assume fixed resolution
                used_tokens += (h_w_p**2) * modality_spec.num_band_sets
            elif modality_spec.is_time_only_varying:
                time_multiply_tokens += modality_spec.num_band_sets
            elif modality_spec.is_static_in_space_and_time:
                used_tokens += modality_spec.num_band_sets
        if time_multiply_tokens == 0:
            # no time-varying inputs, so our return value of t
            # doesn't matter
            return 1
        remaining_tokens = max_tokens_per_instance - used_tokens
        max_t_within_budget = remaining_tokens / time_multiply_tokens
        if max_t_within_budget < 1:
            raise ValueError("patch_size too small for this sample and budget")
        return min(floor(max_t_within_budget), self.time)

    def subset(
        self, patch_size: int, max_tokens_per_instance: int, sampled_hw_p: int
    ) -> "HeliosSample":
        """Subset a HelioSample that is unbatched ie no batch dimension.

        Args:
            patch_size: The patch size being applied to this sample.
            max_tokens_per_instance: The token budget when subsetting. This is used
                to determine the maximum number of timesteps possible for a given
                height and width.
            sampled_hw_p: The number of tokens in the height and width dimensions.

        The returned sample will have shape:
            height = hw_t * patch_size
            width = hw_t * patch_size
            time = max_t
        where hw_t is sampled from hw_to_sample and max_t is the maximum number
        of timesteps allowable so that the total tokens (per instance) is >=
        max_tokens_per_instance
        """
        max_t = self._get_max_t_within_token_budget(
            sampled_hw_p, max_tokens_per_instance
        )
        sampled_hw = sampled_hw_p * patch_size
        start_h = np.random.choice(self.height - sampled_hw + 1)
        start_w = np.random.choice(self.width - sampled_hw + 1)
        start_t = np.random.choice(self.time - max_t + 1)
        new_data_dict: dict[str, ArrayTensor] = {}
        for attribute, modality in self.as_dict(ignore_nones=True).items():
            assert modality is not None
            if attribute == "timestamps":
                new_data_dict[attribute] = modality[start_t : start_t + max_t]
                continue
            modality_spec = Modality.get(attribute)
            if modality_spec.is_spacetime_varying:
                # for now, lets assume fixed resolution
                new_data_dict[attribute] = modality[
                    start_h : start_h + sampled_hw,
                    start_w : start_w + sampled_hw,
                    start_t : start_t + max_t,
                ]
            elif modality_spec.is_space_only_varying:
                # for now, lets assume fixed resolution
                new_data_dict[attribute] = modality[
                    start_h : start_h + sampled_hw, start_w : start_w + sampled_hw
                ]
            elif modality_spec.is_time_only_varying:
                new_data_dict[attribute] = modality[start_t : start_t + max_t]
            elif modality_spec.is_static_in_space_and_time:
                new_data_dict[attribute] = modality
        return HeliosSample(**new_data_dict)


def collate_helios(batch: list[tuple[int, HeliosSample]]) -> tuple[int, HeliosSample]:
    """Collate function that automatically handles any modalities present in the samples."""

    # Stack tensors while handling None values
    def stack_or_none(attr: str) -> torch.Tensor | None:
        """Stack the tensors while handling None values."""
        # For partially missing samples we use MISSING_VALUE so we only check the first sample
        if getattr(batch[0][1], attr) is None:
            return None
        stacked_tensor = torch.stack(
            [torch.from_numpy(getattr(sample, attr)) for _, sample in batch], dim=0
        )
        return stacked_tensor

    # TODO: Gets all non-None modalities ASSUMES ALL SAMPLES HAVE THE SAME MODALITIES
    patch_size, batch_zero = batch[0]
    sample_fields = batch_zero.modalities

    # Create a dictionary of stacked tensors for each field
    collated_dict = {field: stack_or_none(field) for field in sample_fields}
    return patch_size, HeliosSample(**collated_dict)


class GetItemArgs(NamedTuple):
    """Arguments for the __getitem__ method of the HeliosDataset."""

    idx: int
    patch_size: int
    sampled_hw_p: int
    token_budget: int | None = None


class HeliosDataset(Dataset):
    """Helios dataset."""

    PROJECTION_CRS = PROJECTION_CRS
    h5py_folder: str = "h5py_data"

    def __init__(
        self,
        supported_modalities: list[ModalitySpec],
        dtype: DType,
        h5py_dir: UPath | None = None,
        tile_path: UPath | None = None,
        normalize: bool = True,
        use_samples_with_missing_supported_modalities: bool = False,
        multiprocessed_h5_creation: bool = True,
    ):
        """Initialize the dataset.

        To use an already created h5py directory, set h5py_dir to the path of the h5py directory.
        To use a raw tile directory, set tile_path to the path of the tile directory, this will create the h5py files in a prepare step before training.
        Warning from OLMo-core:
            In distributed settings, be sure that the :data:`work_dir` is shared among all local ranks
            and :data:`fs_local_rank` is set accordingly. Once those fields are set you should then call
            :meth:`prepare()` in the main process before doing anything else.

        Args:
            supported_modalities: The modalities to include in the dataset.
            tile_path: The path to the raw dataset (image tile directory). If None we will use the h5py_dir to load the dataset. Mutually exclusive with h5py_dir.
            dtype: The dtype of the data.
            use_samples_with_missing_supported_modalities: If True, use samples that are missing a supported modality.
            normalize: If True, apply normalization to the data, if False, do not apply
                normalization.
            h5py_dir: The path to the h5py directory containing preprocessed data. If None, the dataset will be created from raw data. Mutually exclusive with tile_path.

            h5py_folder: The folder name to store the h5py files when creating from raw data.
            multiprocessed_h5_creation: If True, create the h5py files in parallel using multiprocessing.

        Returns:
            None
        """
        if h5py_dir is None and tile_path is None:
            raise ValueError("Either h5py_dir or tile_path must be provided")
        if h5py_dir is not None and tile_path is not None:
            raise ValueError("Only one of h5py_dir or tile_path can be provided")
        if h5py_dir is not None:
            self.h5py_dir = h5py_dir
            self.tile_path = h5py_dir.parent.parent
            # Ensure that the supported modalities are present in the h5py directory
            for modality in supported_modalities:
                if modality.name not in self.h5py_dir.parent.name:
                    raise ValueError(
                        f"The modality {modality.name} is not present in the h5py directory"
                    )
        else:
            self.tile_path = tile_path
            self.h5py_dir: Path | None = None  # type: ignore

        self.multiprocessed_h5_creation = multiprocessed_h5_creation
        self.supported_modalities = supported_modalities
        self.use_samples_missing_supported_modalities = (
            use_samples_with_missing_supported_modalities
        )

        self.dtype = dtype
        self.normalize = normalize
        if self.normalize:
            self.normalizer_predefined = Normalizer(Strategy.PREDEFINED)
            self.normalizer_computed = Normalizer(Strategy.COMPUTED)

        self._fs_local_rank = get_fs_local_rank()
        self._work_dir: Path | None = None  # type: ignore
        self._work_dir_set = False
        self.sample_indices: np.ndarray | None = None
        self.latlon_distribution: np.ndarray | None = None

    @property
    def fingerprint_version(self) -> str:
        """The version of the fingerprint."""
        return "v0.1"

    @property
    def fingerprint(self) -> str:
        """Can be used to identify/compare a dataset."""
        if not self.is_dataset_prepared:
            raise RuntimeError("Dataset must be prepared before creating a fingerprint")
        sha256_hash = hashlib.sha256()
        sha256_hash.update(
            f"tile_path={self.tile_path},"
            f"supported_modalities={sorted([m.name for m in self.supported_modalities])},"
            f"sample_size={len(self)},"
            f"dtype={self.dtype}".encode()
        )
        return sha256_hash.hexdigest()

    @property
    def fs_local_rank(self) -> int:
        """Get the fs local rank."""
        return self._fs_local_rank

    @fs_local_rank.setter
    def fs_local_rank(self, _fs_local_rank: int) -> None:
        """Set the fs local rank."""
        self._fs_local_rank = _fs_local_rank

    @property
    def work_dir(self) -> Path:
        """Get the working directory."""
        if self._work_dir is not None:
            return self._work_dir
        else:
            return Path(tempfile.gettempdir())

    @work_dir.setter
    def work_dir(self, _work_dir: PathOrStr) -> None:
        """Set the working directory."""
        self._work_dir = Path(_work_dir)
        self._work_dir_set = True

    @property
    def work_dir_set(self) -> bool:
        """Check if the working directory was explicitly set."""
        return self._work_dir_set

    def process_sample_into_h5(
        self, index_sample_tuple: tuple[int, SampleInformation]
    ) -> None:
        """Process a sample into an h5 file."""
        i, sample = index_sample_tuple
        h5_file_path = self._get_h5_file_path(i)
        if h5_file_path.exists():
            return
        self._create_h5_file(sample, h5_file_path)

    def create_h5_dataset(self, samples: list[SampleInformation]) -> None:
        """Create a dataset of the samples in h5 format in a shared weka directory under the given fingerprint."""
        total_sample_indices = len(samples)

        if self.multiprocessed_h5_creation:
            num_processes = max(1, mp.cpu_count() - 2)
            logger.info(f"Creating H5 dataset using {num_processes} processes")
            with mp.Pool(processes=num_processes) as pool:
                # Process samples in parallel and track progress with tqdm
                _ = list(
                    tqdm(
                        pool.imap(self.process_sample_into_h5, enumerate(samples)),
                        total=total_sample_indices,
                        desc="Creating H5 files",
                    )
                )
        else:
            for i, sample in enumerate(samples):
                self.process_sample_into_h5((i, sample))

    def set_h5py_dir(self, num_samples: int) -> None:
        """Set the h5py directory.

        This can only be set once to ensure consistency.

        Args:
            num_samples: Number of samples in the dataset
        """
        if self.h5py_dir is not None:
            logger.warning("h5py_dir is already set, ignoring new value")
            return

        self.h5py_dir = (
            self.tile_path
            / self.h5py_folder
            / "_".join(
                sorted([modality.name for modality in self.supported_modalities])
            )
            / str(num_samples)
        )
        logger.info(f"Setting h5py_dir to {self.h5py_dir}")
        os.makedirs(self.h5py_dir, exist_ok=True)

    def prepare(self, samples: list[SampleInformation] | None = None) -> None:
        """Prepare the dataset.

        THIS SHOULD BE CALLED BY THE MAIN PROCESS ONLY and should happen
        before any other process tries to use the dataset
        """
        logger.info("Preparing dataset...")
        if self.is_dataset_prepared:
            logger.info("Dataset is already prepared")
            return
        if self.h5py_dir is None:
            logger.warning(
                "h5py_dir is not set, Generating H5 files from raw tile directory"
            )
            if samples is None:
                samples = self._get_samples()  # type: ignore
            if len(samples) == 0:
                raise ValueError("No samples provided")
            samples = self._filter_samples(samples)  # type: ignore
            num_samples = len(samples)
            self.set_h5py_dir(num_samples)

            logger.info("Attempting to create H5 files may take some time...")
            self.create_h5_dataset(samples)
        else:
            logger.info("H5 files already exist, skipping creation")
            logger.info(f"H5 files exist in {self.h5py_dir}")
            num_samples = int(self.h5py_dir.name)
        if samples is None:
            samples = []
        self.latlon_distribution = self.get_geographic_distribution(samples)
        self.sample_indices = np.arange(num_samples)

    @property
    def is_dataset_prepared(self) -> bool:
        """Check if the dataset is prepared."""
        return self.sample_indices is not None and self.h5py_dir.exists()

    @property
    def latlon_distribution_path(self) -> UPath:
        """Get the path to the latlon distribution file."""
        return self.h5py_dir / "latlon_distribution.npy"

    def save_latlon_distribution(self, latlons: np.ndarray) -> None:
        """Save the latlon distribution to a file."""
        logger.info(f"Saving latlon distribution to {self.latlon_distribution_path}")
        with self.latlon_distribution_path.open("wb") as f:
            np.save(f, latlons)

    def _log_modality_distribution(self, samples: list[SampleInformation]) -> None:
        """Log the modality distribution."""
        # Log modality distribution
        modality_counts: dict[str, int] = {}
        modality_combinations: dict[frozenset[str], int] = {}

        for sample in samples:
            # Count individual modalities
            for modality in sample.modalities:
                modality_counts[modality.name] = (
                    modality_counts.get(modality.name, 0) + 1
                )

            # Count modality combinations
            combination = frozenset(m.name for m in sample.modalities)
            modality_combinations[combination] = (
                modality_combinations.get(combination, 0) + 1
            )

        # Log individual modality counts
        for modality_name, count in modality_counts.items():
            percentage = (count / len(samples)) * 100
            logger.info(
                f"Modality {modality_name}: {count} samples ({percentage:.1f}%)"
            )

        # Log modality combinations
        logger.info("\nModality combinations:")
        for combination, count in modality_combinations.items():
            percentage = (count / len(samples)) * 100
            logger.info(
                f"{'+'.join(sorted(combination))}: {count} samples ({percentage:.1f}%)"
            )

    def _get_samples(self) -> list[SampleInformation]:
        """Get the samples from the raw dataset (image tile directory)."""
        tiles = parse_helios_dataset(self.tile_path, self.supported_modalities)
        samples = image_tiles_to_samples(tiles, self.supported_modalities)
        logger.info(f"Total samples: {len(samples)}")
        logger.info("Distribution of samples before filtering:\n")
        self._log_modality_distribution(samples)
        return samples

    def _filter_samples(
        self, samples: list[SampleInformation]
    ) -> list[SampleInformation]:
        """Filter samples to adjust to the HeliosSample format."""
        logger.info(f"Number of samples before filtering: {len(samples)}")
        filtered_samples = []
        # For now, we use sentinel2 as the base grid with resolution factor 16
        # Avoid samples with NAIP which has a resolution factor of 1
        resolution_factor = Modality.SENTINEL2_L2A.tile_resolution_factor
        for sample in samples:
            if sample.grid_tile.resolution_factor != resolution_factor:
                continue
            # Check if all the modalities are supported that are read in
            if not all(
                modality in self.supported_modalities
                for modality in sample.modalities
                if not modality.ignore_when_parsing
            ):
                logger.info("Skipping sample because it has unsupported modalities")
                continue

            if self.use_samples_missing_supported_modalities:
                if any(
                    modality not in sample.modalities
                    for modality in self.supported_modalities
                ):
                    continue
            if sample.time_span != TimeSpan.YEAR:
                continue
            # check if sample modalities have s1 and s2
            has_s1 = Modality.SENTINEL1 in sample.modalities
            has_s2 = Modality.SENTINEL2_L2A in sample.modalities
            if not has_s2:
                # If any of our samples don't have S2 this will be a problem
                continue
            if has_s1:
                sentinel1_months = len(
                    set(sample.modalities[Modality.SENTINEL1].images)
                )
                if sentinel1_months != 12:
                    continue
            if has_s2:
                sentinel2_months = len(
                    set(sample.modalities[Modality.SENTINEL2_L2A].images)
                )
                if sentinel2_months != 12:
                    continue
            if has_s1 and has_s2:
                # Check if S1 and S2 all have the same 12 months of data
                if sentinel1_months != sentinel2_months:
                    continue
            filtered_samples.append(sample)
        logger.info(f"Number of samples after filtering: {len(filtered_samples)}")
        logger.info("Distribution of samples after filtering:")
        self._log_modality_distribution(filtered_samples)
        return filtered_samples

    @classmethod
    def get_latlon(cls, sample: SampleInformation) -> np.ndarray:
        """Get the latlon of the sample."""
        # Get coordinates at projection units, and then transform to latlon
        grid_resolution = sample.grid_tile.resolution_factor * BASE_RESOLUTION
        x, y = (
            (sample.grid_tile.col + 0.5) * grid_resolution * IMAGE_TILE_SIZE,
            (sample.grid_tile.row + 0.5) * -grid_resolution * IMAGE_TILE_SIZE,
        )
        transformer = Transformer.from_crs(
            sample.grid_tile.crs, cls.PROJECTION_CRS, always_xy=True
        )
        lon, lat = transformer.transform(x, y)
        return np.array([lat, lon])

    def get_geographic_distribution(
        self, samples: list[SampleInformation]
    ) -> np.ndarray:
        """Get the geographic distribution of the dataset.

        Returns:
            numpy.ndarray: Array of shape (N, 2) containing [latitude, longitude]
            coordinates for each of the N samples in the dataset.
        """
        if self.latlon_distribution_path.exists():
            with self.latlon_distribution_path.open("rb") as f:
                return np.load(f)
        if len(samples) == 0:
            raise ValueError("No samples provided")
        latlons = []
        for sample in samples:
            latlon = self.get_latlon(sample)
            latlons.append(latlon)
        latlons = np.vstack(latlons)
        self.save_latlon_distribution(latlons)
        return latlons

    def get_sample_data_for_histogram(
        self, num_samples: int = 100, num_values: int = 100
    ) -> dict[str, Any]:
        """Get the sample data per modality per band for showing the histogram.

        Args:
            num_samples: The number of samples to sample from the dataset.
            num_values: The number of values to sample from each modality per band.

        Returns:
            dict: A dictionary containing the sample data per modality per band.
        """
        if num_samples > len(self):
            raise ValueError(
                f"num_samples {num_samples} is greater than the number of samples in the dataset {len(self)}"
            )
        indices_to_sample = random.sample(list(range(len(self))), k=num_samples)
        sample_data: dict[str, Any] = {}

        # Assume samples could include different modalities and bands
        # TODO: compute the histogram for each modality and band directly
        for i in tqdm(indices_to_sample):
            get_item_args = GetItemArgs(
                idx=i, patch_size=1, sampled_hw_p=IMAGE_TILE_SIZE
            )
            _, sample = self[get_item_args]
            for modality in sample.modalities:
                if modality == "timestamps" or modality == "latlon":
                    continue
                modality_data = sample.as_dict(ignore_nones=True)[modality]
                if modality_data is None:
                    continue
                modality_spec = Modality.get(modality)
                modality_bands = modality_spec.band_order
                if modality not in sample_data:
                    sample_data[modality] = {band: [] for band in modality_bands}
                # for each band, flatten the data and extend the list
                for idx, band in enumerate(modality_bands):
                    sample_data[modality][band].extend(
                        random.sample(
                            modality_data[:, :, :, idx].flatten().tolist(), num_values
                        )
                    )

        return sample_data

    def _get_timestamps(self, sample: SampleInformation) -> np.ndarray:
        """Get the timestamps of the sample."""
        sample_sentinel2_l2a = sample.modalities[Modality.SENTINEL2_L2A]
        timestamps = [i.start_time for i in sample_sentinel2_l2a.images]
        dt = pd.to_datetime(timestamps)
        # Note that month should be 0-indexed
        return np.array([dt.day, dt.month - 1, dt.year]).T

    def __len__(self) -> int:
        """Get the length of the dataset."""
        if self.sample_indices is None:
            raise ValueError("Dataset is not prepared")
        return self.sample_indices.shape[0]

    @classmethod
    def load_sample(
        self, sample_modality: ModalityTile, sample: SampleInformation
    ) -> np.ndarray:
        """Load the sample."""
        image = load_image_for_sample(sample_modality, sample)

        if image.ndim == 4:
            modality_data = rearrange(image, "t c h w -> h w t c")
        else:
            modality_data = rearrange(image, "c h w -> h w c")
        return modality_data

    def normalize_image(self, modality: ModalitySpec, image: np.ndarray) -> np.ndarray:
        """Normalize the image."""
        # Try computed strategy first, if it fails, try predefined strategy
        # TODO: we can also make modality norm strategy configurable later
        try:
            return self.normalizer_computed.normalize(modality, image)
        except Exception:
            return self.normalizer_predefined.normalize(modality, image)

    def _get_h5_file_path(self, index: int) -> UPath:
        """Get the h5 file path."""
        return self.h5py_dir / f"sample_{index}.h5"

    def _create_h5_file(
        self, sample: SampleInformation, h5_file_path: UPath
    ) -> dict[str, Any]:
        """Create the h5 file."""
        sample_dict = {}
        for modality in sample.modalities:
            sample_modality = sample.modalities[modality]
            image = self.load_sample(sample_modality, sample)
            # Convert Sentinel1 data to dB
            if modality == Modality.SENTINEL1:
                image = convert_to_db(image)
            sample_dict[modality.name] = image
            # Get latlon and timestamps from Sentinel2 data
            if modality == Modality.SENTINEL2_L2A:
                sample_dict["latlon"] = self.get_latlon(sample).astype(self.dtype)
                sample_dict["timestamps"] = self._get_timestamps(sample)
        # Save h5 file on WEKA
        with h5_file_path.open("wb") as f:
            with h5py.File(f, "w") as h5file:
                for modality_name, image in sample_dict.items():
                    h5file.create_dataset(modality_name, data=image)
        return sample_dict

    def fill_sample_with_missing_values(
        self, sample_dict: dict[str, Any]
    ) -> tuple[HeliosSample, list[str]]:
        """Fill the sample with missing values."""
        missing_modalities = []
        sample = HeliosSample(**sample_dict)
        for modality in self.supported_modalities:
            if modality.name not in sample_dict.keys():
                logger.info(f"Filling {modality.name} with missing values")
                sample_dict[modality.name] = np.full(
                    sample.get_expected_shape(modality.name),
                    fill_value=MISSING_VALUE,
                    dtype=self.dtype,
                )
                missing_modalities.append(modality.name)
        return HeliosSample(**sample_dict), missing_modalities

    def apply_subset(self, sample: HeliosSample, args: GetItemArgs) -> HeliosSample:
        """Apply the subset to the sample."""
        if args.token_budget is not None:
            sample_subset = sample.subset(
                patch_size=args.patch_size,
                max_tokens_per_instance=args.token_budget,
                sampled_hw_p=args.sampled_hw_p,
            )
        else:
            sample_subset = sample
        return sample_subset

    def read_h5_file(self, h5_file_path: UPath) -> dict[str, Any]:
        """Read the h5 file."""
        with h5_file_path.open("rb") as f:
            with h5py.File(f, "r") as h5file:
                sample_dict = {k: v[()] for k, v in h5file.items()}
        return sample_dict

    def __getitem__(self, args: GetItemArgs) -> tuple[int, HeliosSample]:
        """Get the sample at the given index."""
        h5_file_path = self._get_h5_file_path(args.idx)

        if not h5_file_path.exists():
            raise FileNotFoundError(
                f"H5 file {h5_file_path} does not exist, Be Sure to run prepare before starting Training"
            )
        # We are currently reading the entire h5 file into memory this can be made faster by chunking the dataset appropriately and only reading in the optimal chunks
        # THis io is the current bottleneck of the getitem operation
        sample_dict = self.read_h5_file(h5_file_path)

        sample, missing_modalities = self.fill_sample_with_missing_values(sample_dict)
        subset_sample = self.apply_subset(sample, args)
        sample_dict = subset_sample.as_dict(ignore_nones=True)

        # Sample modalities should be written into the metadata of the h5 dataset
        sample_modalities = list(
            [Modality.get(key) for key in sample_dict.keys() if key != "timestamps"]
        )

        if self.normalize:
            for modality in sample_modalities:
                # DO NOT NORMALIZE MISSING MODALITIES otherwise the MISSING_VALUE will be normalized
                if modality.name in missing_modalities:
                    continue
                sample_dict[modality.name] = self.normalize_image(
                    modality, sample_dict[modality.name]
                )
                sample_dict[modality.name] = sample_dict[modality.name].astype(
                    self.dtype
                )

        return args.patch_size, HeliosSample(**sample_dict)


@dataclass
class HeliosDatasetConfig(Config):
    """Configuration for the HeliosDataset."""

    h5py_dir: str | None
    supported_modality_names: list[str]
    tile_path: str | None = None
    dtype: DType = DType.float32
    normalize: bool = True
    use_samples_with_missing_supported_modalities: bool = False

    def validate(self) -> None:
        """Validate the configuration and build kwargs.

        Args:
            kwargs: Dictionary of arguments to validate

        Raises:
            ValueError: If any arguments are invalid
        """
        # Validate tile_path
        # Check that either a tile path or h5py_dir is provided
        if self.tile_path is None and self.h5py_dir is None:
            raise ValueError("Either a tile path or h5py_dir must be provided")
        if self.tile_path is not None and self.h5py_dir is not None:
            raise ValueError("Only one of tile_path or h5py_dir must be provided")

        # Validate supported_modalities
        if not isinstance(self.supported_modalities, list):
            raise ValueError("supported_modalities must be a list")
        if not all(isinstance(m, ModalitySpec) for m in self.supported_modalities):
            raise ValueError(
                "All elements in supported_modalities must be ModalitySpec"
            )

    @property
    def supported_modalities(self) -> list[ModalitySpec]:
        """Get the supported modalities."""
        return get_modality_specs_from_names(self.supported_modality_names)

    @property
    def tile_upath(self) -> UPath:
        """Get the tile path."""
        return UPath(self.tile_path)

    @property
    def h5py_dir_upath(self) -> UPath:
        """Get the h5py directory."""
        return UPath(self.h5py_dir)

    def build(self) -> "HeliosDataset":
        """Build the dataset."""
        self.validate()
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        if self.h5py_dir is not None:
            kwargs["h5py_dir"] = self.h5py_dir_upath
        else:
            kwargs["tile_path"] = self.tile_upath
        kwargs.pop("supported_modality_names")
        kwargs["supported_modalities"] = self.supported_modalities
        logger.info(f"HeliosDataset kwargs: {kwargs}")
        return HeliosDataset(**kwargs)

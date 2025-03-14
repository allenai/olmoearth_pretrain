"""Dataset module for helios."""

import hashlib
import logging
import os
import random
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from math import floor
from pathlib import Path
from typing import Any, NamedTuple

import h5py
import numpy as np
import pandas as pd
import torch
from einops import rearrange
from olmo_core.aliases import PathOrStr
from olmo_core.config import Config, DType
from olmo_core.distributed.utils import get_fs_local_rank
from pyproj import Transformer
from torch.utils.data import Dataset
from tqdm import tqdm
from upath import UPath

from helios.data.constants import (
    BASE_RESOLUTION,
    IMAGE_TILE_SIZE,
    PROJECTION_CRS,
    TIMESTAMPS,
    Modality,
    ModalitySpec,
    TimeSpan,
)
from helios.data.normalize import Normalizer, Strategy
from helios.data.utils import convert_to_db, update_streaming_stats
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

    We always require sentinel2 data.
    This is a namedtuple that contains the data of a single sample or a batch of samples from the Helios dataset.
    For each modality, we have an ArrayTensor named by the modality, along with the latlon and timestamps.
    """

    sentinel2_l2a: ArrayTensor  # [B, H, W, T, len(S2_bands)]
    latlon: ArrayTensor | None = None  # [B, 2]
    timestamps: ArrayTensor | None = None  # [B, T, D=3], where D=[day, month, year]
    sentinel1: ArrayTensor | None = None  # [B, H, W, T, len(S1_bands)]
    worldcover: ArrayTensor | None = None  # [B, H, W, 1, len(WC_bands)]

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
            if self.sentinel2_l2a is None:
                raise ValueError("Sentinel2 L2A is not present in the sample")
            attribute_shape = []
            if Modality.get(attribute).get_tile_resolution() > 0:
                # Add batch size (if has), height, width
                attribute_shape += self.sentinel2_l2a.shape[:-2]
            if Modality.get(attribute).is_multitemporal:
                # Add number of timesteps
                attribute_shape += [self.sentinel2_l2a.shape[-2]]
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
        """Get the modalities present in the sample."""
        return list(self.as_dict(ignore_nones=True).keys())

    def to_device(self, device: torch.device) -> "HeliosSample":
        """Move all tensors to the specified device.

        Args:
            device: The device to move the tensors to.

        Returns:
            A new HeliosSample with all tensors moved to the specified device.
        """

        def maybe_move_to_device(tensor: torch.Tensor | None) -> torch.Tensor | None:
            """Move the tensor to the specified device if it is not None."""
            if tensor is None:
                return None
            return tensor.to(device)

        return HeliosSample(
            **{key: maybe_move_to_device(val) for key, val in self.as_dict().items()}
        )

    @property
    def batch_size(self) -> int:
        """Get the batch size of the data."""
        if len(self.sentinel2_l2a.shape) == 5:
            return self.sentinel2_l2a.shape[0]
        else:
            return 1

    @property
    def height(self) -> int:
        """Get the height of the data."""
        return self.sentinel2_l2a.shape[1]

    @property
    def width(self) -> int:
        """Get the width of the data."""
        return self.sentinel2_l2a.shape[2]

    @property
    def time(self) -> int:
        """Get the number of time steps in the data."""
        if self.timestamps is None:
            raise ValueError("Timestamps are not present in the sample")
        return self.timestamps.shape[1]

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
        self, patch_size: int, max_tokens_per_instance: int, hw_to_sample: list[int]
    ) -> "HeliosSample":
        """Subset a HelioSample.

        patch_size: the patch size being applied to this sample
        max_tokens_per_instance: the token budget when subsetting. This is used
            to determine the maximum number of timesteps possible for a given
            height and width.
        hw_to_sample: possible values for the number of tokens in the height and width
            dimensions.

        The returned sample will have shape:
            height = hw_t * patch_size
            width = hw_t * patch_size
            time = max_t
        where hw_t is sampled from hw_to_sample and max_t is the maximum number
        of timesteps allowable so that the total tokens (per instance) is >=
        max_tokens_per_instance
        """
        max_height_width = max(self.height, self.width)
        max_height_width_tokens = int(max_height_width / patch_size)
        hw_to_sample = [x for x in hw_to_sample if x <= max_height_width_tokens]
        if len(hw_to_sample) == 0:
            raise ValueError(
                "max height/width allowed by sample smaller than values in hw_to_sample"
            )
        sampled_hw_p = random.choice(hw_to_sample)
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
                new_data_dict[attribute] = modality[:, start_t : start_t + max_t]
                continue

            modality_spec = Modality.get(attribute)
            if modality_spec.is_spacetime_varying:
                # for now, lets assume fixed resolution
                new_data_dict[attribute] = modality[
                    :,
                    start_h : start_h + sampled_hw,
                    start_w : start_w + sampled_hw,
                    start_t : start_t + max_t,
                ]
            elif modality_spec.is_space_only_varying:
                # for now, lets assume fixed resolution
                new_data_dict[attribute] = modality[
                    :, start_h : start_h + sampled_hw, start_w : start_w + sampled_hw
                ]
            elif modality_spec.is_time_only_varying:
                new_data_dict[attribute] = modality[:, start_t : start_t + max_t]
            elif modality_spec.is_static_in_space_and_time:
                new_data_dict[attribute] = modality
        return HeliosSample(**new_data_dict)


def collate_helios(batch: list[HeliosSample]) -> HeliosSample:
    """Collate function that automatically handles any modalities present in the samples."""

    # Stack tensors while handling None values
    def stack_or_none(attr: str) -> torch.Tensor | None:
        """Stack the tensors while handling None values."""
        if getattr(batch[0], attr) is None:
            return None
        return torch.stack(
            [torch.from_numpy(getattr(sample, attr)) for sample in batch], dim=0
        )

    # TODO: Gets all non-None modalities ASSUMES ALL SAMPLES HAVE THE SAME MODALITIES
    sample_fields = batch[0].modalities

    # Create a dictionary of stacked tensors for each field
    collated_dict = {field: stack_or_none(field) for field in sample_fields}
    return HeliosSample(**collated_dict)


class HeliosDataset(Dataset):
    """Helios dataset."""

    PROJECTION_CRS = PROJECTION_CRS

    def __init__(
        self,
        tile_path: UPath,
        supported_modalities: list[ModalitySpec],
        dtype: DType,
        samples: list[SampleInformation] | None = None,
        normalize: bool = True,
        h5py_folder: str = "h5py_data",
    ):
        """Initialize the dataset.

        Warning from OLMo-core:
            In distributed settings, be sure that the :data:`work_dir` is shared among all local ranks
            and :data:`fs_local_rank` is set accordingly. Once those fields are set you should then call
            :meth:`prepare()` in the main process before doing anything else.

        Args:
            supported_modalities: The modalities to include in the dataset.
            tile_path: The path to the raw dataset (image tile directory).
            samples: The samples to include in the dataset.
            dtype: The dtype of the data.
            normalize: If True, apply normalization to the data, if False, do not apply normalization
            h5py_folder: The folder to store the h5py files.

        Returns:
            None
        """
        self.tile_path = tile_path
        self.supported_modalities = supported_modalities
        # If samples are provided, use them, if not, get them from the tile directory
        if not samples:
            samples = self._get_samples()  # type: ignore
        if len(samples) == 0:
            raise ValueError("No samples provided")
        self.samples = self._filter_samples(samples)  # type: ignore
        self.dtype = dtype
        self.normalize = normalize
        self.h5py_dir = (
            self.tile_path
            / h5py_folder
            / "_".join(
                sorted([modality.name for modality in self.supported_modalities])
            )
            / str(len(self.samples))
        )
        os.makedirs(self.h5py_dir, exist_ok=True)

        if self.normalize:
            self.normalizer_predefined = Normalizer(Strategy.PREDEFINED)
            self.normalizer_computed = Normalizer(Strategy.COMPUTED)

        self._fs_local_rank = get_fs_local_rank()
        self._work_dir: Path | None = None  # type: ignore
        self._work_dir_set = False

    @property
    def fingerprint_version(self) -> str:
        """The version of the fingerprint."""
        return "v0.1"

    @property
    def fingerprint(self) -> str:
        """Can be used to identify/compare a dataset."""
        sha256_hash = hashlib.sha256()
        sha256_hash.update(
            f"tile_path={self.tile_path},"
            f"supported_modalities={sorted([m.name for m in self.supported_modalities])},"
            f"sample_size={len(self.samples)},"
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

    def prepare(self) -> None:
        """Prepare the dataset."""
        len(self)

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
            # Check if all the modalities are available that are read in
            if not all(
                modality in sample.modalities
                for modality in self.supported_modalities
                if not modality.ignore_when_parsing
            ):
                continue
            if not all(
                modality in sample.modalities
                for modality in self.supported_modalities
                if modality != Modality.LATLON
            ):
                continue
            # check if sample modalities have s1 and s2
            has_s1 = Modality.SENTINEL1 in sample.modalities
            has_s2 = Modality.SENTINEL2_L2A in sample.modalities
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
            if sample.time_span != TimeSpan.YEAR:
                continue
            filtered_samples.append(sample)
        logger.info(f"Number of samples after filtering: {len(filtered_samples)}")
        logger.info("Distribution of samples after filtering:")
        self._log_modality_distribution(filtered_samples)
        return filtered_samples

    def get_latlon(self, sample: SampleInformation) -> np.ndarray:
        """Get the latlon of the sample."""
        # Get coordinates at projection units, and then transform to latlon
        grid_resolution = sample.grid_tile.resolution_factor * BASE_RESOLUTION
        x, y = (
            (sample.grid_tile.col + 0.5) * grid_resolution * IMAGE_TILE_SIZE,
            (sample.grid_tile.row + 0.5) * -grid_resolution * IMAGE_TILE_SIZE,
        )
        transformer = Transformer.from_crs(
            sample.grid_tile.crs, self.PROJECTION_CRS, always_xy=True
        )
        lon, lat = transformer.transform(x, y)
        return np.array([lat, lon])

    def get_geographic_distribution(self) -> np.ndarray:
        """Get the geographic distribution of the dataset.

        Returns:
            numpy.ndarray: Array of shape (N, 2) containing [latitude, longitude]
            coordinates for each of the N samples in the dataset.
        """
        latlons = []
        for sample in self.samples:
            latlon = self.get_latlon(sample)
            latlons.append(latlon)
        latlons = np.vstack(latlons)
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
            sample = self[i]
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
        return len(self.samples)

    def compute_normalization_values(
        self,
        estimate_from: int | None = None,
    ) -> dict[str, Any]:
        """Compute the normalization values for the dataset in a streaming manner.

        Args:
            estimate_from: The number of samples to estimate the normalization values from.

        Returns:
            dict: A dictionary containing the normalization values for the dataset.
        """
        if estimate_from is not None:
            indices_to_sample = random.sample(list(range(len(self))), k=estimate_from)
        else:
            indices_to_sample = list(range(len(self)))

        norm_dict: dict[str, Any] = {}

        for i in tqdm(indices_to_sample):
            sample = self[i]
            for modality in sample.modalities:
                # Shall we compute the norm stats for worldcover?
                if modality == "timestamps" or modality == "latlon":
                    continue
                modality_data = sample.as_dict(ignore_nones=True)[modality]
                modality_spec = Modality.get(modality)
                modality_bands = modality_spec.band_order
                if modality_data is None:
                    continue
                if modality not in norm_dict:
                    norm_dict[modality] = {}
                    for band in modality_bands:
                        norm_dict[modality][band] = {
                            "mean": 0.0,
                            "var": 0.0,
                            "std": 0.0,
                            "count": 0,
                        }
                # Compute the normalization stats for the modality
                for idx, band in enumerate(modality_bands):
                    modality_band_data = modality_data[:, :, :, idx]  # (H, W, T, C)
                    current_stats = norm_dict[modality][band]
                    new_count, new_mean, new_var = update_streaming_stats(
                        current_stats["count"],
                        current_stats["mean"],
                        current_stats["var"],
                        modality_band_data,
                    )
                    # Update the normalization stats
                    norm_dict[modality][band]["count"] = new_count
                    norm_dict[modality][band]["mean"] = new_mean
                    norm_dict[modality][band]["var"] = new_var

        # Compute the standard deviation
        for modality in norm_dict:
            for band in norm_dict[modality]:
                norm_dict[modality][band]["std"] = (
                    norm_dict[modality][band]["var"]
                    / norm_dict[modality][band]["count"]
                ) ** 0.5

        norm_dict["total_n"] = len(self)
        norm_dict["sampled_n"] = len(indices_to_sample)
        norm_dict["tile_path"] = self.tile_path

        return norm_dict

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
        self, index: int, sample: SampleInformation, h5_file_path: UPath
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
        with h5py.File(h5_file_path, "w") as f:
            for modality_name, image in sample_dict.items():
                f.create_dataset(modality_name, data=image)
        return sample_dict

    def __getitem__(self, index: int) -> HeliosSample:
        """Get the sample at the given index."""
        sample = self.samples[index]
        h5_file_path = self._get_h5_file_path(index)

        if not h5_file_path.exists():
            sample_dict = self._create_h5_file(index, sample, h5_file_path)
        else:
            # Check how long it takes to read h5 data
            if index == 0:
                start_time = time.time()
            with h5py.File(h5_file_path, "r") as f:
                sample_dict = {k: v[()] for k, v in f.items()}
            if index == 0:
                end_time = time.time()
                logger.info(
                    f"Time taken to read h5 data: {end_time - start_time} seconds"
                )

        if self.normalize:
            for modality in sample.modalities:
                sample_dict[modality.name] = self.normalize_image(
                    modality, sample_dict[modality.name]
                )
                sample_dict[modality.name] = sample_dict[modality.name].astype(
                    self.dtype
                )
        return HeliosSample(**sample_dict)


@dataclass
class HeliosDatasetConfig(Config):
    """Configuration for the HeliosDataset."""

    tile_path: str
    supported_modality_names: list[str]
    samples: list[SampleInformation] | None = None
    dtype: DType = DType.float32
    normalize: bool = True

    def validate(self) -> None:
        """Validate the configuration and build kwargs.

        Args:
            kwargs: Dictionary of arguments to validate

        Raises:
            ValueError: If any arguments are invalid
        """
        # Validate tile_path
        if not isinstance(self.tile_upath, UPath):
            raise ValueError("tile_path must be a UPath")

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

    def build(self) -> "HeliosDataset":
        """Build the dataset."""
        self.validate()
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs["tile_path"] = self.tile_upath
        kwargs.pop("supported_modality_names")
        kwargs["supported_modalities"] = self.supported_modalities
        logger.info(f"HeliosDataset kwargs: {kwargs}")
        return HeliosDataset(**kwargs)

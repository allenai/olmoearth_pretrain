"""Dataset module for helios."""

import hashlib
import logging
import random
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass
from math import floor
from typing import Any, NamedTuple, cast

import h5py
import numpy as np
import pandas as pd
import torch
from olmo_core.config import Config, DType
from torch.distributed import DeviceMesh
from torch.distributed.tensor import distribute_tensor
from torch.utils.data import Dataset
from tqdm import tqdm
from upath import UPath

from helios.data.constants import (
    IMAGE_TILE_SIZE,
    MISSING_VALUE,
    TIMESTAMPS,
    Modality,
    ModalitySpec,
)
from helios.data.normalize import Normalizer, Strategy
from helios.dataset.convert_to_h5py import ConvertToH5py
from helios.dataset.sample import SampleInformation
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
    srtm: ArrayTensor | None = None  # [B, H, W, 1, len(SRTM_bands)]
    landsat: ArrayTensor | None = None  # [B, H, W, T, len(LANDSAT_bands)]
    # Unsure what the shapes should be for this one
    naip: ArrayTensor | None = None  # [B, H, W, T, len(NAIP_bands)]

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


# TODO should training modalities be str or modality_spec
class HeliosDataset(Dataset):
    """Helios dataset."""

    def __init__(
        self,
        h5py_dir: UPath,
        training_modalities: list[str],
        dtype: DType,
        normalize: bool = True,
        use_samples_with_missing_supported_modalities: bool = False,
        cache_dir: UPath | None = None,
        samples_per_sec: float | None = None,
    ):
        """Initialize the dataset.

        To use an already created h5py directory, set h5py_dir to the path of the h5py directory.
        To use a raw tile directory, set tile_path to the path of the tile directory, this will create the h5py files in a prepare step before training.
        Warning from OLMo-core:
            In distributed settings, be sure that the :data:`work_dir` is shared among all local ranks
            and :data:`fs_local_rank` is set accordingly. Once those fields are set you should then call
            :meth:`prepare()` in the main process before doing anything else.

        Args:
            h5py_dir: The path to the h5py directory containing preprocessed data.
            training_modalities: The modalities to use for training.
            dtype: The dtype of the data.
            normalize: If True, apply normalization to the data, if False, do not apply
                normalization.
            use_samples_with_missing_supported_modalities: If True, use samples that are missing a supported modality.
            cache_dir: optional local directory to cache the H5 files.
            samples_per_sec: throttle to reading this many samples per second. This
                throttling only applies when reading from the h5py_dir, not the
                cache_dir (if set).

        Returns:
            None
        """
        self.h5py_dir = h5py_dir
        if not self.h5py_dir.exists():
            raise FileNotFoundError(f"H5PY directory does not exist: {self.h5py_dir}")
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.training_modalities = training_modalities
        self.use_samples_with_missing_supported_modalities = (
            use_samples_with_missing_supported_modalities
        )

        self.dtype = dtype
        self.normalize = normalize
        if self.normalize:
            self.normalizer_predefined = Normalizer(Strategy.PREDEFINED)
            self.normalizer_computed = Normalizer(Strategy.COMPUTED)

        if samples_per_sec is None:
            self.sec_per_sample = None
        else:
            self.sec_per_sample = 1 / samples_per_sec
        self.last_read_time = time.time()

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
        # Parse from the h5py_dir
        supported_modalities_folder = self.h5py_dir.parent.name
        supported_modalities = supported_modalities_folder.split("_")
        # join back sentinel_l2a and openstreetmap_raster if applicable
        if "l2a" in supported_modalities:
            supported_modalities.remove("l2a")
            supported_modalities.remove("sentinel2")
            supported_modalities.append("sentinel2_l2a")
        if "raster" in supported_modalities:
            supported_modalities.remove("raster")
            supported_modalities.remove("openstreetmap")
            supported_modalities.append("openstreetmap_raster")
        num_samples = int(self.h5py_dir.name)

        tile_path = self.h5py_dir.parent.parent.parent

        logger.info(f"tile_path: {tile_path}")
        logger.info(f"supported_modalities: {supported_modalities}")
        logger.info(f"num_samples: {num_samples}")
        logger.info(f"dtype: {self.dtype}")

        sha256_hash.update(
            f"tile_path={tile_path},"
            f"supported_modalities={sorted(supported_modalities)},"
            f"sample_size={num_samples},"
            f"dtype={self.dtype}".encode()
        )
        return sha256_hash.hexdigest()

    @property
    def sample_metadata_path(self) -> UPath:
        """Get the path to the sample metadata file."""
        return self.h5py_dir / ConvertToH5py.sample_metadata_fname

    @property
    def latlon_distribution_path(self) -> UPath:
        """Get the path to the latlon distribution file."""
        return self.h5py_dir / ConvertToH5py.latlon_distribution_fname

    @property
    def is_dataset_prepared(self) -> bool:
        """Check if the dataset is prepared."""
        return self.sample_indices is not None

    def _filter_sample_indices_for_training(self) -> None:
        """Filter the sample indices for training.

        Updates the sample indices numpy array to only include the indices we want to train on.
        """
        # Read the metadata CSV
        metadata_df = pd.read_csv(self.sample_metadata_path)
        logger.info(f"Metadata CSV has {len(metadata_df)} samples")
        logger.info(f"columns: {metadata_df.columns}")
        # For now we want to filter out any samples that have NAIP DATA or don't have any of the training modalities
        # Get the indices of samples that have NAIP data
        if "naip" in metadata_df.columns:
            naip_indices = metadata_df[metadata_df["naip"] == 1].index
            self.naip_indices = naip_indices
        else:
            self.naip_indices = np.array([])
        logger.info(f"NAIP indices: {self.naip_indices}")

        # Get the indices of samples that don't have any training modalities that are
        # multi-temporal.
        multitemporal_training_modalities = [
            modality
            for modality in self.training_modalities
            if Modality.get(modality).is_multitemporal
        ]
        if len(multitemporal_training_modalities) == 0:
            raise ValueError("no multi-temporal modalities are specified for training")
        no_multitemporal_indices = metadata_df[
            metadata_df[multitemporal_training_modalities].sum(axis=1) == 0
        ].index
        # Filter these indices out
        logger.info(f"Filtering out {len(self.naip_indices)} samples with NAIP data")
        logger.info(
            f"Filtering out {len(no_multitemporal_indices)} samples without any training modalities"
        )
        self.sample_indices = np.setdiff1d(self.sample_indices, self.naip_indices)
        self.sample_indices = np.setdiff1d(
            self.sample_indices, no_multitemporal_indices
        )
        # raise an error if any of the naip indices are still in the sample indices
        if any(index in self.naip_indices for index in self.sample_indices):
            raise ValueError("Some NAIP indices are still in the sample indices")
        logger.info(
            f"Filtered {len(self.naip_indices) + len(no_multitemporal_indices)} samples to {self.sample_indices.shape} samples"
        )

    def prepare(self) -> None:
        """Prepare the dataset.

        THIS SHOULD BE CALLED BY THE MAIN PROCESS ONLY and should happen
        before any other process tries to use the dataset
        """
        logger.info("Preparing dataset...")
        if self.is_dataset_prepared:
            logger.info("Dataset is already prepared")
            return

        logger.info("H5 files already exist, skipping creation")
        logger.info(f"H5 files exist in {self.h5py_dir}")
        num_samples = int(self.h5py_dir.name)
        self.latlon_distribution = self.get_geographic_distribution()
        self.sample_indices = np.arange(num_samples)
        self._filter_sample_indices_for_training()

    # TODO: Needs to be gotten or owned from th other class

    def save_latlon_distribution(self, latlons: np.ndarray) -> None:
        """Save the latlon distribution to a file."""
        logger.info(f"Saving latlon distribution to {self.latlon_distribution_path}")
        with self.latlon_distribution_path.open("wb") as f:
            np.save(f, latlons)

    def _log_modality_distribution(self, samples: list[SampleInformation]) -> None:
        """Log the modality distribution."""
        # TODO: have a version that reads this from the sample metadata file
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

    def get_geographic_distribution(self) -> np.ndarray:
        """Get the geographic distribution of the dataset.

        Returns:
            numpy.ndarray: Array of shape (N, 2) containing [latitude, longitude]
            coordinates for each of the N samples in the dataset.
        """
        if self.latlon_distribution_path.exists():
            with self.latlon_distribution_path.open("rb") as f:
                return np.load(f)

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

    def __len__(self) -> int:
        """Get the length of the dataset."""
        if self.sample_indices is None:
            raise ValueError("Dataset is not prepared")
        return self.sample_indices.shape[0]

    def normalize_image(self, modality: ModalitySpec, image: np.ndarray) -> np.ndarray:
        """Normalize the image."""
        # Try computed strategy first, if it fails, try predefined strategy
        # TODO: we can also make modality norm strategy configurable later
        try:
            return self.normalizer_computed.normalize(modality, image)
        except Exception:
            return self.normalizer_predefined.normalize(modality, image)

    def fill_sample_with_missing_values(
        self, sample_dict: dict[str, Any]
    ) -> tuple[HeliosSample, list[str]]:
        """Fill the sample with missing values."""
        missing_modalities = []
        sample = HeliosSample(**sample_dict)
        for modality in self.training_modalities:
            if modality not in sample_dict.keys():
                logger.debug(f"Filling {modality} with missing values")
                sample_dict[modality] = np.full(
                    sample.get_expected_shape(modality),
                    fill_value=MISSING_VALUE,
                    dtype=self.dtype,
                )
                missing_modalities.append(modality)
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

    def _apply_throttling(self) -> None:
        """Apply read throttling.

        This function is called when reading a sample from the h5py_dir, and it applies
        the configured throttling.
        """
        if self.sec_per_sample is None:
            return
        elapsed = time.time() - self.last_read_time
        time_to_sleep = self.sec_per_sample - elapsed
        self.last_read_time = time.time()
        logger.info(f"{elapsed} elapsed since last read, sleeping for {time_to_sleep}")
        if time_to_sleep <= 0:
            return
        time.sleep(time_to_sleep)

    def read_h5_file(self, h5_file_path: UPath) -> dict[str, Any]:
        """Read the h5 file."""
        if self.cache_dir is not None:
            cache_file_path = self.cache_dir / h5_file_path.name
            logger.info(f"Caching H5 file {h5_file_path} to {cache_file_path}")
            if not cache_file_path.exists():
                self._apply_throttling()
                # Copy to a temp file first and then atomically rename it to avoid
                # concurrency issues.
                tmp_file_path = self.cache_dir / (h5_file_path.name + ".tmp")
                with h5_file_path.open("rb") as src, tmp_file_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                tmp_file_path.rename(cache_file_path)
            h5_file_path = cache_file_path

        else:
            self._apply_throttling()

        sample_dict = {}
        with h5_file_path.open("rb") as f:
            with h5py.File(f, "r") as h5file:
                logger.info(f"Reading h5 file {h5_file_path} with keys {h5file.keys()}")
                # Not sure lat lon should be here
                sample_dict = {
                    k: v[()]
                    for k, v in h5file.items()
                    if k in self.training_modalities or k in ["latlon", "timestamps"]
                }
        return sample_dict

    def _get_h5_file_path(self, index: int) -> UPath:
        """Get the h5 file path."""
        return self.h5py_dir / ConvertToH5py.sample_file_pattern.format(index=index)

    def __getitem__(self, args: GetItemArgs) -> tuple[int, HeliosSample]:
        """Get the sample at the given index."""
        if hasattr(self, "sample_indices") and self.sample_indices is not None:
            index = self.sample_indices[args.idx]
        else:
            index = args.idx
        h5_file_path = self._get_h5_file_path(index)

        if not h5_file_path.exists():
            raise FileNotFoundError(
                f"H5 file {h5_file_path} does not exist, Be Sure to run prepare before starting Training"
            )
        # We are currently reading the entire h5 file into memory this can be made faster by chunking the dataset appropriately and only reading in the optimal chunks
        # THis io is the current bottleneck of the getitem operation
        sample_dict = self.read_h5_file(h5_file_path)

        # Fill any training modalities that are not present in the h5 file with missing values
        sample, missing_modalities = self.fill_sample_with_missing_values(sample_dict)
        subset_sample = self.apply_subset(sample, args)
        sample_dict = subset_sample.as_dict(ignore_nones=True)
        logger.info(f"Sample dict keys {sample_dict.keys()}")
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

    h5py_dir: str
    training_modalities: list[str]
    dtype: DType = DType.float32
    normalize: bool = True
    use_samples_with_missing_supported_modalities: bool = False
    cache_dir: str | None = None
    samples_per_sec: float | None = None

    def validate(self) -> None:
        """Validate the configuration and build kwargs.

        Args:
            kwargs: Dictionary of arguments to validate

        Raises:
            ValueError: If any arguments are invalid
        """
        # Validate supported_modalities
        if not isinstance(self.training_modalities, list):
            raise ValueError("training_modalities must be a list")

    @property
    def h5py_dir_upath(self) -> UPath:
        """Get the h5py directory."""
        return UPath(self.h5py_dir)

    @property
    def cache_dir_upath(self) -> UPath:
        """Get the cache directory."""
        return UPath(self.cache_dir)

    def build(self) -> "HeliosDataset":
        """Build the dataset."""
        self.validate()
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs["h5py_dir"] = self.h5py_dir_upath
        kwargs["cache_dir"] = self.cache_dir_upath
        logger.info(f"HeliosDataset kwargs: {kwargs}")
        return HeliosDataset(**kwargs)

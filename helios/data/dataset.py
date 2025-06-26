"""Dataset module for helios."""

import hashlib
import logging
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass
from math import floor
from typing import Any, NamedTuple, cast

import h5py

# hdf5 plugin is needed to decompress the data for certain compression types
import hdf5plugin  # noqa: F401
import numpy as np
import pandas as pd
import torch
from olmo_core.config import Config
from torch.distributed import DeviceMesh
from torch.distributed.tensor import distribute_tensor
from torch.utils.data import Dataset
from upath import UPath

from helios.data.constants import (
    MAX_SEQUENCE_LENGTH,
    MISSING_VALUE,
    TIMESTAMPS,
    Modality,
    ModalitySpec,
)
from helios.data.normalize import Normalizer, Strategy
from helios.dataset.convert_to_h5py import ConvertToH5py
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
    # naip with different tile resolution is currently not used in favor of naip_10.
    naip: ArrayTensor | None = None  # [B, H, W, T, len(NAIP_bands)]
    # naip_10 is currently 4x the height/width of sentinel2_l2a.
    naip_10: ArrayTensor | None = None  # [B, H, W, T, len(NAIP_bands)]

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
            return self.get_expected_shape(attribute, mask)

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
        """Get the height of the data at resolution_factor == 16."""
        for modality in self.modalities:
            if modality == "timestamps":
                continue
            modality_spec = Modality.get(modality)
            if not modality_spec.is_spatial:
                continue
            x = getattr(self, modality)
            if x is not None:
                if len(x.shape) == 5:
                    return x.shape[1] // modality_spec.image_tile_size_factor
                else:
                    # no batch dimension
                    if len(x.shape) != 4:
                        raise ValueError(f"Unexpected shape {x.shape} for {modality}")
                    return x.shape[0] // modality_spec.image_tile_size_factor
        raise ValueError("No modality with height or width present")

    @property
    def width(self) -> int:
        """Get the width of the data at resolution_factor == 16."""
        for modality in self.modalities:
            if modality == "timestamps":
                continue
            modality_spec = Modality.get(modality)
            if not modality_spec.is_spatial:
                continue
            x = getattr(self, modality)
            if x is not None:
                if len(x.shape) == 5:
                    return x.shape[2] // modality_spec.image_tile_size_factor
                else:
                    # no batch dimension
                    if len(x.shape) != 4:
                        raise ValueError(f"Unexpected shape {x.shape} for {modality}")
                    return x.shape[1] // modality_spec.image_tile_size_factor
        raise ValueError("No modality with height or width present")

    @property
    def time(self) -> int:
        """Get the number of time steps in the data."""
        if self.timestamps is None:
            raise ValueError("Timestamps are not present in the sample")
        return self.timestamps.shape[-2]

    @property
    def valid_time(self) -> int:
        """Get the minimum number of valid time steps in a batch."""
        return self.timesteps_with_at_least_one_modality.shape[0]

    @property
    def timesteps_with_at_least_one_modality(self) -> torch.Tensor:
        """Get timesteps with at least one modality present."""
        per_modality_present_masks = []
        for modality in self.modalities:
            if modality == "timestamps":
                continue
            modality_spec = Modality.get(modality)
            if modality_spec.is_multitemporal:
                data = getattr(self, modality)
                if isinstance(data, np.ndarray):
                    raise ValueError(
                        "timesteps_with_at_least_one_modality is not yet supported for numpy arrays"
                    )
                # Get all timestamps that are present for all samples for the given modality
                present_mask = (data != MISSING_VALUE).all(dim=(0, 1, 2, 4))
                per_modality_present_masks.append(present_mask)
        at_least_one_modality_present_timestep_mask = torch.stack(
            per_modality_present_masks, dim=1
        ).any(dim=1)
        timesteps_with_at_least_one_modality = torch.where(
            at_least_one_modality_present_timestep_mask
        )[0]
        return timesteps_with_at_least_one_modality

    def get_expected_shape(self, attribute: str, mask: bool = False) -> tuple[int, ...]:
        """Get the expected shape of an attribute."""
        modality_spec = Modality.get(attribute)
        if mask:
            num_bands = modality_spec.num_band_sets
        else:
            num_bands = modality_spec.num_bands

        if modality_spec.is_spacetime_varying:
            return (
                self.height * modality_spec.image_tile_size_factor,
                self.width * modality_spec.image_tile_size_factor,
                self.time,
                num_bands,
            )
        elif modality_spec.is_space_only_varying:
            return (
                self.height * modality_spec.image_tile_size_factor,
                self.width * modality_spec.image_tile_size_factor,
                1,
                num_bands,
            )
        elif modality_spec.is_time_only_varying:
            return (1, 1, self.time, num_bands)
        else:
            return (1, 1, 1, num_bands)

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

    @staticmethod
    def _get_valid_start_ts(
        missing_timesteps: dict[str, Any], max_t: int, current_length: int
    ) -> list[int]:
        """Get valid starting timesteps."""
        if current_length > max_t:
            # We can randomly sample from the range of valid starting timesteps because current_length exceeds max_t
            if not missing_timesteps:
                # No missing timesteps info available - all timesteps are potentially valid
                # Create a range of all possible starting positions that fit within max_t
                valid_start_ts = list(range(current_length - max_t + 1))
            else:
                # We have missing timesteps info - need to find valid starting positions
                # that ensure we have at least some present data at the chosen start_t
                start_ts = set()
                for modality in missing_timesteps:
                    valid_timesteps = np.flatnonzero(missing_timesteps[modality])
                    valid_timesteps = valid_timesteps[
                        valid_timesteps + max_t <= current_length
                    ]
                    start_ts.update(valid_timesteps)
                valid_start_ts = list(start_ts)
        else:
            # Picking the first timestep aims to maximize the number of present timesteps
            valid_start_ts = [0]
        if len(valid_start_ts) == 0:
            logger.warning(
                f"No valid start timesteps found for {missing_timesteps} with max_t {max_t} and current_length {current_length}"
            )
            raise ValueError(
                f"No valid start timesteps found for {missing_timesteps} with max_t {max_t} and current_length {current_length}"
            )
        return sorted(valid_start_ts)

    def subset(
        self,
        patch_size: int,
        max_tokens_per_instance: int | None,
        sampled_hw_p: int,
        current_length: int,
        missing_timesteps_masks: dict[str, Any] = {},
    ) -> "HeliosSample":
        """Subset a HelioSample that is unbatched ie no batch dimension.

        Args:
            patch_size: The patch size being applied to this sample.
            max_tokens_per_instance: The token budget when subsetting. This is used
                to determine the maximum number of timesteps possible for a given
                height and width. If None, this operation is a no-op.
            sampled_hw_p: The number of tokens in the height and width dimensions.
            current_length: The current maximum sequence length of the sample.
            missing_timesteps_masks: A dictionary of missing timesteps masks.

        We apply current_length here to ensure that the subset focuses on the valid timesteps
        instead of the padded timesteps.

        The returned sample will have shape:
            height = hw_t * patch_size
            width = hw_t * patch_size
            time = max_t
        where hw_t is sampled from hw_to_sample and max_t is the maximum number
        of timesteps allowable so that the total tokens (per instance) is >=
        max_tokens_per_instance
        """
        if max_tokens_per_instance is None:
            return self
        max_t = self._get_max_t_within_token_budget(
            sampled_hw_p, max_tokens_per_instance
        )
        sampled_hw = sampled_hw_p * patch_size
        start_h = np.random.choice(self.height - sampled_hw + 1)
        start_w = np.random.choice(self.width - sampled_hw + 1)

        valid_start_ts = self._get_valid_start_ts(
            missing_timesteps_masks, max_t, current_length
        )
        start_t = np.random.choice(valid_start_ts)

        new_data_dict: dict[str, ArrayTensor] = {}
        for attribute, modality in self.as_dict(ignore_nones=True).items():
            assert modality is not None
            if attribute == "timestamps":
                new_data_dict[attribute] = modality[start_t : start_t + max_t]
                continue
            modality_spec = Modality.get(attribute)
            if modality_spec.is_spacetime_varying:
                new_data_dict[attribute] = modality[
                    start_h * modality_spec.image_tile_size_factor : (
                        start_h + sampled_hw
                    )
                    * modality_spec.image_tile_size_factor,
                    start_w * modality_spec.image_tile_size_factor : (
                        start_w + sampled_hw
                    )
                    * modality_spec.image_tile_size_factor,
                    start_t : start_t + max_t,
                ]
            elif modality_spec.is_space_only_varying:
                new_data_dict[attribute] = modality[
                    start_h * modality_spec.image_tile_size_factor : (
                        start_h + sampled_hw
                    )
                    * modality_spec.image_tile_size_factor,
                    start_w * modality_spec.image_tile_size_factor : (
                        start_w + sampled_hw
                    )
                    * modality_spec.image_tile_size_factor,
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
        dtype: np.dtype,
        max_sequence_length: int = MAX_SEQUENCE_LENGTH,
        normalize: bool = True,
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
            max_sequence_length: The maximum sequence length that we pad all time dimensions to.
            normalize: If True, apply normalization to the data, if False, do not apply
                normalization.
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

        self.dtype = dtype
        self.normalize = normalize
        if self.normalize:
            self.normalizer_predefined = Normalizer(Strategy.PREDEFINED)
            self.normalizer_computed = Normalizer(Strategy.COMPUTED)
        self.max_sequence_length = max_sequence_length

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

        if "naip" in supported_modalities and "10" in supported_modalities:
            supported_modalities.remove("naip")
            supported_modalities.remove("10")
            supported_modalities.append("naip_10")
        # latlons are saved with every h5py file, see
        # helios.dataset.convert_to_h5py.ConvertToH5py._create_h5_file
        supported_modalities.append("latlon")
        num_samples = int(self.h5py_dir.name)

        tile_path = self.h5py_dir.parent.parent.parent

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

        # Get the indices of samples that don't have any training modalities that are
        # multi-temporal. We want to remove these samples.
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
        logger.info(
            f"Filtering out {len(no_multitemporal_indices)} samples without any training modalities"
        )
        self.sample_indices = np.setdiff1d(
            self.sample_indices, no_multitemporal_indices
        )
        logger.info(
            f"Filtered {len(no_multitemporal_indices)} samples to {self.sample_indices.shape} samples"
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

        num_samples = int(self.h5py_dir.name)
        self.latlon_distribution = self.get_geographic_distribution()
        self.sample_indices = np.arange(num_samples)
        self._filter_sample_indices_for_training()

    def get_geographic_distribution(self) -> np.ndarray:
        """Get the geographic distribution of the dataset.

        Returns:
            numpy.ndarray: Array of shape (N, 2) containing [latitude, longitude]
            coordinates for each of the N samples in the dataset.
        """
        if self.latlon_distribution_path.exists():
            with self.latlon_distribution_path.open("rb") as f:
                return np.load(f)

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

    def _fill_missing_timesteps(
        self,
        modality_data: np.ndarray,
        missing_timestep_mask: np.ndarray,
    ) -> np.ndarray:
        """Fill the missing timesteps with the missing value."""
        # cast to appropriate dtype to prevent overflow from missing values
        modality_data = modality_data.astype(self.dtype)
        # Get the shape of the data to create properly sized temporal layers
        h, w, t, c = modality_data.shape

        full_timesteps_data = np.full(
            (h, w, self.max_sequence_length, c),
            MISSING_VALUE,
            dtype=self.dtype,
        )

        # Copy the existing data to the appropriate timestep positions
        present_indices = np.where(missing_timestep_mask)[0]
        for i, idx in enumerate(present_indices):
            if i < t:  # Only copy if we have data for this timestep
                full_timesteps_data[..., idx, :] = modality_data[..., i, :]

        return full_timesteps_data

    def _fill_missing_modality(
        self, sample: HeliosSample, modality: str
    ) -> HeliosSample:
        """Fill an array of shape of modality with the missing value."""
        expected_shape = sample.get_expected_shape(modality)
        logger.info(f"Filling {modality} with shape {expected_shape}")
        return np.full(
            expected_shape,
            fill_value=MISSING_VALUE,
            dtype=self.dtype,
        )

    def fill_sample_with_missing_values(
        self, sample_dict: dict[str, Any], missing_timesteps_masks: dict[str, Any]
    ) -> tuple[HeliosSample, list[str]]:
        """Fill the sample with missing values."""
        assert (
            sample_dict["timestamps"].shape[0] == self.max_sequence_length
        ), f"Timestamps shape {sample_dict['timestamps'].shape[0]} does not match max_sequence_length {self.max_sequence_length}"
        missing_modalities = []
        sample = HeliosSample(**sample_dict)
        for modality in self.training_modalities:
            # If one modality is completely missing, we need to fill it all with missing values
            if modality not in sample_dict.keys():
                logger.debug(f"Filling {modality} with missing values")
                sample_dict[modality] = self._fill_missing_modality(sample, modality)
                missing_modalities.append(modality)
                continue

            # For multi-temporal modalities, we need to handle missing timesteps
            # The missing_timesteps_masks indicates which timesteps are present (True) or missing (False)
            if modality in missing_timesteps_masks:
                mask = missing_timesteps_masks[modality]
                modality_data = sample_dict[modality]
                # cast to appropriate dtype to prevent overflow from missing values
                modality_data = modality_data.astype(self.dtype)

                # As long as the #timesteps is less than the max_sequence_length, we will impute by missing value
                has_missing_timesteps = (
                    not np.all(mask) or len(mask) < self.max_sequence_length
                )
                if has_missing_timesteps:
                    # By default, we will fill missing timesteps with the missing value
                    modality_data = self._fill_missing_timesteps(modality_data, mask)
                # Update the sample dictionary with the potentially imputed data
                sample_dict[modality] = modality_data
        return HeliosSample(**sample_dict), missing_modalities

    def _pad_timestamps(
        self, sample_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], int]:
        """Pad the timestamps to the max_sequence_length."""
        timestamps_data = sample_dict["timestamps"]
        current_length = timestamps_data.shape[0]
        if current_length < self.max_sequence_length:
            pad_width = ((0, self.max_sequence_length - current_length), (0, 0))
            # We pad at the end with copies of the last timestep
            padded_timestamps = np.pad(
                timestamps_data, pad_width=pad_width, mode="edge"
            )
            sample_dict["timestamps"] = padded_timestamps
        return sample_dict, current_length

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

    def read_h5_file(
        self, h5_file_path: UPath
    ) -> tuple[dict[str, Any], dict[str, Any]]:
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
                # timestamps should not be a floating string
                sample_dict = {
                    k: v[()]
                    for k, v in h5file.items()
                    if k in self.training_modalities
                    or k in [Modality.LATLON.name, "timestamps"]
                }

                # Log the dtype for each modality
                for k, v in sample_dict.items():
                    logger.info(f"Modality {k} has dtype {v.dtype}")

                if (
                    missing_mask_group_name
                    := ConvertToH5py.missing_timesteps_mask_group_name
                ) in h5file:
                    missing_timesteps_masks = {
                        k: v[()]
                        for k, v in h5file[missing_mask_group_name].items()
                        if k in self.training_modalities
                    }
                else:
                    # To preserve backwards compatibility, we set missing_timesteps_masks to an empty dict if it doesn't exist in file
                    missing_timesteps_masks = {}
        return sample_dict, missing_timesteps_masks

    def _get_h5_file_path(self, index: int) -> UPath:
        """Get the h5 file path."""
        return self.h5py_dir / ConvertToH5py.sample_file_pattern.format(index=index)

    @staticmethod
    def _crop_timestamps_and_masks(
        timestamps: np.ndarray, missing_timesteps_masks: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Crop the timestamps to the first and last valid timestep of the present modalities."""
        # Assumes that the missing timesteps masks has already been filtered for training modalities
        # get first present timestep
        if not missing_timesteps_masks:
            first_valid_timestep = 0
            last_valid_timestep = MAX_SEQUENCE_LENGTH
        else:
            # Timestep masks are the same length as the timestamps
            first_valid_timestep = MAX_SEQUENCE_LENGTH
            last_valid_timestep = 0
            for timestep_mask in missing_timesteps_masks.values():
                valid_timesteps = np.where(timestep_mask)[0]
                if len(valid_timesteps) > 0:
                    first_valid_timestep = min(first_valid_timestep, valid_timesteps[0])
                    last_valid_timestep = max(last_valid_timestep, valid_timesteps[-1])
        timestamps = timestamps[first_valid_timestep : last_valid_timestep + 1]
        for modality, timestep_mask in missing_timesteps_masks.items():
            missing_timesteps_masks[modality] = timestep_mask[
                first_valid_timestep : last_valid_timestep + 1
            ]
        return timestamps, missing_timesteps_masks

    def __getitem__(self, args: GetItemArgs) -> tuple[int, HeliosSample]:
        """Get the sample at the given index."""
        if hasattr(self, "sample_indices") and self.sample_indices is not None:
            index = self.sample_indices[args.idx]
        else:
            index = args.idx
        h5_file_path = self._get_h5_file_path(index)

        sample_dict, missing_timesteps_masks = self.read_h5_file(h5_file_path)
        timestamps, missing_timesteps_masks = self._crop_timestamps_and_masks(
            sample_dict["timestamps"], missing_timesteps_masks
        )
        sample_dict["timestamps"] = timestamps
        sample_dict, current_length = self._pad_timestamps(sample_dict)
        # fill sample currently takes like .08 seconds which may bottleneck smaller models
        sample, missing_modalities = self.fill_sample_with_missing_values(
            sample_dict, missing_timesteps_masks
        )

        subset_sample = sample.subset(
            patch_size=args.patch_size,
            max_tokens_per_instance=args.token_budget,
            sampled_hw_p=args.sampled_hw_p,
            current_length=current_length,
            missing_timesteps_masks=missing_timesteps_masks,
        )

        sample_dict = subset_sample.as_dict(ignore_nones=True)

        if self.normalize:
            for modality_name in sample_dict.keys():
                if modality_name == "timestamps":
                    continue
                # DO NOT NORMALIZE MISSING MODALITIES otherwise the MISSING_VALUE will be normalized
                if modality_name in missing_modalities:
                    logger.info(
                        f"Skipping normalization for {modality_name} because it is in missing_modalities"
                    )
                    continue
                logger.info(f"Normalizing {modality_name}")
                modality_data = sample_dict[modality_name]
                missing_mask = modality_data == MISSING_VALUE
                normalized_data = self.normalize_image(
                    Modality.get(modality_name), modality_data
                )
                # Sentinel Values must be reset after normalization so they can be recognized by missing mask
                sample_dict[modality_name] = np.where(
                    missing_mask, modality_data, normalized_data
                ).astype(self.dtype)

        return args.patch_size, HeliosSample(**sample_dict)


@dataclass
class HeliosDatasetConfig(Config):
    """Configuration for the HeliosDataset."""

    h5py_dir: str
    training_modalities: list[str]
    dtype: str = "float32"
    normalize: bool = True
    cache_dir: str | None = None
    samples_per_sec: float | None = None

    def get_numpy_dtype(self) -> np.dtype:
        """Get the numpy dtype."""
        if self.dtype == "float16":
            return np.float16
        elif self.dtype == "float32":
            return np.float32
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")

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
        kwargs["cache_dir"] = (
            self.cache_dir_upath if self.cache_dir is not None else None
        )
        kwargs["dtype"] = self.get_numpy_dtype()
        logger.info(f"HeliosDataset kwargs: {kwargs}")
        return HeliosDataset(**kwargs)

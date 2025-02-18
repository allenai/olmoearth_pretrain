"""Dataset module for helios."""

import hashlib
import logging
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from olmo_core.aliases import PathOrStr
from olmo_core.distributed.utils import get_fs_local_rank
from pyproj import Transformer
from torch.utils.data import Dataset
from upath import UPath

from helios.data.constants import (
    BASE_RESOLUTION,
    IMAGE_TILE_SIZE,
    TIMESTAMPS,
    Modality,
    ModalitySpec,
    TimeSpan,
)
from helios.data.normalize import NORMALIZE_STRATEGY, Normalizer, Strategy
from helios.data.utils import convert_to_db
from helios.dataset.parse import ModalityTile
from helios.dataset.sample import SampleInformation, load_image_for_sample
from helios.types import ArrayTensor

logger = logging.getLogger(__name__)


class HeliosSample(NamedTuple):
    """A sample of the data from the Helios dataset.

    We always require sentinel2 data.
    This is a namedtuple that contains the data of a single sample or a batch of samples from the Helios dataset.
    For each modality, we have an ArrayTensor named by the modality, along with the latlon and timestamps.
    """

    sentinel2: ArrayTensor  # [B, H, W, T, len(S2_bands)]
    sentinel1: ArrayTensor | None = None  # [B, H, W, T, len(S1_bands)]
    worldcover: ArrayTensor | None = None  # [B, H, W, len(WC_bands)]
    latlon: ArrayTensor | None = None  # [B, 2]
    timestamps: ArrayTensor | None = None  # [B, T, D=3], where D=[day, month, year]

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
            if self.sentinel2 is None:
                raise ValueError("Sentinel2 is not present in the sample")
            attribute_shape = []
            if Modality.get(attribute).get_tile_resolution() > 0:
                # Add batch size (if has), height, width
                attribute_shape += self.sentinel2.shape[:-2]
            if Modality.get(attribute).is_multitemporal:
                # Add number of timesteps
                attribute_shape += [self.sentinel2.shape[-2]]
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


def collate_helios(batch: list[HeliosSample]) -> HeliosSample:
    """Collate function."""

    # Stack tensors while handling None values
    def stack_or_none(attr: str) -> torch.Tensor | None:
        """Stack the tensors while handling None values."""
        if batch[0].__getattribute__(attr) is None:
            # TODO: THis will need to updated to handle sometimes missing modalities
            return None
        return torch.stack(
            [torch.from_numpy(getattr(sample, attr)) for sample in batch], dim=0
        )

    return HeliosSample(
        sentinel2=stack_or_none("sentinel2"),
        sentinel1=stack_or_none("sentinel1"),
        worldcover=stack_or_none("worldcover"),
        latlon=stack_or_none("latlon"),
        timestamps=stack_or_none("timestamps"),
    )


class HeliosDataset(Dataset):
    """Helios dataset."""

    def __init__(
        self,
        *samples: SampleInformation,
        path: UPath,
        supported_modalities: list[ModalitySpec],
        dtype: np.dtype = np.float32,
    ):
        """Initialize the dataset.

        Warning from OLMo-core:
            In distributed settings, be sure that the :data:`work_dir` is shared among all local ranks
            and :data:`fs_local_rank` is set accordingly. Once those fields are set you should then call
            :meth:`prepare()` in the main process before doing anything else.

        Args:
            samples: The samples to include in the dataset.
            path: The path to the dataset root directory.
            supported_modalities: The modalities to include in the dataset.
            dtype: The dtype of the data.
        """
        self.supported_modalities = supported_modalities
        self.samples = self._filter_samples(list(samples))
        self.path = path
        self.dtype = dtype
        # Initialize both normalizers for different modalities
        self.normalizer_predefined = Normalizer(Strategy.PREDEFINED)
        self.normalizer_computed = Normalizer(Strategy.COMPUTED)
        self._fs_local_rank = get_fs_local_rank()
        self._work_dir: Path | None = None  # type: ignore
        self._work_dir_set = False

    def _filter_samples(
        self, samples: list[SampleInformation]
    ) -> list[SampleInformation]:
        """Filter samples to adjust to the HeliosSample format."""
        logger.info(f"Number of samples before filtering: {len(samples)}")
        filtered_samples = []
        # For now, we use sentinel2 as the base grid with resolution factor 16
        # Avoid samples with NAIP which has a resolution factor of 1
        resolution_factor = Modality.SENTINEL2.tile_resolution_factor
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
            # check if sample modalities have s1 and s2
            has_s1 = Modality.SENTINEL1 in sample.modalities
            has_s2 = Modality.SENTINEL2 in sample.modalities
            if has_s1:
                sentinel1_months = len(
                    set(sample.modalities[Modality.SENTINEL1].images)
                )
                if sentinel1_months != 12:
                    continue
            if has_s2:
                sentinel2_months = len(
                    set(sample.modalities[Modality.SENTINEL2].images)
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
        return filtered_samples

    @property
    def fingerprint_version(self) -> str:
        """The version of the fingerprint."""
        return "v0.1"

    @property
    def fingerprint(self) -> str:
        """Can be used to identify/compare a dataset."""
        sha256_hash = hashlib.sha256()
        sha256_hash.update(
            f"path={self.path},"
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

    def _get_latlon(self, sample: SampleInformation) -> np.ndarray:
        """Get the latlon of the sample."""
        # Get coordinates at projection units, and then transform to latlon
        grid_resolution = sample.grid_tile.resolution_factor * BASE_RESOLUTION
        x, y = (
            (sample.grid_tile.col + 0.5) * grid_resolution * IMAGE_TILE_SIZE,
            (sample.grid_tile.row + 0.5) * -grid_resolution * IMAGE_TILE_SIZE,
        )
        transformer = Transformer.from_crs(
            sample.grid_tile.crs, "EPSG:4326", always_xy=True
        )
        lon, lat = transformer.transform(x, y)
        return np.array([lat, lon])

    def _get_timestamps(self, sample: SampleInformation) -> np.ndarray:
        """Get the timestamps of the sample."""
        sample_sentinel2 = sample.modalities[Modality.SENTINEL2]
        timestamps = [i.start_time for i in sample_sentinel2.images]
        dt = pd.to_datetime(timestamps)
        # Note that month should be 0-indexed
        return np.array([dt.day, dt.month - 1, dt.year]).T

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.samples)

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
        # TODO: THere should be a dict per modality
        return modality_data

    def normalize_image(self, modality: ModalitySpec, image: np.ndarray) -> np.ndarray:
        """Normalize the image."""
        if NORMALIZE_STRATEGY[modality] == Strategy.PREDEFINED:
            return self.normalizer_predefined.normalize(modality, image)
        elif NORMALIZE_STRATEGY[modality] == Strategy.COMPUTED:
            return self.normalizer_computed.normalize(modality, image)
        else:
            raise ValueError("Unknown normalization strategy!")

    def __getitem__(self, index: int) -> HeliosSample:
        """Get the item at the given index."""
        sample = self.samples[index]
        sample_dict = {}
        for modality in sample.modalities:
            sample_modality = sample.modalities[modality]
            image = self.load_sample(sample_modality, sample)
            # Convert Sentinel1 data to dB
            if modality == Modality.SENTINEL1:
                image = convert_to_db(image)
            # Normalize data and convert to dtype
            image = self.normalize_image(modality, image)
            sample_dict[modality.name] = image.astype(self.dtype)
            # Get latlon and timestamps from Sentinel2 data
            if modality == Modality.SENTINEL2:
                sample_dict["latlon"] = self._get_latlon(sample).astype(self.dtype)
                sample_dict["timestamps"] = self._get_timestamps(sample)

        return HeliosSample(**sample_dict)

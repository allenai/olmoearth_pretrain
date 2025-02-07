"""Dataset module for helios."""

import hashlib
import logging
import tempfile
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from olmo_core.aliases import PathOrStr
from olmo_core.distributed.utils import get_fs_local_rank
from pyproj import Transformer
from torch.utils.data import Dataset
from upath import UPath

from helios.constants import LATLON_BANDS, S2_BANDS, TIMESTAMPS
from helios.data.constants import BASE_RESOLUTION, IMAGE_TILE_SIZE, Modality
from helios.dataset.parse import TimeSpan
from helios.dataset.sample import SampleInformation, load_image_for_sample
from helios.types import ArrayTensor

logger = logging.getLogger(__name__)


# TODO: THIS SHOULD BE THE OUTPUT OF THE DATASET GET ITEM
class HeliosSample(NamedTuple):
    """A sample of the data from the Helios dataset.

    This is a namedtuple that contains the data for a single sample from the Helios dataset.
    For each modality. we have an ArrayTensor named by modality, positions in lat lon of each sample and
    timestamps of each sample.

    Args:
        s2: ArrayTensor | None = None  # [B, len(S2_bands), T H, W]
        latlon: ArrayTensor | None = None  # [B, 2]
        timestamps: ArrayTensor | None = None  # [B, D=3, T], where D=[day, month, year]
    """

    # if an attribute is added here, its bands must also
    # be added to attribute_to_bands

    # input shape is (B, C, T, H, W)
    s2: ArrayTensor | None = None  # [B, len(S2_bands), T H, W]
    latlon: ArrayTensor | None = None  # [B, 2]
    timestamps: ArrayTensor | None = None  # [B, D=3, T], where D=[day, month, year]

    def as_dict(self, ignore_nones: bool = True) -> dict[str, Any]:
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

    @staticmethod
    def attribute_to_bands() -> dict[str, list[str]]:
        """Get the bands for each attribute.

        Returns:
            A dictionary mapping attribute names to their corresponding bands.
        """
        return {"s2": S2_BANDS, "latlon": LATLON_BANDS, "timestamps": TIMESTAMPS}

    @property
    def b(self) -> int:
        """Get the batch size.

        Returns:
            The batch size of the sample.
        """
        if self.s2 is None:
            raise ValueError("S2 is not present in the sample")
        if len(self.s2.shape) == 5:
            return self.s2.shape[0]
        else:
            raise ValueError("This is a single sample and not a batch")

    @property
    def t(self) -> int:
        """Get the number of timesteps.

        Returns:
            The number of timesteps in the sample.
        """
        if self.s2 is None:
            raise ValueError("S2 is not present in the sample")
        return self.s2.shape[-3]

    @property
    def h(self) -> int:
        """Get the height of the image.

        Returns:
            The height of the image in the sample.
        """
        if self.s2 is None:
            raise ValueError("S2 is not present in the sample")
        return self.s2.shape[-2]

    @property
    def w(self) -> int:
        """Get the width of the image.

        Returns:
            The width of the image in the sample.
        """
        if self.s2 is None:
            raise ValueError("S2 is not present in the sample")
        return self.s2.shape[-1]


def collate_helios(batch: list[HeliosSample]) -> HeliosSample:
    """Collate function."""

    # Stack tensors while handling None values
    def stack_or_none(attr: str) -> torch.Tensor | None:
        """Stack the tensors while handling None values."""
        if batch[0].__getattribute__(attr) is None:
            return None
        return torch.stack([getattr(sample, attr) for sample in batch], dim=0)

    return HeliosSample(
        s2=stack_or_none("s2"),
        latlon=stack_or_none("latlon"),
        timestamps=stack_or_none("timestamps"),
    )


class HeliosDataset(Dataset):
    """Helios dataset."""

    def __init__(
        self,
        *samples: SampleInformation,
        path: UPath,
        dtype: np.dtype = np.float32,
    ):
        """Initialize the dataset.

        Warning from OLMo-core:
            In distributed settings, be sure that the :data:`work_dir` is shared among all local ranks
            and :data:`fs_local_rank` is set accordingly. Once those fields are set you should then call
            :meth:`prepare()` in the main process before doing anything else.

        Args:
            samples: The samples to include in the dataset.
            path: The path to the dataset.
            dtype: The dtype of the data.
        """
        self.samples = self._filter_samples(list(samples))
        self.path = path
        self.dtype = dtype
        self._fs_local_rank = get_fs_local_rank()
        self._work_dir: Path | None = None  # type: ignore
        self._work_dir_set = False

    def _filter_samples(
        self, samples: list[SampleInformation]
    ) -> list[SampleInformation]:
        """Filter samples to adjust to the HeliosSample format."""
        # Right now, we only need S2 data with complete year data (12 months)
        # Later, more modalities can be easily added
        filtered_samples = []
        for sample in samples:
            for modality, image_tile in sample.modalities.items():
                if modality == Modality.S2 and sample.time_span == TimeSpan.YEAR:
                    timestamps = [i.start_time for i in image_tile.images]
                    if len(timestamps) == 12:
                        filtered_samples.append(sample)
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

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.samples)

    def __getitem__(self, index: int) -> HeliosSample:
        """Get the item at the given index."""
        sample = self.samples[index]
        sample_s2 = sample.modalities[Modality.S2]
        timestamps = [i.start_time for i in sample_s2.images]
        image = load_image_for_sample(sample_s2, sample)
        s2_data = rearrange(image, "t c h w -> c t h w")
        dt = pd.to_datetime(timestamps)
        time_data = np.array([dt.day, dt.month, dt.year])  # [3, T]
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
        latlon_data = np.array([lat, lon])
        return HeliosSample(
            s2=s2_data,
            latlon=latlon_data,
            timestamps=time_data,
        )

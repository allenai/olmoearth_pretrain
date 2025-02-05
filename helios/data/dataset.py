"""Dataset module for helios."""

import logging
from typing import Any, NamedTuple, Optional

import numpy as np
from torch.utils.data import Dataset
from upath import UPath
from pathlib import Path

from olmo_core.distributed.utils import get_fs_local_rank
from olmo_core.aliases import PathOrStr

from helios.constants import LATLON_BANDS, S2_BANDS, TIMESTAMPS
from helios.data.data_source_io import DataSourceReader, DataSourceReaderRegistry
from helios.dataset.index import SampleInformation
from helios.types import ArrayTensor

from helios.dataset.sample import SampleInformation, load_image_for_sample
from helios.data.constants import Modality

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


class HeliosDataset(Dataset):
    """Helios dataset."""

    def __init__(
        self,
        *samples: SampleInformation,
        path: UPath,
        dtype: np.dtype,
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
        self.samples = list(samples)
        self.path = path
        self.dtype = dtype
        self.fs_local_rank = get_fs_local_rank()
        self.work_dir: Optional[Path] = None
        self.work_dir_set = False

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
        return self.fs_local_rank

    @fs_local_rank.setter
    def fs_local_rank(self, fs_local_rank: int):
        self.fs_local_rank = fs_local_rank

    @property
    def work_dir(self) -> Path:
        if self.work_dir is not None:
            return self.work_dir
        else:
            return Path(tempfile.gettempdir())

    @work_dir.setter
    def work_dir(self, work_dir: PathOrStr):
        self.work_dir = Path(work_dir)
        self.work_dir_set = True

    def prepare(self):
        len(self)

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.samples)

    def __getitem__(self, index: int) -> HeliosSample:
        """Get the item at the given index."""
        sample: SampleInformation = self.samples[index]
        for modality, image_tiles in sample.modalities.items():
            image = load_image_for_sample(image_tile, sample)
            if modality == Modality.S1:
                s1_data = image.permute(1, 0, 2, 3)  # from TxCxHxW to CxTxHxW
            elif modality == Modality.S2:
                s2_data = image.permute(1, 0, 2, 3)  # from TxCxHxW to CxTxHxW
            else:
                pass
        # TODO: convert grid_tile to latlon and timestamps
        # TODO: either do imputation here, or just add padding to collate function, make all modalities have the same shape
        return HeliosSample(s1=s1_data, s2=s2_data, latlon=sample.grid_tile.latlon, timestamps=sample.timestamps)

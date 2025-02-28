"""Base eval dataset class."""

from collections.abc import Sequence
from enum import Enum

import torch.multiprocessing
from torch.utils.data import Dataset, default_collate

from helios.train.masking import MaskedHeliosSample


class EvalType(Enum):
    """Possible evaluation task types."""

    classifciaton = "classification"
    segmentation = "segmentation"


class BaseEvalDataset(Dataset):
    """Base evaluation dataset."""

    class_or_seg: EvalType

    def __len__(self) -> int:
        """Length of this dataset."""
        raise NotImplementedError

    @staticmethod
    def collate_fn(
        batch: Sequence[tuple[MaskedHeliosSample, torch.Tensor]],
    ) -> tuple[MaskedHeliosSample, torch.Tensor]:
        """Collate function for DataLoaders."""
        samples, targets = zip(*batch)
        # we assume that the same values are consistently None
        collated_sample = default_collate(
            [s.as_dict(return_none=False) for s in samples]
        )
        collated_target = default_collate([t for t in targets])
        return MaskedHeliosSample(**collated_sample), collated_target

    def __getitem__(self, idx: int) -> tuple[MaskedHeliosSample, torch.Tensor]:
        """Return an item from the dataset."""
        raise NotImplementedError

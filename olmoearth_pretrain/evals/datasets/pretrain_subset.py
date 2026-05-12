"""Eval dataset adapter that loads a subset of pretraining data.

Wraps OlmoEarthDataset to expose the eval dataset interface
(returns MaskedOlmoEarthSample, dummy_label) so it can be used
with the downstream evaluator callback for embedding diagnostics.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch.utils.data import Dataset
from upath import UPath

from olmoearth_pretrain.data.dataset import GetItemArgs, OlmoEarthDataset
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample

logger = logging.getLogger(__name__)

DEFAULT_PATCH_SIZE = 4
DEFAULT_HW_P = 8
DEFAULT_MAX_SAMPLES = 512


class PretrainSubsetDataset(Dataset):
    """Wraps OlmoEarthDataset for use as an eval dataset.

    Returns (MaskedOlmoEarthSample, dummy_label) to match the eval
    dataset interface. Uses a fixed subset of indices for reproducibility.
    """

    def __init__(
        self,
        h5py_dir: str,
        training_modalities: list[str],
        max_samples: int = DEFAULT_MAX_SAMPLES,
        patch_size: int = DEFAULT_PATCH_SIZE,
        hw_p: int = DEFAULT_HW_P,
        seed: int = 42,
    ) -> None:
        """Initialize with a fixed reproducible subset of training indices."""
        self.patch_size = patch_size
        self.hw_p = hw_p
        self.max_samples = max_samples

        self._dataset = OlmoEarthDataset(
            h5py_dir=UPath(h5py_dir),
            training_modalities=training_modalities,
            dtype=np.float32,
            normalize=True,
        )
        self._dataset.prepare()

        total = len(self._dataset)
        n = min(max_samples, total)
        rng = np.random.RandomState(seed)
        self._indices = rng.choice(total, size=n, replace=False).tolist()

    def __len__(self) -> int:
        """Return number of samples in the subset."""
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[MaskedOlmoEarthSample, torch.Tensor]:
        """Return (MaskedOlmoEarthSample, dummy_label) for the given index."""
        real_idx = self._indices[idx]
        args = GetItemArgs(
            idx=real_idx,
            patch_size=self.patch_size,
            sampled_hw_p=self.hw_p,
        )
        _, sample = self._dataset[args]
        masked = MaskedOlmoEarthSample.from_olmoearthsample(sample)
        dummy_label = torch.tensor(0, dtype=torch.long)
        return masked, dummy_label

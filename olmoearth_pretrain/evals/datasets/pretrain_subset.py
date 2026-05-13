"""Eval dataset adapter that loads a subset of pretraining data.

Wraps OlmoEarthDataset to expose the eval dataset interface
(returns MaskedOlmoEarthSample, dummy_label) so it can be used
with the downstream evaluator callback for embedding diagnostics.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from upath import UPath

from olmoearth_pretrain.data.dataset import GetItemArgs, OlmoEarthDataset
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample

logger = logging.getLogger(__name__)

DEFAULT_PATCH_SIZE = 4
DEFAULT_HW_P = 8
DEFAULT_MAX_SAMPLES = 512
WORLDCOVER_CLASSES = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100])
OSM_TARGET_MODALITY = "openstreetmap_raster"
SRTM_TARGET_MODALITY = "srtm"
WORLDCOVER_TARGET_MODALITY = "worldcover"


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
        split: str = "train",
        target_modality: str | None = None,
        label_seed: int = 42,
        train_samples: int = DEFAULT_MAX_SAMPLES,
        valid_samples: int = DEFAULT_MAX_SAMPLES,
        test_samples: int = DEFAULT_MAX_SAMPLES,
    ) -> None:
        """Initialize with a fixed reproducible subset of training indices."""
        self.patch_size = patch_size
        self.hw_p = hw_p
        self.max_samples = max_samples
        self.target_modality = target_modality

        self._dataset = OlmoEarthDataset(
            h5py_dir=UPath(h5py_dir),
            training_modalities=training_modalities,
            dtype=np.float32,
            normalize=True,
        )
        self._dataset.prepare()
        self._label_dataset = None
        if target_modality is not None:
            # Include the input modalities so extract_hwt_from_sample_dict has a
            # spatially-present modality to read H/W/T from even when the
            # (often non-multitemporal) target is missing for a given sample.
            self._label_dataset = OlmoEarthDataset(
                h5py_dir=UPath(h5py_dir),
                training_modalities=list(training_modalities) + [target_modality],
                dtype=np.float32,
                normalize=False,
            )
            self._label_dataset.prepare()
            # Align positional indexing with the input dataset so the same
            # GetItemArgs.idx resolves to the same H5 sample for both.
            self._label_dataset.sample_indices = self._dataset.sample_indices.copy()

        if target_modality is None:
            total = len(self._dataset)
            n = min(max_samples, total)
            rng = np.random.RandomState(seed)
            self._indices = rng.choice(total, size=n, replace=False).tolist()
        else:
            eligible_positions = self._positions_with_target_present(
                self._dataset, target_modality
            )
            selected = self._select_split_indices(
                total=len(eligible_positions),
                split=split,
                seed=label_seed,
                train_samples=train_samples,
                valid_samples=valid_samples,
                test_samples=test_samples,
            )
            self._indices = eligible_positions[selected].tolist()

    @staticmethod
    def _positions_with_target_present(
        dataset: OlmoEarthDataset, target_modality: str
    ) -> np.ndarray:
        """Positions into dataset.sample_indices whose H5 sample has the target."""
        metadata_df = pd.read_csv(str(dataset.sample_metadata_path))
        if target_modality not in metadata_df.columns:
            raise ValueError(
                f"Target modality '{target_modality}' has no presence column in "
                f"{dataset.sample_metadata_path}"
            )
        present_by_h5_idx = metadata_df[target_modality].to_numpy() > 0
        eligible_mask = present_by_h5_idx[dataset.sample_indices]
        eligible_positions = np.where(eligible_mask)[0]
        if eligible_positions.size == 0:
            raise ValueError(
                f"No samples with target modality '{target_modality}' present "
                f"after input-modality filtering."
            )
        return eligible_positions

    @staticmethod
    def _select_split_indices(
        total: int,
        split: str,
        seed: int,
        train_samples: int,
        valid_samples: int,
        test_samples: int,
    ) -> list[int]:
        """Select deterministic disjoint index subsets for held-out target probes."""
        split_sizes = {
            "train": train_samples,
            "valid": valid_samples,
            "val": valid_samples,
            "test": test_samples,
        }
        if split not in split_sizes:
            raise ValueError(f"Unsupported pretrain subset split: {split}")

        rng = np.random.RandomState(seed)
        indices = rng.permutation(total)
        train_end = min(train_samples, total)
        valid_end = min(train_end + valid_samples, total)
        test_end = min(valid_end + test_samples, total)
        split_to_slice = {
            "train": slice(0, train_end),
            "valid": slice(train_end, valid_end),
            "val": slice(train_end, valid_end),
            "test": slice(valid_end, test_end),
        }
        selected = indices[split_to_slice[split]]
        if selected.size == 0:
            raise ValueError(
                f"No samples selected for split {split}; total={total}, "
                f"train={train_samples}, valid={valid_samples}, test={test_samples}"
            )
        return selected.tolist()

    @staticmethod
    def _squeeze_label(label: torch.Tensor) -> torch.Tensor:
        """Remove singleton batch/time/channel axes from pretrain target arrays."""
        label = label.squeeze()
        if label.ndim == 3 and label.shape[-1] == 1:
            label = label.squeeze(-1)
        return label

    @staticmethod
    def _worldcover_label(label: torch.Tensor) -> torch.Tensor:
        """Map raw ESA WorldCover class codes to contiguous class ids."""
        label = PretrainSubsetDataset._squeeze_label(label).long()
        if (
            label.numel() > 0
            and label.min() >= 0
            and label.max() < len(WORLDCOVER_CLASSES)
        ):
            return label
        mapped = torch.full_like(label, fill_value=-1)
        classes = WORLDCOVER_CLASSES.to(label.device)
        for class_idx, class_code in enumerate(classes):
            mapped[label == class_code] = class_idx
        return mapped

    @staticmethod
    def _osm_label(label: torch.Tensor) -> torch.Tensor:
        """Convert multi-channel OSM raster labels to a single class id per pixel."""
        label = label.float().squeeze()
        if label.ndim != 3:
            raise ValueError(
                f"Expected OSM label with 3 dims [H, W, C], got {label.shape}"
            )
        if label.shape[0] in (29, 30) and label.shape[-1] not in (29, 30):
            channels_last = label.movedim(0, -1)
        else:
            channels_last = label
        valid = channels_last.sum(dim=-1) > 0
        classes = channels_last.argmax(dim=-1).long()
        return classes.masked_fill(~valid, -1)

    @staticmethod
    def _srtm_label(label: torch.Tensor) -> torch.Tensor:
        """Return continuous SRTM elevation labels."""
        return PretrainSubsetDataset._squeeze_label(label).float()

    def _get_label(self, args: GetItemArgs) -> torch.Tensor:
        """Load the unnormalized target label for a selected pretrain sample."""
        if self.target_modality is None:
            return torch.tensor(0, dtype=torch.long)
        if self._label_dataset is None:
            raise RuntimeError("Label dataset is not initialized")
        _, label_sample = self._label_dataset[args]
        label = getattr(label_sample, self.target_modality)
        if label is None:
            raise ValueError(f"Target modality {self.target_modality} is missing")
        label = torch.as_tensor(label)
        if self.target_modality == WORLDCOVER_TARGET_MODALITY:
            return self._worldcover_label(label)
        if self.target_modality == OSM_TARGET_MODALITY:
            return self._osm_label(label)
        if self.target_modality == SRTM_TARGET_MODALITY:
            return self._srtm_label(label)
        raise ValueError(
            f"Unsupported pretrain target modality: {self.target_modality}"
        )

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
        return masked, self._get_label(args)

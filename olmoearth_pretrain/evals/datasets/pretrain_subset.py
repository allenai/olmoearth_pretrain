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
from olmoearth_pretrain.evals.metrics import SEGMENTATION_IGNORE_LABEL

logger = logging.getLogger(__name__)

DEFAULT_PATCH_SIZE = 4
DEFAULT_HW_P = 8
DEFAULT_MAX_SAMPLES = 512
WORLDCOVER_CLASSES = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100])
OSM_TARGET_MODALITY = "openstreetmap_raster"
SRTM_TARGET_MODALITY = "srtm"
WORLDCOVER_TARGET_MODALITY = "worldcover"
CDL_TARGET_MODALITY = "cdl"
WORLDCEREAL_TARGET_MODALITY = "worldcereal"
WRI_CANOPY_TARGET_MODALITY = "wri_canopy_height_map"
# WorldCereal channel used for the binary "is annual temporary crops" probe.
WORLDCEREAL_PRIMARY_CHANNEL = 0
# CDL uses 0 to mark no-data / background.
CDL_IGNORE_CODE = 0


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
        split_strategy: str = "random",
        geographic_bin_size_deg: float = 5.0,
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
            if split_strategy == "random":
                selected = self._select_split_indices(
                    total=len(eligible_positions),
                    split=split,
                    seed=label_seed,
                    train_samples=train_samples,
                    valid_samples=valid_samples,
                    test_samples=test_samples,
                )
                self._indices = eligible_positions[selected].tolist()
            elif split_strategy == "geographic":
                self._indices = self._geographic_split_positions(
                    latlons=self._dataset.latlon_distribution,
                    candidate_positions=eligible_positions,
                    split=split,
                    seed=label_seed,
                    train_samples=train_samples,
                    valid_samples=valid_samples,
                    test_samples=test_samples,
                    bin_size_deg=geographic_bin_size_deg,
                ).tolist()
            else:
                raise ValueError(
                    f"Unsupported split_strategy '{split_strategy}'. "
                    f"Expected 'random' or 'geographic'."
                )

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
    def _geographic_split_positions(
        latlons: np.ndarray,
        candidate_positions: np.ndarray,
        split: str,
        seed: int,
        train_samples: int,
        valid_samples: int,
        test_samples: int,
        bin_size_deg: float = 5.0,
        train_frac: float = 0.70,
        valid_frac: float = 0.15,
    ) -> np.ndarray:
        """Pick positions for `split` using a deterministic latlon-bin holdout.

        Bins each candidate sample's lat/lon into `bin_size_deg`-degree cells and
        deterministically assigns whole bins to one of train/valid/test using a
        seeded RNG. This guarantees the test set is geographically disjoint from
        train, exercising spatial generalization.
        """
        if latlons is None:
            raise ValueError(
                "Dataset has no latlon_distribution; geographic split is unavailable."
            )
        split_sizes = {
            "train": train_samples,
            "valid": valid_samples,
            "val": valid_samples,
            "test": test_samples,
        }
        if split not in split_sizes:
            raise ValueError(f"Unsupported split for geographic strategy: {split}")
        if not (0.0 < train_frac < 1.0 and 0.0 < valid_frac < 1.0 - train_frac):
            raise ValueError(
                f"Invalid bucket fractions train={train_frac}, valid={valid_frac}"
            )

        sample_latlons = latlons[candidate_positions]
        lat_bin = np.floor(sample_latlons[:, 0] / bin_size_deg).astype(np.int64)
        lon_bin = np.floor(sample_latlons[:, 1] / bin_size_deg).astype(np.int64)

        unique_bins, inverse = np.unique(
            np.stack([lat_bin, lon_bin], axis=1), axis=0, return_inverse=True
        )
        bin_rng = np.random.RandomState(seed)
        rolls = bin_rng.random(len(unique_bins))
        bucket_for_bin = np.where(
            rolls < train_frac,
            "train",
            np.where(rolls < train_frac + valid_frac, "valid", "test"),
        )
        normalized_split = "valid" if split == "val" else split
        in_split_mask = bucket_for_bin[inverse] == normalized_split

        split_positions = candidate_positions[in_split_mask]
        if split_positions.size == 0:
            raise ValueError(
                f"Geographic split '{split}' produced no samples; try a smaller "
                f"bin_size_deg or check latlon coverage."
            )

        n_target = split_sizes[split]
        if split_positions.size > n_target:
            sample_rng = np.random.RandomState(seed + hash(normalized_split) % 1000)
            split_positions = sample_rng.choice(
                split_positions, size=n_target, replace=False
            )
        return np.asarray(split_positions, dtype=np.int64)

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

    @staticmethod
    def _canopy_label(label: torch.Tensor) -> torch.Tensor:
        """Return continuous WRI canopy height labels (meters)."""
        return PretrainSubsetDataset._squeeze_label(label).float()

    @staticmethod
    def _cdl_label(label: torch.Tensor) -> torch.Tensor:
        """Return CDL class-code labels with no-data pixels marked as ignore."""
        label = PretrainSubsetDataset._squeeze_label(label).long()
        return label.masked_fill(label == CDL_IGNORE_CODE, SEGMENTATION_IGNORE_LABEL)

    @staticmethod
    def _worldcereal_label(label: torch.Tensor) -> torch.Tensor:
        """Binary segmentation label from a single WorldCereal classification channel.

        WorldCereal stores 8 binary channels. The probe currently targets the
        primary annual-temporary-crops channel; pixels with no positive label in
        any channel are treated as no-data.
        """
        label = label.float().squeeze()
        if label.ndim != 3:
            raise ValueError(
                f"Expected WorldCereal label with 3 dims [H, W, C], got {label.shape}"
            )
        if label.shape[0] in (8,) and label.shape[-1] != 8:
            channels_last = label.movedim(0, -1)
        else:
            channels_last = label
        valid = channels_last.sum(dim=-1) > 0
        positive = channels_last[..., WORLDCEREAL_PRIMARY_CHANNEL] > 0
        return positive.long().masked_fill(~valid, SEGMENTATION_IGNORE_LABEL)

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
        if self.target_modality == WRI_CANOPY_TARGET_MODALITY:
            return self._canopy_label(label)
        if self.target_modality == CDL_TARGET_MODALITY:
            return self._cdl_label(label)
        if self.target_modality == WORLDCEREAL_TARGET_MODALITY:
            return self._worldcereal_label(label)
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

"""Eval dataset adapter for pretraining samples and auxiliary map probes.

The dataset returns ``MaskedOlmoEarthSample`` inputs plus either a dummy label
for embedding diagnostics or a spatial target label for pretrain-derived probe
tasks such as WorldCover, OSM, SRTM, CDL, canopy height, and WorldCereal.
"""

from __future__ import annotations

import logging
from functools import cache

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from upath import UPath

from olmoearth_pretrain.data.constants import (
    MISSING_VALUE,
    Modality,
)
from olmoearth_pretrain.data.constants import (
    WORLDCOVER_CLASSES as WORLDCOVER_CLASS_CODES,
)
from olmoearth_pretrain.data.dataset import GetItemArgs, OlmoEarthDataset
from olmoearth_pretrain.datatypes import (
    MaskedOlmoEarthSample,
    MaskValue,
    OlmoEarthSample,
)
from olmoearth_pretrain.evals.metrics import SEGMENTATION_IGNORE_LABEL

logger = logging.getLogger(__name__)

DEFAULT_PATCH_SIZE = 4
DEFAULT_HW_P = 8
DEFAULT_MAX_SAMPLES = 512
WORLDCOVER_CLASSES = torch.tensor(WORLDCOVER_CLASS_CODES)
OSM_TARGET_MODALITY = "openstreetmap_raster"
SRTM_TARGET_MODALITY = "srtm"
WORLDCOVER_TARGET_MODALITY = "worldcover"
CDL_TARGET_MODALITY = "cdl"
WORLDCEREAL_TARGET_MODALITY = "worldcereal"
WRI_CANOPY_TARGET_MODALITY = "wri_canopy_height_map"
GLO30_TARGET_MODALITY = "glo30"
META_CANOPY_TARGET_MODALITY = "meta_canopy_height"
# Band indices into the stored 3-band glo30 target.
GLO30_BAND_ELEVATION = 0
GLO30_BAND_SLOPE = 1
GLO30_BAND_ASPECT = 2
# Virtual band indices for the circular aspect encoding. Aspect is a compass
# bearing, so regressing degrees directly is ill-posed: 359 and 1 are one degree
# apart but land at opposite ends of the target range, and a near-uniform
# bearing distribution makes RMSE/R2 in degrees uninterpretable. Probing sin and
# cos of the bearing instead is continuous across north and bounded to [-1, 1].
GLO30_LABEL_ASPECT_SIN = 3
GLO30_LABEL_ASPECT_COS = 4
# Aspect marks flat pixels with -1; they have no compass bearing at all.
GLO30_ASPECT_FLAT = -1.0
# WorldCereal channel used for the binary "is annual temporary crops" probe.
WORLDCEREAL_PRIMARY_CHANNEL = 0
# CDL uses 0 to mark no-data / background.
CDL_IGNORE_CODE = 0


@cache
def _read_sample_metadata(path: str) -> pd.DataFrame:
    """Cached read of the immutable per-dataset sample-metadata CSV.

    Each probe builds train/valid/test datasets that all reference the same
    metadata file; without this cache the CSV is parsed three times per probe
    on every eval cycle. Callers must not mutate the returned frame.
    """
    return pd.read_csv(path)


class PretrainSubsetDataset(Dataset):
    """Wrap ``OlmoEarthDataset`` for downstream evaluation.

    When ``target_modality`` is unset, this exposes a deterministic random
    subset with dummy labels for embedding diagnostics. When ``target_modality``
    is set, it selects samples where that target is present, creates disjoint
    train/valid/test splits, and returns normalized inputs with unnormalized
    target labels.
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
        target_band_index: int | None = None,
    ) -> None:
        """Initialize a deterministic pretrain eval subset.

        Args:
            h5py_dir: Path to the pretraining HDF5 dataset directory.
            training_modalities: Modalities used as model inputs.
            max_samples: Maximum diagnostic samples when no target modality is set.
            patch_size: Patch size passed through to ``OlmoEarthDataset``.
            hw_p: Spatial patch count passed through to ``OlmoEarthDataset``.
            seed: Random seed for diagnostic subset selection.
            split: Split name for target probes: ``train``, ``valid``/``val``, or
                ``test``.
            target_modality: Optional modality to load as the probe target.
            label_seed: Random seed for target-probe split assignment.
            train_samples: Maximum train samples after split assignment.
            valid_samples: Maximum validation samples after split assignment.
            test_samples: Maximum test samples after split assignment.
            split_strategy: ``random`` for shuffled 80/10/10 sample splits or
                ``geographic`` for shuffled 80/10/10 lat/lon-bin splits.
            geographic_bin_size_deg: Geographic bin size for spatial holdouts.
            target_band_index: For multi-band regression targets (e.g. glo30),
                the band to select for this single-channel probe. ``None`` keeps
                the full target as-is.
        """
        self.patch_size = patch_size
        self.hw_p = hw_p
        self.max_samples = max_samples
        self.target_modality = target_modality
        self.target_band_index = target_band_index

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
            assert self._dataset.sample_indices is not None, (
                "OlmoEarthDataset.prepare() must populate sample_indices."
            )
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
        metadata_df = _read_sample_metadata(str(dataset.sample_metadata_path))
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
        train_frac: float = 0.80,
        valid_frac: float = 0.10,
    ) -> np.ndarray:
        """Pick positions for `split` using a deterministic latlon-bin holdout.

        Bins each candidate sample's lat/lon into `bin_size_deg`-degree cells and
        shuffles the bins once before slicing them 80/10/10 into train/valid/test.
        This guarantees split disjointness while preserving geographic holdouts.
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
        bin_order = bin_rng.permutation(len(unique_bins))
        train_end = int(len(unique_bins) * train_frac)
        valid_end = train_end + int(len(unique_bins) * valid_frac)
        bucket_for_bin = np.full(len(unique_bins), "test", dtype=object)
        bucket_for_bin[bin_order[:train_end]] = "train"
        bucket_for_bin[bin_order[train_end:valid_end]] = "valid"
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
            split_positions = bin_rng.permutation(split_positions)[:n_target]
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
        """Select deterministic disjoint target-probe indices.

        Shuffles the eligible population once with ``seed`` and slices off
        exactly ``train_samples`` / ``valid_samples`` / ``test_samples`` items
        in order. Splits are disjoint by construction; if the eligible pool is
        smaller than the requested totals the trailing splits shrink first.
        """
        if split not in {"train", "valid", "val", "test"}:
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
    def _meta_canopy_label(label: torch.Tensor) -> torch.Tensor:
        """Return continuous Meta canopy height labels (meters)."""
        return PretrainSubsetDataset._squeeze_label(label).float()

    @staticmethod
    def _glo30_label(label: torch.Tensor, band_index: int | None) -> torch.Tensor:
        """Return one continuous GLO-30 DSM target band.

        The glo30 label tensor carries 3 bands in the last axis and the
        single-channel regression head consumes one band per probe, so
        ``band_index`` selects which one before squeezing to [H, W]. Besides the
        three stored bands it accepts the two virtual aspect bands
        (``GLO30_LABEL_ASPECT_SIN``/``_COS``), which encode the compass bearing
        as sin/cos so the target is continuous across north.

        Every aspect-derived label marks flat pixels (aspect ``-1``) as NaN,
        since a flat pixel has no bearing to predict. ``MaskedMSELoss`` and the
        regression target statistics both ignore non-finite pixels, so those
        pixels drop out of training and metrics instead of colliding with due
        north (0 degrees), which is what they would normalize to otherwise.
        """
        if band_index is None:
            raise ValueError(
                "glo30 target requires a target_band_index: "
                f"{GLO30_BAND_ELEVATION}=elevation, {GLO30_BAND_SLOPE}=slope, "
                f"{GLO30_BAND_ASPECT}=aspect (degrees), "
                f"{GLO30_LABEL_ASPECT_SIN}=sin(aspect), "
                f"{GLO30_LABEL_ASPECT_COS}=cos(aspect)"
            )
        if band_index in (GLO30_BAND_ELEVATION, GLO30_BAND_SLOPE):
            band = label[..., band_index].float()
        elif band_index in (
            GLO30_BAND_ASPECT,
            GLO30_LABEL_ASPECT_SIN,
            GLO30_LABEL_ASPECT_COS,
        ):
            aspect = label[..., GLO30_BAND_ASPECT].float()
            if band_index == GLO30_BAND_ASPECT:
                band = aspect
            else:
                radians = torch.deg2rad(aspect)
                band = (
                    torch.sin(radians)
                    if band_index == GLO30_LABEL_ASPECT_SIN
                    else torch.cos(radians)
                )
            band = band.masked_fill(aspect == GLO30_ASPECT_FLAT, float("nan"))
        else:
            raise ValueError(f"Unsupported glo30 target_band_index: {band_index}")
        return PretrainSubsetDataset._squeeze_label(band).float()

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
        if self.target_modality == META_CANOPY_TARGET_MODALITY:
            return self._meta_canopy_label(label)
        if self.target_modality == GLO30_TARGET_MODALITY:
            return self._glo30_label(label, self.target_band_index)
        if self.target_modality == CDL_TARGET_MODALITY:
            return self._cdl_label(label)
        if self.target_modality == WORLDCEREAL_TARGET_MODALITY:
            return self._worldcereal_label(label)
        raise ValueError(
            f"Unsupported pretrain target modality: {self.target_modality}"
        )

    @staticmethod
    def _missing_aware_masked_sample(sample: OlmoEarthSample) -> MaskedOlmoEarthSample:
        """Create ONLINE masks while preserving tokens filled from missing timesteps."""
        masked_sample_dict = {}
        for modality_name, data in sample.as_dict(include_nones=True).items():
            if modality_name == "timestamps":
                masked_sample_dict[modality_name] = data
                continue

            mask_name = MaskedOlmoEarthSample.get_masked_modality_name(modality_name)
            if data is None:
                masked_sample_dict[modality_name] = None
                masked_sample_dict[mask_name] = None
                continue

            tensor = torch.as_tensor(data)
            modality = Modality.get(modality_name)
            mask = torch.full(
                sample.shape(modality_name, mask=True),
                MaskValue.ONLINE_ENCODER.value,
                dtype=torch.long,
            )
            for bandset_idx, band_indices in enumerate(modality.bandsets_as_indices()):
                bandset = tensor[..., band_indices]
                missing = (bandset == MISSING_VALUE).any(dim=-1)
                mask[..., bandset_idx] = torch.where(
                    missing,
                    MaskValue.MISSING.value,
                    mask[..., bandset_idx],
                )

            masked_sample_dict[modality_name] = data
            masked_sample_dict[mask_name] = mask
        return MaskedOlmoEarthSample(**masked_sample_dict)

    def __len__(self) -> int:
        """Return number of samples in the subset."""
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[MaskedOlmoEarthSample, torch.Tensor]:
        """Return a masked input sample and its evaluation label."""
        real_idx = self._indices[idx]
        args = GetItemArgs(
            idx=real_idx,
            patch_size=self.patch_size,
            sampled_hw_p=self.hw_p,
        )
        _, sample = self._dataset[args]
        masked = self._missing_aware_masked_sample(sample)
        return masked, self._get_label(args)

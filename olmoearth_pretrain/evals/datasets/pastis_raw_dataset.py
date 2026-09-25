"""Raw-acquisition PASTIS-format dataset for AnySat.

Every other eval path reads the rslearn export, whose Sentinel-2 axis is twelve
monthly mosaics (``sentinel2_l2a_mo01..mo12``) with the day pinned to the first
of the month. AnySat is designed for *raw* time series -- individual
acquisitions carrying their true dates -- so this dataset reads the PASTIS-format
source directly (``DATA_S2/S2_<id>.npy`` plus the per-acquisition ``dates-S2``
in ``metadata.geojson``), giving ~81-87 dated timesteps per patch instead of 12.

Sequence length varies per patch, so batches are padded to a fixed
``max_timesteps`` and the padded steps are flagged ``MaskValue.MISSING`` in the
modality mask; real steps are ``MaskValue.ONLINE_ENCODER``.
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskValue
from olmoearth_pretrain.evals.metrics import SEGMENTATION_IGNORE_LABEL
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

logger = logging.getLogger(__name__)

# Band order of the 10-band PASTIS Sentinel-2 arrays, as shipped.
PASTIS_S2_BAND_ORDER = [
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B11",
    "B12",
]

# Folds in the PASTIS-format metadata map one-to-one onto the eval splits used
# by the rslearn export (1896 / 411 / 443).
FOLD_TO_SPLIT = {1: "train", 4: "val", 5: "test"}


def _build_band_gather() -> tuple[list[int], list[int]]:
    """Map the PASTIS 10-band axis onto OlmoEarth's 12-band Sentinel-2 order.

    Returns:
        (dst_indices, src_indices) such that
        ``out[..., dst] = pastis[..., src]``. OlmoEarth bands absent from PASTIS
        (B01, B09) are simply never written, so they stay zero and are flagged
        missing in the mask.
    """
    dst, src = [], []
    for pastis_idx, band in enumerate(PASTIS_S2_BAND_ORDER):
        if band in Modality.SENTINEL2_L2A.band_order:
            dst.append(Modality.SENTINEL2_L2A.band_order.index(band))
            src.append(pastis_idx)
    return dst, src


class PastisRawTimeSeriesDataset(Dataset):
    """PASTIS-format raw Sentinel-2 acquisitions with true per-acquisition dates."""

    def __init__(
        self,
        path_to_pastis: str | Path,
        split: str = "train",
        input_modalities: list[str] | None = None,
        window_size: int | None = 16,
        max_timesteps: int | None = None,
        date_range: tuple[int, int] | None = None,
        label_key: str = "ANNOTATIONS",
        label_prefix: str = "TARGET",
        void_value: int = 255,
    ) -> None:
        """Init the raw PASTIS time-series dataset.

        Args:
            path_to_pastis: root of the PASTIS-format export (holds DATA_S2/,
                ANNOTATIONS/ and metadata.geojson).
            split: one of "train", "val", "test"; mapped from the Fold field.
            input_modalities: modalities to load. Only Sentinel-2 is supported
                here, since it is the axis AnySat needs at native cadence.
            window_size: tile each 128x128 patch into window_size x window_size
                windows, matching the rslearn eval convention. None keeps 128.
            max_timesteps: pad/truncate every sequence to this length. If None,
                the maximum sequence length over the split is used.
            date_range: inclusive (YYYYMMDD, YYYYMMDD) bounds; acquisitions
                outside are dropped before padding. None keeps the full
                2018-09..2019-12 export.
            label_key: subdirectory holding the label arrays.
            label_prefix: filename prefix of the label arrays.
            void_value: on-disk value for void pixels; remapped to
                SEGMENTATION_IGNORE_LABEL so loss and metrics skip them.

        Raises:
            ValueError: if the split is unknown or a modality is unsupported.
        """
        # The eval stack uses "valid" for some datasets and "val" for
        # others; normalize so either spelling resolves to Fold 4.
        split = "val" if split == "valid" else split
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")
        self.input_modalities = input_modalities or [Modality.SENTINEL2_L2A.name]
        unsupported = [
            m for m in self.input_modalities if m != Modality.SENTINEL2_L2A.name
        ]
        if unsupported:
            raise ValueError(
                f"PastisRawTimeSeriesDataset supports only "
                f"{Modality.SENTINEL2_L2A.name}; got {unsupported}"
            )

        self.root = Path(path_to_pastis)
        self.split = split
        self.window_size = window_size
        self.label_key = label_key
        self.label_prefix = label_prefix
        self.void_value = void_value

        meta = json.loads((self.root / "metadata.geojson").read_text())
        self.patch_ids: list[int] = []
        self.dates: dict[int, list[int]] = {}
        self.keep_positions: dict[int, list[int]] = {}
        for feat in meta["features"]:
            props = feat["properties"]
            if FOLD_TO_SPLIT.get(props.get("Fold")) != split:
                continue
            pid = int(props["ID_PATCH"])
            raw_dates = props["dates-S2"]
            # dates-S2 is a {"0": YYYYMMDD, ...} map keyed by string position.
            ordered = [int(raw_dates[k]) for k in sorted(raw_dates, key=int)]
            # Keep the original positions alongside the dates so the imagery
            # axis and the timestamps are indexed identically after filtering.
            positions = list(range(len(ordered)))
            if date_range is not None:
                lo, hi = date_range
                pairs = [(i, d) for i, d in zip(positions, ordered) if lo <= d <= hi]
                if not pairs:
                    logger.warning(
                        "patch %d has no acquisitions in %s", pid, date_range
                    )
                    continue
                positions = [i for i, _ in pairs]
                ordered = [d for _, d in pairs]
            self.patch_ids.append(pid)
            self.keep_positions[pid] = positions
            self.dates[pid] = ordered

        if not self.patch_ids:
            raise ValueError(f"no patches found for split {split!r}")

        lengths = [len(self.dates[p]) for p in self.patch_ids]
        self.max_timesteps = max_timesteps or max(lengths)
        logger.info(
            "PASTIS raw %s: %d patches, T in [%d, %d], padding to %d",
            split,
            len(self.patch_ids),
            min(lengths),
            max(lengths),
            self.max_timesteps,
        )

        self._dst_bands, self._src_bands = _build_band_gather()
        self._num_bands = len(Modality.SENTINEL2_L2A.band_order)
        self._tiles_per_side = 1 if window_size is None else 128 // window_size

    def __len__(self) -> int:
        """Number of (patch, tile) pairs in the split."""
        return len(self.patch_ids) * self._tiles_per_side**2

    def _select_indices(self, n_available: int) -> np.ndarray:
        """Pick which acquisitions to keep, preserving the full time span.

        Sequences longer than ``max_timesteps`` are subsampled *uniformly*
        rather than truncated at the head: the raw series runs 2018-09 to
        2019-12, and keeping only the first N would drop the tail of the
        growing season for the longest patches (at max_timesteps=96, 124 train
        patches would have ended at 2019-06 instead of 2019-12).
        """
        if n_available <= self.max_timesteps:
            return np.arange(n_available)
        return np.linspace(0, n_available - 1, self.max_timesteps).round().astype(int)

    def _load_s2(self, pid: int) -> tuple[np.ndarray, int]:
        """Load one patch as (H, W, T_pad, C) plus its true sequence length."""
        arr = np.load(self.root / "DATA_S2" / f"S2_{pid}.npy")  # (T, C, H, W)
        # Restrict to the date-filtered acquisitions, then subsample those.
        positions = np.asarray(self.keep_positions[pid], dtype=int)
        sel = self._select_indices(len(positions))
        keep = positions[sel]
        t_real = len(keep)
        arr = arr[keep].astype(np.float32)
        arr = np.transpose(arr, (2, 3, 0, 1))  # (H, W, T, C_pastis)

        h, w = arr.shape[:2]
        out = np.zeros((h, w, self.max_timesteps, self._num_bands), dtype=np.float32)
        out[:, :, :t_real, self._dst_bands] = arr[:, :, :, self._src_bands]
        return out, t_real

    def _build_timestamps(self, pid: int) -> torch.Tensor:
        """Per-acquisition [day, month0, year], padded to max_timesteps."""
        ts = torch.zeros((self.max_timesteps, 3), dtype=torch.long)
        all_dates = self.dates[pid]
        dates = [all_dates[i] for i in self._select_indices(len(all_dates))]
        for i, yyyymmdd in enumerate(dates):
            year, month, day = (
                yyyymmdd // 10000,
                (yyyymmdd // 100) % 100,
                yyyymmdd % 100,
            )
            # OlmoEarth stores month 0-indexed; see MaskedOlmoEarthSample.
            ts[i] = torch.tensor([day, month - 1, year], dtype=torch.long)
        if dates:
            # Padded steps repeat the final real date so downstream date
            # encoders never see a year-0 value; the mask marks them missing.
            ts[len(dates) :] = ts[len(dates) - 1]
        return ts

    def __getitem__(self, idx: int) -> tuple[MaskedOlmoEarthSample, torch.Tensor]:
        """Return one (sample, label) pair for the given tile index."""
        base_pos, tile = divmod(idx, self._tiles_per_side**2)
        pid = self.patch_ids[base_pos]

        if self.window_size is None:
            rows = cols = slice(None)
        else:
            tile_row, tile_col = divmod(tile, self._tiles_per_side)
            rows = slice(tile_row * self.window_size, (tile_row + 1) * self.window_size)
            cols = slice(tile_col * self.window_size, (tile_col + 1) * self.window_size)

        s2, t_real = self._load_s2(pid)
        s2_t = torch.from_numpy(s2[rows, cols]).float()

        labels = np.load(self.root / self.label_key / f"{self.label_prefix}_{pid}.npy")
        if labels.ndim == 3:
            labels = labels[0]
        labels_t = torch.from_numpy(np.asarray(labels)[rows, cols]).long()
        # On disk void is 255 (the rslearn nodata convention). The eval stack
        # expects SEGMENTATION_IGNORE_LABEL (-1), which cross_entropy and the
        # mIoU metric both skip; leaving 255 would be read as a class index.
        labels_t = labels_t.masked_fill(
            labels_t == self.void_value, SEGMENTATION_IGNORE_LABEL
        )

        # Valid acquisitions are visible to the encoder; padded steps and the
        # two OlmoEarth bands PASTIS does not carry are flagged missing.
        mask = torch.full(s2_t.shape, float(MaskValue.ONLINE_ENCODER.value))
        mask[:, :, t_real:, :] = MaskValue.MISSING.value
        absent = [i for i in range(self._num_bands) if i not in self._dst_bands]
        if absent:
            mask[:, :, :, absent] = MaskValue.MISSING.value

        sample = MaskedOlmoEarthSample(
            timestamps=self._build_timestamps(pid),
            sentinel2_l2a=s2_t,
            sentinel2_l2a_mask=mask,
        )
        return sample, labels_t

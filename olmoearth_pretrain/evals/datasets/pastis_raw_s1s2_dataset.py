"""Raw-acquisition S1+S2 PASTIS dataset for the Tessera live encoder.

Deliberately separate from pastis_raw_dataset.PastisRawTimeSeriesDataset (the
AnySat S2-only view): Tessera needs both modalities, and the two sensors have
different acquisition dates and counts (2019 medians: ~99 S1 vs ~69 S2), so the
time axis has to be built differently. Nothing here is shared with the AnySat
path, and no other task reads this class.

MaskedOlmoEarthSample carries ONE timestamps tensor for all modalities, so the
axis is the *union* of the S1 and S2 acquisition dates. Each slot's timestamp is
therefore the true date of whichever sensor observed on that date; the sensor
that did not observe is zero-filled and flagged MaskValue.MISSING. Sequences
longer than max_timesteps are subsampled uniformly so the full year is spanned
rather than truncated at the head.
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

# Band order of the shipped PASTIS arrays.
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
PASTIS_S1_BAND_ORDER = ["vv", "vh"]

FOLD_TO_SPLIT = {1: "train", 4: "val", 5: "test"}


def _gather(
    pastis_order: list[str], olmoearth_order: list[str]
) -> tuple[list[int], list[int]]:
    """Map a PASTIS band axis onto the OlmoEarth band order.

    Args:
        pastis_order: band names as stored in the PASTIS arrays.
        olmoearth_order: the target Modality band_order.

    Returns:
        (dst_indices, src_indices) such that out[..., dst] = pastis[..., src].
    """
    dst, src = [], []
    for i, band in enumerate(pastis_order):
        if band in olmoearth_order:
            dst.append(olmoearth_order.index(band))
            src.append(i)
    return dst, src


class PastisRawS1S2Dataset(Dataset):
    """Raw dated S1+S2 acquisitions on a shared union time axis."""

    def __init__(
        self,
        path_to_pastis: str | Path,
        split: str = "train",
        input_modalities: list[str] | None = None,
        window_size: int | None = 16,
        max_timesteps: int = 120,
        date_range: tuple[int, int] = (20190101, 20191231),
        label_key: str = "ANNOTATIONS",
        label_prefix: str = "TARGET",
        void_value: int = 255,
    ) -> None:
        """Init the raw S1+S2 dataset.

        Args:
            path_to_pastis: root of the PASTIS-format export.
            split: "train", "val"/"valid", or "test" (mapped from Fold).
            input_modalities: must be S1 and/or S2; defaults to both.
            window_size: tile each 128x128 patch into window_size windows.
            max_timesteps: cap on the union axis; longer sequences are
                subsampled uniformly so the full span is preserved. Defaults to
                120; the 2019 union median is ~142, so the cap trims the tail:
                at 96 it dropped a median patch from ~66 S2 / ~99 S1 native
                acquisitions down to ~39 S2 / ~66 S1.
            date_range: inclusive (YYYYMMDD, YYYYMMDD) bounds.
            label_key: subdirectory holding the label arrays.
            label_prefix: filename prefix of the label arrays.
            void_value: on-disk void value, remapped to the ignore label.

        Raises:
            ValueError: on an unknown split or unsupported modality.
        """
        split = "val" if split == "valid" else split
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")
        self.input_modalities = input_modalities or [
            Modality.SENTINEL1.name,
            Modality.SENTINEL2_L2A.name,
        ]
        allowed = {Modality.SENTINEL1.name, Modality.SENTINEL2_L2A.name}
        bad = [m for m in self.input_modalities if m not in allowed]
        if bad:
            raise ValueError(f"PastisRawS1S2Dataset supports only {allowed}; got {bad}")

        self.root = Path(path_to_pastis)
        self.split = split
        self.window_size = window_size
        self.max_timesteps = max_timesteps
        self.label_key = label_key
        self.label_prefix = label_prefix
        self.void_value = void_value

        self._s2_dst, self._s2_src = _gather(
            PASTIS_S2_BAND_ORDER, list(Modality.SENTINEL2_L2A.band_order)
        )
        self._s1_dst, self._s1_src = _gather(
            PASTIS_S1_BAND_ORDER, list(Modality.SENTINEL1.band_order)
        )
        self._n_s2 = len(Modality.SENTINEL2_L2A.band_order)
        self._n_s1 = len(Modality.SENTINEL1.band_order)

        meta = json.loads((self.root / "metadata.geojson").read_text())
        lo, hi = date_range
        self.patch_ids: list[int] = []
        # per patch: date -> (s2_index | None, s1_index | None)
        self.axis: dict[int, list[tuple[int, int | None, int | None]]] = {}
        for feat in meta["features"]:
            props = feat["properties"]
            if FOLD_TO_SPLIT.get(props.get("Fold")) != split:
                continue
            pid = int(props["ID_PATCH"])
            s2 = {
                int(v): i
                for i, v in enumerate(
                    [props["dates-S2"][k] for k in sorted(props["dates-S2"], key=int)]
                )
                if lo <= int(v) <= hi
            }
            s1 = {
                int(v): i
                for i, v in enumerate(
                    [props["dates-S1"][k] for k in sorted(props["dates-S1"], key=int)]
                )
                if lo <= int(v) <= hi
            }
            dates = sorted(set(s2) | set(s1))
            if not dates:
                logger.warning("patch %d has no acquisitions in %s", pid, date_range)
                continue
            self.patch_ids.append(pid)
            self.axis[pid] = [(d, s2.get(d), s1.get(d)) for d in dates]

        if not self.patch_ids:
            raise ValueError(f"no patches found for split {split!r}")
        lens = [len(self.axis[p]) for p in self.patch_ids]
        logger.info(
            "PASTIS raw S1+S2 %s: %d patches, union T in [%d, %d], cap %d",
            split,
            len(self.patch_ids),
            min(lens),
            max(lens),
            max_timesteps,
        )
        self._tiles_per_side = 1 if window_size is None else 128 // window_size

    def __len__(self) -> int:
        """Number of (patch, tile) pairs."""
        return len(self.patch_ids) * self._tiles_per_side**2

    def _select(self, n: int) -> np.ndarray:
        """Uniformly subsample n union slots down to max_timesteps."""
        if n <= self.max_timesteps:
            return np.arange(n)
        return np.linspace(0, n - 1, self.max_timesteps).round().astype(int)

    def __getitem__(self, idx: int) -> tuple[MaskedOlmoEarthSample, torch.Tensor]:
        """Return one (sample, label) pair."""
        base, tile = divmod(idx, self._tiles_per_side**2)
        pid = self.patch_ids[base]
        if self.window_size is None:
            rows = cols = slice(None)
        else:
            tr, tc = divmod(tile, self._tiles_per_side)
            rows = slice(tr * self.window_size, (tr + 1) * self.window_size)
            cols = slice(tc * self.window_size, (tc + 1) * self.window_size)

        slots = [self.axis[pid][i] for i in self._select(len(self.axis[pid]))]
        T = self.max_timesteps
        hw = self.window_size or 128

        ts = torch.zeros((T, 3), dtype=torch.long)
        for i, (d, _, _) in enumerate(slots):
            y, m, dd = d // 10000, (d // 100) % 100, d % 100
            ts[i] = torch.tensor([dd, m - 1, y], dtype=torch.long)  # month 0-indexed
        if slots:
            ts[len(slots) :] = ts[len(slots) - 1]

        sample: dict[str, torch.Tensor] = {"timestamps": ts}

        if Modality.SENTINEL2_L2A.name in self.input_modalities:
            raw = np.load(self.root / "DATA_S2" / f"S2_{pid}.npy", mmap_mode="r")
            out = np.zeros((hw, hw, T, self._n_s2), dtype=np.float32)
            mask = torch.full((hw, hw, T, self._n_s2), float(MaskValue.MISSING.value))
            for i, (_, s2i, _) in enumerate(slots):
                if s2i is None:
                    continue
                fr = np.asarray(raw[s2i], dtype=np.float32)[:, rows, cols]  # (C,h,w)
                out[:, :, i, self._s2_dst] = np.transpose(fr, (1, 2, 0))[
                    :, :, self._s2_src
                ]
                mask[:, :, i, self._s2_dst] = MaskValue.ONLINE_ENCODER.value
            sample[Modality.SENTINEL2_L2A.name] = torch.from_numpy(out)
            sample[Modality.SENTINEL2_L2A.name + "_mask"] = mask

        if Modality.SENTINEL1.name in self.input_modalities:
            raw = np.load(self.root / "DATA_S1" / f"S1_{pid}.npy", mmap_mode="r")
            out = np.zeros((hw, hw, T, self._n_s1), dtype=np.float32)
            mask = torch.full((hw, hw, T, self._n_s1), float(MaskValue.MISSING.value))
            for i, (_, _, s1i) in enumerate(slots):
                if s1i is None:
                    continue
                fr = np.asarray(raw[s1i], dtype=np.float32)[:, rows, cols]
                out[:, :, i, self._s1_dst] = np.transpose(fr, (1, 2, 0))[
                    :, :, self._s1_src
                ]
                mask[:, :, i, self._s1_dst] = MaskValue.ONLINE_ENCODER.value
            sample[Modality.SENTINEL1.name] = torch.from_numpy(out)
            sample[Modality.SENTINEL1.name + "_mask"] = mask

        labels = np.load(self.root / self.label_key / f"{self.label_prefix}_{pid}.npy")
        if labels.ndim == 3:
            labels = labels[0]
        y = torch.from_numpy(np.asarray(labels)[rows, cols]).long()
        y = y.masked_fill(y == self.void_value, SEGMENTATION_IGNORE_LABEL)

        return MaskedOlmoEarthSample(**sample), y

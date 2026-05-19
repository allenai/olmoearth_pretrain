"""Custom dataloaders for GeoBench v2, replacing the GeoBenchV2 package dependency.

Each dataset class reads directly from .tortilla files via tacoreader + rasterio/h5py
and returns raw (un-normalized) tensors in the same dict-key format expected by
_sample_to_olmoearth / _extract_label in geobench_v2_dataset.py.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import rasterio
import tacoreader
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _raster_f32(row: Any, idx: int) -> torch.Tensor:
    """Read rasterio subfile → float32 (C, H, W)."""
    with rasterio.open(row.read(idx)) as src:
        return torch.from_numpy(src.read().astype(np.float32))


def _raster_i64(row: Any, idx: int) -> torch.Tensor:
    """Read rasterio subfile → int64 (C, H, W)."""
    with rasterio.open(row.read(idx)) as src:
        return torch.from_numpy(src.read().astype(np.int64))


def _resize(x: torch.Tensor, size: int, mode: str = "bilinear") -> torch.Tensor:
    """Resize a (C, H, W) tensor to (size, size)."""
    return F.interpolate(
        x.unsqueeze(0),
        size=(size, size),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    ).squeeze(0)


class _BaseGeobenchDataset(Dataset):
    """Loads a .tortilla index, filters by split, provides helpers."""

    TORTILLA: list[str]
    band_order: Any  # consumed by _sample_to_olmoearth

    def __init__(self, root: str, split: str) -> None:
        """Load tortilla file(s) and filter rows to the requested split."""
        names = list(self.TORTILLA)
        paths = [os.path.join(root, n) for n in names]
        df = tacoreader.load(paths)
        if split in ("val", "valid"):
            split = "validation"
        self._df = df[df["tortilla:data_split"] == split].reset_index(drop=True)

    def __len__(self) -> int:
        """Return number of samples in the split."""
        return len(self._df)


class BurnScarsDataset(_BaseGeobenchDataset):
    """HLS Burn Scars: 6-band S2 → segmentation (0=bg, 1=burn, 2=no-data)."""

    TORTILLA = ["geobench_burn_scars.tortilla"]
    band_order = {"s2": ["B02", "B03", "B04", "B8A", "B11", "B12"]}

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Load sample at idx."""
        row = self._df.read(idx)
        image = _raster_f32(row, 0)
        mask = _raster_i64(row, 1).squeeze(0)
        mask = mask.masked_fill(mask == -1, 2)
        return {"image": image, "mask": mask}


class CaFFeDataset(_BaseGeobenchDataset):
    """Calving Front: grayscale → segmentation (4 classes)."""

    TORTILLA = ["geobench_caffe.tortilla"]
    band_order = ["gray"]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Load sample at idx."""
        row = self._df.read(idx)
        image = _raster_f32(row, 0)
        mask = _raster_i64(row, 1).squeeze(0)
        return {"image": image, "mask": mask}


class CloudSen12Dataset(_BaseGeobenchDataset):
    """CloudSen12: 12-band S2 → cloud segmentation (4 classes)."""

    TORTILLA = ["geobench_cloudsen12.tortilla"]
    # tortilla has 14 bands (12 S2 + B10 cirrus + cloud prob); truncation to 12 happens in _sample_to_olmoearth
    band_order = {
        "s2": [
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B09",
            "B11",
            "B12",
        ]
    }

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Load sample at idx."""
        row = self._df.read(idx)
        image = _raster_f32(row, 0)  # (14, H, W)
        mask = _raster_i64(row, 1).squeeze(0)
        return {"image": image, "mask": mask}


class SpaceNet7Dataset(_BaseGeobenchDataset):
    """SpaceNet7: 4-band PlanetScope (RGBN) → building segmentation."""

    TORTILLA = ["geobench_spacenet7.tortilla"]
    band_order = {"s2": ["red", "green", "blue", "nir"]}

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Load sample at idx."""
        row = self._df.read(idx)
        image = _raster_f32(row, 0)
        mask = _raster_i64(row, 1).squeeze(0) + 1  # shift: background → class 1
        return {"image": image, "mask": mask}


class SpaceNet2Dataset(_BaseGeobenchDataset):
    """SpaceNet2: WorldView-2 (8-band) + panchromatic (1-band) → building seg."""

    TORTILLA = ["geobench_spacenet2.tortilla"]
    band_order = {
        "worldview": [
            "coastal",
            "blue",
            "green",
            "yellow",
            "red",
            "red_edge",
            "nir1",
            "nir2",
        ],
        "pan": ["pan"],
    }
    TARGET_SIZE = 512  # raw tiles are 650×650; resize to match config height_width

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Load sample at idx."""
        row = self._df.read(idx)
        image_worldview = _raster_f32(row, 0)
        image_pan = _raster_f32(row, 1)
        mask = _raster_i64(row, 2).squeeze(0) + 1  # shift: background → class 1

        if image_worldview.shape[-1] != self.TARGET_SIZE:
            image_worldview = _resize(image_worldview, self.TARGET_SIZE)
            image_pan = _resize(image_pan, self.TARGET_SIZE)
            mask = (
                _resize(mask.unsqueeze(0).float(), self.TARGET_SIZE, mode="nearest")
                .squeeze(0)
                .long()
            )

        return {
            "image_worldview": image_worldview,
            "image_pan": image_pan,
            "mask": mask,
        }


_BENV2_LABELS = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland, shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
]
_BENV2_L2I = {c: i for i, c in enumerate(_BENV2_LABELS)}


class BENV2Dataset(_BaseGeobenchDataset):
    """BigEarthNet V2: S1 (2-band) + S2 (12-band) → 19-class multi-label."""

    TORTILLA = ["geobench_benv2.tortilla"]
    band_order = {
        "s1": ["VV", "VH"],
        "s2": [
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B09",
            "B11",
            "B12",
        ],
    }

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Load sample at idx."""
        row_df = self._df.iloc[idx]
        row = self._df.read(idx)
        image_s1 = _raster_f32(row, 0)
        image_s2 = _raster_f32(row, 1)
        label = torch.zeros(len(_BENV2_LABELS), dtype=torch.long)
        for name in row_df["labels"]:
            label[_BENV2_L2I[name]] = 1
        return {"image_s1": image_s1, "image_s2": image_s2, "label": label}


# No Water→0, Permanent Water→1, Flood→2, No Data→-1 (ignored)
_KURO_SIWO_CLASS_MAP = torch.tensor([0, 1, 2, -1])


class KuroSiwoDataset(_BaseGeobenchDataset):
    """KuroSiwo: SAR (pre/post) + DEM → flood segmentation (3 classes + ignore)."""

    TORTILLA = ["geobench_kuro_siwo.tortilla"]
    band_order = {"sar": ["vv", "vh"], "dem": ["dem"]}

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Load sample at idx."""
        row = self._df.read(idx)
        # items: [0]=pre_1 SAR, [1]=pre_2 SAR, [2]=post SAR, [3]=DEM, [4]=mask
        image_pre_1 = _raster_f32(row, 0).nan_to_num(0.0)
        image_pre_2 = _raster_f32(row, 1).nan_to_num(0.0)
        image_post = _raster_f32(row, 2).nan_to_num(0.0)
        image_dem = _raster_f32(row, 3)
        mask = _KURO_SIWO_CLASS_MAP[_raster_i64(row, 4).squeeze(0)]
        return {
            "image_pre_1": image_pre_1,
            "image_pre_2": image_pre_2,
            "image_post": image_post,
            "image_dem": image_dem,
            "mask": mask,
        }


class BioMasstersDataset(_BaseGeobenchDataset):
    """BioMassters: S1 (4-band) + S2 (10-band), 1 time step → AGB regression."""

    TORTILLA = [
        "geobench_biomassters.0000.part.tortilla",
        "geobench_biomassters.0001.part.tortilla",
        "geobench_biomassters.0002.part.tortilla",
        "geobench_biomassters.0003.part.tortilla",
        "geobench_biomassters.0004.part.tortilla",
        "geobench_biomassters.0005.part.tortilla",
        "geobench_biomassters.0006.part.tortilla",
    ]
    band_order = {
        "s1": ["VV_asc", "VH_asc", "VV_desc", "VH_desc"],
        "s2": ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
    }

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Load sample at idx."""
        row = self._df.read(idx)
        s1_idx = row[row["modality"] == "S1"].index
        s2_idx = row[row["modality"] == "S2"].index
        agbm_idx = row[row["modality"] == "AGBM"].index[0]
        image_s1 = _raster_f32(row, s1_idx[0]).nan_to_num(0.0)
        image_s1[image_s1 == -9999] = 0.0
        image_s2 = _raster_f32(row, s2_idx[0])[:10]  # file may have 11 bands; keep 10
        mask = _raster_f32(row, agbm_idx).squeeze(0)
        return {"image_s1": image_s1, "image_s2": image_s2, "mask": mask}


SLUG_TO_DATASET: dict[str, type[_BaseGeobenchDataset]] = {
    "benv2": BENV2Dataset,
    "biomassters": BioMasstersDataset,
    "burn_scars": BurnScarsDataset,
    "caffe": CaFFeDataset,
    "cloudsen12": CloudSen12Dataset,
    "kuro_siwo": KuroSiwoDataset,
    "spacenet2": SpaceNet2Dataset,
    "spacenet7": SpaceNet7Dataset,
}

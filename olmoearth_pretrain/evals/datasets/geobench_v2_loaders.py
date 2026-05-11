"""Custom dataloaders for GeoBench v2, replacing the GeoBenchV2 package dependency.

Each dataset class reads directly from .tortilla files via tacoreader + rasterio/h5py
and returns raw (un-normalized) tensors in the same dict-key format expected by
_sample_to_olmoearth / _extract_label in geobench_v2_dataset.py.
"""

from __future__ import annotations

import io
import json
import os
from typing import Any

import h5py
import numpy as np
import rasterio
import tacoreader
import torch
from torch.utils.data import Dataset

# ─── helpers ──────────────────────────────────────────────────────────────────


def _read_subfile_bytes(path: str) -> bytes:
    """Extract raw bytes from a /vsisubfile/offset_length,file path."""
    vsi, fpath = path.split(",", 1)
    offset, length = (int(x) for x in vsi.replace("/vsisubfile/", "").split("_"))
    with open(fpath, "rb") as f:
        f.seek(offset)
        return f.read(length)


def _raster_f32(row: Any, idx: int) -> torch.Tensor:
    """Read rasterio subfile → float32 (C, H, W)."""
    with rasterio.open(row.read(idx)) as src:
        return torch.from_numpy(src.read().astype(np.float32))


def _raster_i64(row: Any, idx: int) -> torch.Tensor:
    """Read rasterio subfile → int64 (C, H, W)."""
    with rasterio.open(row.read(idx)) as src:
        return torch.from_numpy(src.read().astype(np.int64))


def _polygon_to_mask(
    vertices: list[float], width: int = 228, height: int = 228
) -> np.ndarray:
    """Convert flat polygon vertex list to binary uint8 mask."""
    from PIL import Image, ImageDraw

    img = Image.new("L", (width, height), 0)
    points = [(vertices[i], vertices[i + 1]) for i in range(0, len(vertices), 2)]
    if len(points) >= 3:
        ImageDraw.Draw(img).polygon(points, outline=1, fill=1)
    return np.array(img, dtype=np.uint8)


def _load_json_annotations(
    row: Any, idx: int, img_w: int, img_h: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Parse h5py JSON annotation (substation/nzcattle style).

    Returns (boxes (N,4) float32 xyxy, labels (N,) int64, masks (N,H,W) int64).
    """
    data = _read_subfile_bytes(row.read(idx))
    with h5py.File(io.BytesIO(data)) as hf:
        annotations = json.loads(hf.attrs["annotation"])
    items = annotations.get("sample_annotations", annotations.get("boxes", []))
    boxes, labels, masks = [], [], []
    for anno in items:
        cat = anno.get("category_id", 1)
        if "bbox" in anno:
            x, y, w, h = anno["bbox"]
            boxes.append([x, y, x + w, y + h])
        else:
            boxes.append([0.0, 0.0, float(img_w), float(img_h)])
        labels.append(cat)
        if "mask" in anno:
            masks.append(_polygon_to_mask(anno["mask"][0], img_w, img_h))
        else:
            masks.append(np.zeros((img_h, img_w), dtype=np.uint8))
    if boxes:
        return (
            torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.int64),
            torch.from_numpy(np.stack(masks)).long(),
        )
    # empty: no instances
    return (
        torch.zeros((0, 4), dtype=torch.float32),
        torch.zeros((0,), dtype=torch.int64),
        torch.zeros((0, img_h, img_w), dtype=torch.int64),
    )


# ─── base class ───────────────────────────────────────────────────────────────


class _BaseGeobenchDataset(Dataset):
    """Loads a .tortilla index, filters by split, provides helpers."""

    TORTILLA: str | list[str]
    band_order: Any  # consumed by _sample_to_olmoearth

    def __init__(self, root: str, split: str) -> None:
        names = (
            [self.TORTILLA] if isinstance(self.TORTILLA, str) else list(self.TORTILLA)
        )
        paths = [os.path.join(root, n) for n in names]
        df = tacoreader.load(paths)
        if split in ("val", "valid"):
            split = "validation"
        self._df = df[df["tortilla:data_split"] == split].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self._df)


# ─── S2 segmentation ──────────────────────────────────────────────────────────


class BurnScarsDataset(_BaseGeobenchDataset):
    """HLS Burn Scars: 6-band S2 → segmentation (0=bg, 1=burn, 2=no-data)."""

    TORTILLA = "geobench_burn_scars.tortilla"
    band_order = ["B02", "B03", "B04", "B8A", "B11", "B12"]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # noqa: D105
        row = self._df.read(idx)
        image = _raster_f32(row, 0)  # (6, H, W)
        mask = _raster_i64(row, 1).squeeze(0)  # (H, W)
        mask = mask.masked_fill(mask == -1, 2)
        return {"image": image, "mask": mask}


class CaFFeDataset(_BaseGeobenchDataset):
    """Calving Front: grayscale → segmentation (4 classes)."""

    TORTILLA = "geobench_caffe.tortilla"
    band_order = ["gray"]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # noqa: D105
        row = self._df.read(idx)
        image = _raster_f32(row, 0)  # (1, H, W)
        mask = _raster_i64(row, 1).squeeze(0)  # (H, W)
        return {"image": image, "mask": mask}


class CloudSen12Dataset(_BaseGeobenchDataset):
    """CloudSen12: 12-band S2 → cloud segmentation (4 classes)."""

    TORTILLA = "geobench_cloudsen12.tortilla"
    band_order = [
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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # noqa: D105
        row = self._df.read(idx)
        image = _raster_f32(
            row, 0
        )  # (14, H, W) — 12 S2 bands + B10 cirrus + cloud prob
        mask = _raster_i64(row, 1).squeeze(0)  # (H, W)
        return {"image": image, "mask": mask}


class SpaceNet7Dataset(_BaseGeobenchDataset):
    """SpaceNet7: 4-band PlanetScope (RGBN) → building segmentation (+1 class offset)."""

    TORTILLA = "geobench_spacenet7.tortilla"
    band_order = ["red", "green", "blue", "nir"]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # noqa: D105
        row = self._df.read(idx)
        image = _raster_f32(row, 0)  # (3, H, W)
        mask = _raster_i64(row, 1).squeeze(0)  # (H, W)
        mask = mask + 1  # shift: 0 → background becomes class 1
        return {"image": image, "mask": mask}


class SpaceNet2Dataset(_BaseGeobenchDataset):
    """SpaceNet2: WorldView-2 (8-band) + panchromatic (1-band) → building seg."""

    TORTILLA = "geobench_spacenet2.tortilla"
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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # noqa: D105
        import torch.nn.functional as F

        row = self._df.read(idx)
        image_worldview = _raster_f32(row, 0)  # (8, H, W)
        image_pan = _raster_f32(row, 1)  # (1, H, W)
        mask = _raster_i64(row, 2).squeeze(0)  # (H, W)
        mask = mask + 1  # shift for background class

        if image_worldview.shape[-1] != self.TARGET_SIZE:
            image_worldview = F.interpolate(
                image_worldview.unsqueeze(0),
                size=(self.TARGET_SIZE, self.TARGET_SIZE),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            image_pan = F.interpolate(
                image_pan.unsqueeze(0),
                size=(self.TARGET_SIZE, self.TARGET_SIZE),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            mask = (
                F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0).float(),
                    size=(self.TARGET_SIZE, self.TARGET_SIZE),
                    mode="nearest",
                )
                .squeeze()
                .long()
            )

        return {
            "image_worldview": image_worldview,
            "image_pan": image_pan,
            "mask": mask,
        }


# ─── multi-label classification ───────────────────────────────────────────────

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

_TREESATAI_CLASSES = [
    "Abies",
    "Acer",
    "Alnus",
    "Betula",
    "Cleared",
    "Fagus",
    "Fraxinus",
    "Larix",
    "Picea",
    "Pinus",
    "Populus",
    "Prunus",
    "Pseudotsuga",
    "Quercus",
    "Tilia",
]
_TREESATAI_L2I = {c: i for i, c in enumerate(_TREESATAI_CLASSES)}


class BENV2Dataset(_BaseGeobenchDataset):
    """BigEarthNet V2: S1 (2-band) + S2 (12-band) → 19-class multi-label."""

    TORTILLA = "geobench_benv2.tortilla"
    # item[0]=S1(VV,VH) float32, item[1]=S2(12-band) uint16
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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # noqa: D105
        row_df = self._df.iloc[idx]
        row = self._df.read(idx)
        image_s1 = _raster_f32(row, 0)  # (2, 120, 120)
        image_s2 = _raster_f32(row, 1)  # (12, 120, 120) cast from uint16
        label_names: list[str] = row_df["labels"]
        label = torch.zeros(len(_BENV2_LABELS), dtype=torch.long)
        for name in label_names:
            label[_BENV2_L2I[name]] = 1
        return {"image_s1": image_s1, "image_s2": image_s2, "label": label}


class TreeSatAIDataset(_BaseGeobenchDataset):
    """TreeSatAI: aerial (4-band) + S1 (3-band) + S2 (12-band) → 15-class multi-label."""

    TORTILLA = "geobench_treesatai.tortilla"
    # item[0]=aerial(4), item[1]=s1(3), item[2]=s2(12)
    band_order = {
        "aerial": ["red", "green", "blue", "nir"],
        "s1": ["vv", "vh", "vv/vh"],
        "s2": [
            "B02",
            "B03",
            "B04",
            "B08",
            "B05",
            "B06",
            "B07",
            "B8A",
            "B11",
            "B12",
            "B01",
            "B09",
        ],
    }

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # noqa: D105
        row_df = self._df.iloc[idx]
        row = self._df.read(idx)
        image_aerial = _raster_f32(row, 0)  # (4, H, W)
        image_s1 = _raster_f32(row, 1)  # (3, H, W)
        image_s2 = _raster_f32(row, 2)  # (12, H, W)
        label_names: list[str] = row_df["species_labels"]
        label = torch.zeros(len(_TREESATAI_CLASSES), dtype=torch.long)
        for name in label_names:
            if name in _TREESATAI_L2I:
                label[_TREESATAI_L2I[name]] = 1
        return {
            "image_aerial": image_aerial,
            "image_s1": image_s1,
            "image_s2": image_s2,
            "label": label,
        }


# ─── KuroSiwo ─────────────────────────────────────────────────────────────────

_KURO_SIWO_CLASS_MAP = {
    0: 1,
    1: 2,
    2: 3,
    3: 0,
}  # No Water→1, Perm Water→2, Flood→3, No Data→0


class KuroSiwoDataset(_BaseGeobenchDataset):
    """KuroSiwo: SAR (pre/post) + DEM → 4-class flood segmentation."""

    TORTILLA = "geobench_kuro_siwo.tortilla"
    band_order = {"sar": ["vv", "vh"], "dem": ["dem"]}

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # noqa: D105
        row = self._df.read(idx)
        # items: [0]=pre_1 SAR, [1]=pre_2 SAR, [2]=post SAR, [3]=DEM, [4]=mask, [5]=invalid
        image_pre_1 = _raster_f32(row, 0)  # (2, 224, 224)
        image_pre_2 = _raster_f32(row, 1)  # (2, 224, 224)
        image_post = _raster_f32(row, 2)  # (2, 224, 224)
        image_dem = _raster_f32(row, 3)  # (1, 224, 224)
        invalid = _raster_i64(row, 5)  # (1, 224, 224)

        raw_mask = _raster_i64(row, 4).squeeze(0)
        lookup = torch.tensor(
            [_KURO_SIWO_CLASS_MAP[i] for i in range(len(_KURO_SIWO_CLASS_MAP))]
        )
        remapped = lookup[raw_mask]

        # NaN handling for SAR
        for img in (image_pre_1, image_pre_2, image_post):
            img[torch.isnan(img)] = 0.0

        return {
            "image_pre_1": image_pre_1,
            "image_pre_2": image_pre_2,
            "image_post": image_post,
            "image_dem": image_dem,
            "mask": remapped,
            "invalid_data": invalid,
        }


# ─── Substation ───────────────────────────────────────────────────────────────


class SubstationDataset(_BaseGeobenchDataset):
    """Substation: 13-band S2 → binary presence classification (via bbox)."""

    TORTILLA = "geobench_substation.tortilla"
    band_order = [
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
        "B10",
        "B11",
        "B12",
    ]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # noqa: D105
        row = self._df.read(idx)
        image = _raster_f32(row, 0)  # (13, 228, 228)
        _, h, w = image.shape
        boxes, labels, masks = _load_json_annotations(row, 1, w, h)
        return {"image": image, "bbox_xyxy": boxes, "label": labels, "mask": masks}


# ─── Timeseries tasks ─────────────────────────────────────────────────────────


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
    # File channel order per DatasetBandRegistry
    band_order = {
        "s1": ["VV_asc", "VH_asc", "VV_desc", "VH_desc"],
        "s2": ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
    }
    _NUM_S2_BANDS = 10  # registry has 10 S2 bands; file may have 11

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # noqa: D105
        row = self._df.read(idx)
        s1_idx = row[row["modality"] == "S1"].index
        s2_idx = row[row["modality"] == "S2"].index
        agbm_idx = row[row["modality"] == "AGBM"].index[0]

        # Use first available time step for each modality (num_time_steps=1)
        image_s1 = _raster_f32(row, s1_idx[0])  # (4, H, W)
        image_s1[image_s1 == -9999] = 0.0

        # Clip to first _NUM_S2_BANDS channels in case the file has extras
        image_s2 = _raster_f32(row, s2_idx[0])[: self._NUM_S2_BANDS]  # (10, H, W)

        mask = _raster_f32(row, agbm_idx).squeeze(0)  # (H, W)

        return {"image_s1": image_s1, "image_s2": image_s2, "mask": mask}


# ─── registry ─────────────────────────────────────────────────────────────────

SLUG_TO_DATASET: dict[str, type[_BaseGeobenchDataset]] = {
    "benv2": BENV2Dataset,
    "biomassters": BioMasstersDataset,
    "burn_scars": BurnScarsDataset,
    "caffe": CaFFeDataset,
    "cloudsen12": CloudSen12Dataset,
    "kuro_siwo": KuroSiwoDataset,
    "spacenet2": SpaceNet2Dataset,
    "spacenet7": SpaceNet7Dataset,
    "substation": SubstationDataset,
    "treesatai": TreeSatAIDataset,
}

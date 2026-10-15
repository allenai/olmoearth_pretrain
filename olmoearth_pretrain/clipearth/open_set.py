"""Read open-set segmentation windows as OlmoEarth inputs plus class masks.

The windows are the ones Molmo's ``open_set_seg`` training set was built from: rslearn
windows of 128x128 px at 10 m, with every band group's monthly periods stacked
band-major in one GeoTIFF (``(bands * periods, H, W)``, all months of the first band,
then all months of the next). Most windows cover 12 months. Change datasets (burn
scars, floods, damage, ...) were later split into ``<name>_pre`` and ``<name>_post``
windows of 6 months each; the labels live on ``_post``, which is read in their place.
Windows with fewer than 12 periods are padded with missing months. Lower-resolution band
groups are upsampled to 10 m by repeating pixels.

Class names and masks come from the Molmo copy of the set
(``rs_datasets/open_set_seg_train``): ``manifest_{train,val}.json`` lists each window's
classes as ``{"value", "name"}`` and ``masks/<window>.png`` holds those values per pixel
(255 = unlabeled).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import rasterio
import torch
from PIL import Image
from rasterio.warp import transform as warp_transform
from torch.utils.data import Dataset

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.normalize import Normalizer, Strategy
from olmoearth_pretrain.data.utils import convert_to_db

WINDOWS_DIR = (
    "/weka/dfive-default/olmoearth_pretrain/dataset_creation/open_set_dataset/"
    "windows/open_set"
)
MOLMO_SET_DIR = "/weka/oe-training-default/mm-olmo/rs_datasets/open_set_seg_train"
NUM_PERIODS = 12
WINDOW_SIZE = 128
MASK_NODATA = 255

# (layer, band group dir, bands in the file's order, nodata value)
_LAYERS: dict[str, list[tuple[str, list[str], float]]] = {
    Modality.SENTINEL2_L2A.name: [
        ("B02_B03_B04_B08", ["B02", "B03", "B04", "B08"], 0),
        (
            "B05_B06_B07_B8A_B11_B12",
            ["B05", "B06", "B07", "B8A", "B11", "B12"],
            0,
        ),
        ("B01_B09", ["B01", "B09"], 0),
    ],
    Modality.SENTINEL1.name: [("vv_vh", ["vv", "vh"], -32768)],
    Modality.LANDSAT.name: [
        ("B8", ["B8"], 0),
        (
            "B1_B2_B3_B4_B5_B6_B7_B9_B10_B11",
            ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"],
            0,
        ),
    ],
}
MODALITIES = list(_LAYERS)


@dataclass
class WindowRecord:
    """One window of the Molmo open-set set."""

    name: str
    split: str
    slug: str
    classes: dict[int, str]  # mask value -> class name


def load_records(molmo_dir: str = MOLMO_SET_DIR) -> list[WindowRecord]:
    """Return every train and val window with its class names."""
    index_path = os.path.join(molmo_dir, "open_set_index.json")
    slugs = {
        window: slug
        for slug, windows in json.load(open(index_path)).items()
        for window in windows
    }
    records = []
    for split in ("train", "val"):
        for entry in json.load(open(os.path.join(molmo_dir, f"manifest_{split}.json"))):
            name = entry["image_id"]
            records.append(
                WindowRecord(
                    name=name,
                    split=split,
                    slug=slugs.get(name, name.rsplit("_", 1)[0]),
                    classes={int(a["value"]): a["name"] for a in entry["anns"]},
                )
            )
    return records


def _read_modality(
    layer_dir: str, modality: str
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(data [H, W, T, C] float32 raw units, missing [H, W, T, C] bool)``."""
    spec = Modality.get(modality)
    data = np.zeros((WINDOW_SIZE, WINDOW_SIZE, NUM_PERIODS, spec.num_bands), np.float32)
    missing = np.ones(data.shape, bool)
    for group, bands, nodata in _LAYERS[modality]:
        path = os.path.join(layer_dir, modality, group, "geotiff.tif")
        if not os.path.exists(path):
            return None
        with rasterio.open(path) as src:
            raw = src.read().astype(np.float32)
        periods, remainder = divmod(raw.shape[0], len(bands))
        if remainder or not 0 < periods <= NUM_PERIODS:
            return None
        scale = WINDOW_SIZE // raw.shape[1]
        if scale > 1:
            raw = raw.repeat(scale, axis=1).repeat(scale, axis=2)
        # band-major: (bands, periods, H, W) -> (H, W, periods, bands)
        raw = raw.reshape(len(bands), periods, WINDOW_SIZE, WINDOW_SIZE)
        raw = raw.transpose(2, 3, 1, 0)
        absent = (raw == nodata) | ~np.isfinite(raw)
        for i, band in enumerate(bands):
            c = spec.band_order.index(band)
            data[:, :, :periods, c] = raw[..., i]
            missing[:, :, :periods, c] = absent[..., i]
    if modality == Modality.SENTINEL1.name:
        data = np.where(missing, data, convert_to_db(np.where(missing, 1, data)))
    return data, missing


def monthly_timestamps(start: str, num_periods: int = NUM_PERIODS) -> torch.Tensor:
    """``[T, 3]`` (day, month 0-indexed, year) for each month from ``start``.

    Same convention as ``evals.datasets.rslearn_dataset.get_timestamps`` (the first of
    each month), which is not imported because ``evals.datasets`` pulls in every eval
    dataset and their dependencies.
    """
    first = datetime.fromisoformat(start)
    months = [first.year * 12 + first.month - 1 + k for k in range(num_periods)]
    return torch.tensor([[1, m % 12, m // 12] for m in months], dtype=torch.long)


def tile_centers(metadata: dict, tile_size: int) -> np.ndarray:
    """Lat/lon of each ``tile_size`` px tile's centre, ``[H/tile, W/tile, 2]``."""
    projection = metadata["projection"]
    x0, y0, x1, y1 = metadata["bounds"]
    n_rows, n_cols = (y1 - y0) // tile_size, (x1 - x0) // tile_size
    cols = x0 + (np.arange(n_cols) + 0.5) * tile_size
    rows = y0 + (np.arange(n_rows) + 0.5) * tile_size
    grid_y, grid_x = np.meshgrid(rows, cols, indexing="ij")
    lon, lat = warp_transform(
        projection["crs"],
        "EPSG:4326",
        (grid_x * projection["x_resolution"]).ravel().tolist(),
        (grid_y * projection["y_resolution"]).ravel().tolist(),
    )
    return np.stack([lat, lon], axis=-1).reshape(n_rows, n_cols, 2)


class OpenSetWindows(Dataset):
    """Yield normalized OlmoEarth inputs, class masks and metadata per window.

    Each item is a dict with:
        ``<modality>``: ``[H, W, T, C]`` float32, normalized, 0 where missing.
        ``<modality>_missing``: ``[H, W, T]`` bool, a pixel and timestep with any band
            of the modality missing.
        ``timestamps``: ``[T, 3]`` long (day, month 0-indexed, year).
        ``mask``: ``[H, W]`` uint8 class values (255 = unlabeled).
        ``tile_lat_lon``: ``[H/tile, W/tile, 2]`` float64 centre of each tile.
        plus ``name``, ``split``, ``slug`` and ``classes``.
    A window whose data cannot be read yields ``None``.
    """

    def __init__(
        self,
        records: list[WindowRecord],
        modalities: list[str] | None = None,
        tile_size: int = 16,
        windows_dir: str = WINDOWS_DIR,
        molmo_dir: str = MOLMO_SET_DIR,
    ) -> None:
        """Set up the dataset."""
        self.records = records
        self.modalities = modalities or MODALITIES
        self.tile_size = tile_size
        self.windows_dir = windows_dir
        self.molmo_dir = molmo_dir
        self.normalizer = Normalizer(Strategy.COMPUTED)

    def __len__(self) -> int:
        """Return the number of windows."""
        return len(self.records)

    def __getitem__(self, index: int) -> dict | None:
        """Return one window, or None if it cannot be read."""
        record = self.records[index]
        window_dir = os.path.join(self.windows_dir, record.name)
        if not os.path.isdir(window_dir):
            window_dir += "_post"
        try:
            metadata = json.load(open(os.path.join(window_dir, "metadata.json")))
            item: dict = {}
            for modality in self.modalities:
                read = _read_modality(os.path.join(window_dir, "layers"), modality)
                if read is None:
                    continue
                data, missing = read
                data = self.normalizer.normalize(Modality.get(modality), data)
                any_missing = missing.any(axis=-1)
                data[any_missing] = 0
                item[modality] = torch.from_numpy(data.astype(np.float32))
                item[f"{modality}_missing"] = torch.from_numpy(any_missing)
            if Modality.SENTINEL2_L2A.name not in item:
                return None
            item["timestamps"] = monthly_timestamps(metadata["time_range"][0])
            mask_path = os.path.join(self.molmo_dir, "masks", f"{record.name}.png")
            item["mask"] = torch.from_numpy(np.array(Image.open(mask_path)))
            item["tile_lat_lon"] = tile_centers(metadata, self.tile_size)
        except (OSError, ValueError, KeyError):
            return None
        item.update(
            name=record.name,
            split=record.split,
            slug=record.slug,
            classes=record.classes,
        )
        return item


__all__ = [
    "MASK_NODATA",
    "MODALITIES",
    "OpenSetWindows",
    "WindowRecord",
    "load_records",
]

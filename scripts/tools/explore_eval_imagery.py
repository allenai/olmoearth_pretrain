"""Browse the raw imagery behind an rslearn eval dataset: S2, S1, Landsat, label.

This is a look-and-see tool for the ``*_year_aligned`` eval datasets (and their
originals). It reads the materialized geotiffs straight off weka -- no model, no
h5 conversion, no eval harness -- and renders, per window, a grid of
month x modality thumbnails plus the label raster:

    rows    = views (S2 true colour / NDVI / SCL, Landsat true colour / QA, S1 ...)
    columns = the twelve 30-day layers, mo01 .. mo12 (ascending in time)

Run it on a weka-connected host and forward the port:

    # on the weka host, in the helios venv
    python scripts/tools/explore_eval_imagery.py --host 0.0.0.0 --port 8765

    # locally
    ssh -L 8765:localhost:8765 <host>
    # then open http://localhost:8765

By default it serves ethiopia_crops_year_aligned and descals_year_aligned,
resolved to their registered ``weka_path`` via the eval registry. Point it
anywhere with ``--ds_path name=/path/to/dataset`` (repeatable).

The first visit to a dataset builds a window index (group, name, split,
time range, centre label class) and caches it under ``--cache_dir``; pass
``--refresh`` to rebuild it. Indexing descals (17k windows) takes a couple of
minutes; ethiopia (2.5k) is seconds.

Headless smoke test, no browser needed::

    python scripts/tools/explore_eval_imagery.py \
        --dump ethiopia_crops_year_aligned --dump_dir /tmp/probe

which writes one PNG per view/month for the first window and prints a
layer-availability table.

Notes on what you are looking at:

* Reflectance. S2 DNs are divided by 10000; Landsat Collection-2 L1 DNs are
  converted to TOA reflectance (DN * 2.75e-5 - 0.2) so the two sensors are on
  the same scale and are directly comparable month to month.
* Stretch. ``fixed`` (the default) uses the same reflectance range for every
  month, which is what you want for judging cloudiness or phenology -- a
  per-image stretch makes a cloudy month look like a clear one. ``window``
  stretches on 2/98 percentiles pooled over all twelve months, ``image``
  per cell.
* Missing layers are drawn as a hatched placeholder, not skipped: for these
  datasets ``landsat``, ``scl`` and ``landsat_qa`` are optional inputs, so a
  gap here is exactly what the loader sees as all-MISSING.
* The centre pixel is outlined on the label panel because the AEF-style tasks
  score the centre pixel only (``label_at_center_pixel``). Classes are shown as
  the raw integers the label raster stores; 255 is nodata.
* Two trees. ``--datasets`` resolves to the registry's ``weka_path`` -- the
  REGISTERED copies the evals actually read. The staging tree
  (``/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/...``) is where
  scl/landsat/landsat_qa were materialized first; use ``--use_source_path`` to
  look at it. If a layer is on disk but the dataset's config.json never gained
  it, the rasters are still rendered (the band sets are known constants) and
  ``--list`` names the discrepancy in both directions -- which is the quickest
  way to tell "layer absent" from "layer present but unregistered".
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import rasterio.warp
from flask import Flask, Response, abort, redirect, render_template_string, request
from PIL import Image, ImageDraw

logger = logging.getLogger("explore_eval_imagery")

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = (
    REPO_ROOT / "olmoearth_pretrain" / "evals" / "studio_ingest" / "registry.json"
)

DEFAULT_DATASETS = ("ethiopia_crops_year_aligned", "descals_year_aligned")
MONTHS = 12

# rslearn writes -32768 for S1 nodata; the exports additionally carry a -100
# floor from clipping, and both mean "no radar here" for display purposes.
S1_NODATA_BELOW = -60.0

S2_REFLECTANCE_SCALE = 1.0 / 10000.0
# Landsat Collection 2 Level-1 TOA reflectance rescaling (the constant
# coefficients that hold for every C2 L1 scene).
LANDSAT_REFLECTANCE_MULT = 2.75e-5
LANDSAT_REFLECTANCE_ADD = -0.2

# Band sets as declared by data/rslearn_dataset_configs/config_*.json. They are
# needed because a dataset's own config.json can lag its rasters: the scl /
# landsat / landsat_qa layers were materialized on the staging tree and only
# later merged into the registered tree's config, so a layer can be on disk yet
# undeclared. Rendering keys off the raster's existence, not the declaration.
FALLBACK_BAND_SETS: dict[str, list[list[str]]] = {
    "sentinel2_l2a": [
        ["B02", "B03", "B04", "B08"],
        ["B05", "B06", "B07", "B8A", "B11", "B12"],
        ["B01", "B09"],
    ],
    "sentinel1": [["vv", "vh"]],
    "sentinel2_scl": [["SCL"]],
    "landsat": [
        ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"],
        ["B8"],
    ],
    "landsat_qa": [["QA_PIXEL"]],
    "label_raster": [["label"]],
}

SCL_CLASS_NAMES = {
    0: "nodata",
    1: "saturated",
    2: "dark",
    3: "shadow",
    4: "vegetation",
    5: "bare",
    6: "water",
    7: "unclassified",
    8: "cloud-med",
    9: "cloud-high",
    10: "cirrus",
    11: "snow",
}
SCL_CLOUD_CLASSES = (0, 1, 3, 8, 9, 10)
SCL_COLORS = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (60, 60, 60),
    3: (120, 90, 160),
    4: (40, 150, 60),
    5: (190, 160, 110),
    6: (40, 90, 200),
    7: (140, 140, 140),
    8: (200, 200, 200),
    9: (255, 255, 255),
    10: (120, 220, 230),
    11: (255, 120, 220),
}

# QA_PIXEL bit meanings we care about (Landsat 8/9 Collection 2).
QA_BITS = {
    "fill": 1 << 0,
    "dilated": 1 << 1,
    "cirrus": 1 << 2,
    "cloud": 1 << 3,
    "shadow": 1 << 4,
    "snow": 1 << 5,
}
QA_COLORS = {
    "clear": (30, 90, 40),
    "fill": (0, 0, 0),
    "dilated": (170, 170, 170),
    "cirrus": (120, 220, 230),
    "cloud": (255, 255, 255),
    "shadow": (120, 90, 160),
    "snow": (255, 120, 220),
}

TAB10 = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
]

# Hand-transcribed colormap stops, so this script needs no matplotlib.
VIRIDIS_STOPS = [
    (0.0, (68, 1, 84)),
    (0.25, (59, 82, 139)),
    (0.5, (33, 145, 140)),
    (0.75, (94, 201, 98)),
    (1.0, (253, 231, 37)),
]
RDYLGN_STOPS = [
    (0.0, (165, 0, 38)),
    (0.25, (244, 109, 67)),
    (0.5, (255, 255, 191)),
    (0.75, (102, 189, 99)),
    (1.0, (0, 104, 55)),
]
MAGMA_STOPS = [
    (0.0, (0, 0, 4)),
    (0.25, (80, 18, 123)),
    (0.5, (182, 54, 121)),
    (0.75, (251, 136, 97)),
    (1.0, (252, 253, 191)),
]
COLORMAPS = {
    "viridis": VIRIDIS_STOPS,
    "rdylgn": RDYLGN_STOPS,
    "magma": MAGMA_STOPS,
}


# ---------------------------------------------------------------------------
# View definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ViewSpec:
    """One row of the month grid: which layer/bands to read and how to draw them."""

    key: str
    title: str
    layer: str
    bands: tuple[str, ...]
    kind: str
    scale: str = "raw"
    vmin: float = 0.0
    vmax: float = 1.0
    cmap: str = "viridis"
    monthly: bool = True
    source: str | None = None
    note: str = ""

    def layer_name(self, month: int) -> str:
        """The rslearn layer name for this view at the given month (1-based)."""
        if "{m" not in self.layer:
            return self.layer
        return self.layer.format(m=month)


VIEW_LIST: list[ViewSpec] = [
    ViewSpec(
        "s2_rgb",
        "S2 true colour",
        "sentinel2_l2a_mo{m:02d}",
        ("B04", "B03", "B02"),
        "rgb",
        scale="s2",
        vmax=0.30,
        note="B04/B03/B02, 10 m",
    ),
    ViewSpec(
        "s2_nir",
        "S2 NIR false colour",
        "sentinel2_l2a_mo{m:02d}",
        ("B08", "B04", "B03"),
        "rgb",
        scale="s2",
        vmax=0.45,
        note="B08/B04/B03 -- vegetation is red",
    ),
    ViewSpec(
        "s2_swir",
        "S2 SWIR false colour",
        "sentinel2_l2a_mo{m:02d}",
        ("B12", "B08", "B04"),
        "rgb",
        scale="s2",
        vmax=0.50,
        note="B12/B08/B04, SWIR upsampled from 20 m",
    ),
    ViewSpec(
        "s2_ndvi",
        "S2 NDVI",
        "sentinel2_l2a_mo{m:02d}",
        ("B08", "B04"),
        "ndvi",
        scale="s2",
        vmin=-0.2,
        vmax=0.9,
        cmap="rdylgn",
    ),
    ViewSpec(
        "scl",
        "S2 SCL",
        "sentinel2_scl_mo{m:02d}",
        ("SCL",),
        "scl",
        note="scene classification; white/grey = cloud",
    ),
    ViewSpec(
        "l8_rgb",
        "Landsat true colour",
        "landsat_mo{m:02d}",
        ("B4", "B3", "B2"),
        "rgb",
        scale="landsat",
        vmax=0.30,
        note="B4/B3/B2, 30 m",
    ),
    ViewSpec(
        "l8_nir",
        "Landsat NIR false colour",
        "landsat_mo{m:02d}",
        ("B5", "B4", "B3"),
        "rgb",
        scale="landsat",
        vmax=0.45,
    ),
    ViewSpec(
        "l8_swir",
        "Landsat SWIR false colour",
        "landsat_mo{m:02d}",
        ("B7", "B5", "B4"),
        "rgb",
        scale="landsat",
        vmax=0.50,
    ),
    ViewSpec(
        "l8_pan",
        "Landsat panchromatic",
        "landsat_mo{m:02d}",
        ("B8",),
        "gray",
        scale="landsat",
        vmin=0.0,
        vmax=0.35,
        cmap="viridis",
        note="B8, 15 m -- the only full-resolution Landsat band",
    ),
    ViewSpec(
        "l8_therm",
        "Landsat thermal",
        "landsat_mo{m:02d}",
        ("B10",),
        "gray",
        scale="raw",
        vmin=20000.0,
        vmax=32000.0,
        cmap="magma",
        note="B10 raw DN (radiance, not reflectance)",
    ),
    ViewSpec(
        "l8_qa",
        "Landsat QA_PIXEL",
        "landsat_qa_mo{m:02d}",
        ("QA_PIXEL",),
        "qa",
        note="cloud / shadow / cirrus bit flags",
    ),
    ViewSpec(
        "s1",
        "S1 vv/vh/ratio",
        "sentinel1_mo{m:02d}",
        ("vv", "vh"),
        "s1",
        note="RTC dB: vv, vh, vv-vh",
    ),
    ViewSpec(
        "s1_vv",
        "S1 vv",
        "sentinel1_mo{m:02d}",
        ("vv",),
        "gray",
        scale="raw",
        vmin=-25.0,
        vmax=5.0,
        cmap="viridis",
    ),
    ViewSpec(
        "s1_vh",
        "S1 vh",
        "sentinel1_mo{m:02d}",
        ("vh",),
        "gray",
        scale="raw",
        vmin=-32.0,
        vmax=0.0,
        cmap="viridis",
    ),
]

STATIC_VIEW_LIST: list[ViewSpec] = [
    ViewSpec(
        "label",
        "Label",
        "label_raster",
        ("label",),
        "label",
        monthly=False,
        note="centre pixel outlined; 255 = nodata",
    ),
    ViewSpec(
        "s2_median",
        "S2 median composite",
        "sentinel2_l2a_mo{m:02d}",
        ("B04", "B03", "B02"),
        "median",
        scale="s2",
        vmax=0.30,
        monthly=False,
        source="s2_rgb",
        note="per-pixel median over the twelve months",
    ),
    ViewSpec(
        "l8_median",
        "Landsat median composite",
        "landsat_mo{m:02d}",
        ("B4", "B3", "B2"),
        "median",
        scale="landsat",
        vmax=0.30,
        monthly=False,
        source="l8_rgb",
        note="per-pixel median over the twelve months",
    ),
]

VIEWS: dict[str, ViewSpec] = {v.key: v for v in VIEW_LIST + STATIC_VIEW_LIST}
DEFAULT_VIEWS = ("s2_rgb", "s2_ndvi", "scl", "l8_rgb", "l8_qa", "s1")


# ---------------------------------------------------------------------------
# Dataset plumbing
# ---------------------------------------------------------------------------


def bandset_dirname(bands: list[str]) -> str:
    """Directory name rslearn uses for a band set (mirrors rslearn's own rule)."""
    if any("_" in band for band in bands):
        return hashlib.sha256(json.dumps(bands).encode()).hexdigest()
    dirname = "_".join(bands)
    if len(dirname) > 64:
        dirname = hashlib.sha256(dirname.encode()).hexdigest()
    return dirname


@dataclass
class Dataset:
    """One rslearn dataset on disk, plus the band index derived from its config."""

    name: str
    path: Path
    config: dict[str, Any] = field(default_factory=dict)
    # layer name -> band name -> (band set dirname, 1-based band index)
    band_index: dict[str, dict[str, tuple[str, int]]] = field(default_factory=dict)
    windows: list[dict[str, Any]] = field(default_factory=list)
    has_labels: bool = False
    declared_layers: set[str] = field(default_factory=set)
    _seen_layers: set[str] | None = None

    def load_config(self) -> None:
        """Read the dataset's config.json and build the layer/band lookup."""
        config_path = self.path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"{self.name}: no config.json at {config_path}")
        self.config = json.loads(config_path.read_text())
        for layer_name, layer in self.config.get("layers", {}).items():
            if layer.get("type") != "raster":
                continue
            per_band: dict[str, tuple[str, int]] = {}
            for band_set in layer.get("band_sets", []):
                bands = list(band_set.get("bands", []))
                dirname = bandset_dirname(bands)
                for idx, band in enumerate(bands):
                    per_band[band] = (dirname, idx + 1)
            self.band_index[layer_name] = per_band
            self.declared_layers.add(layer_name)
        self._add_fallback_layers()

    def _add_fallback_layers(self) -> None:
        """Add band indexes for standard layers this config.json does not declare."""
        for prefix, band_sets in FALLBACK_BAND_SETS.items():
            names = (
                [prefix]
                if prefix == "label_raster"
                else [f"{prefix}_mo{m:02d}" for m in range(1, MONTHS + 1)]
            )
            for layer_name in names:
                if layer_name in self.band_index:
                    continue
                per_band: dict[str, tuple[str, int]] = {}
                for bands in band_sets:
                    dirname = bandset_dirname(bands)
                    for idx, band in enumerate(bands):
                        per_band[band] = (dirname, idx + 1)
                self.band_index[layer_name] = per_band

    def seen_layers(self, sample: int = 5) -> set[str]:
        """Layer directories that actually exist, sampled over a few windows."""
        if self._seen_layers is None:
            seen: set[str] = set()
            for row in self.windows[:sample]:
                layers_dir = self.window_root(row["group"], row["name"]) / "layers"
                if layers_dir.exists():
                    seen.update(entry.name for entry in os.scandir(layers_dir))
            self._seen_layers = seen
        return self._seen_layers

    def window_root(self, group: str, name: str) -> Path:
        """Directory holding one window's metadata and layers."""
        return self.path / "windows" / group / name

    def available_views(self) -> list[ViewSpec]:
        """Views whose layer is declared in the config or present on disk."""
        seen = self.seen_layers()
        out = []
        for view in VIEW_LIST + STATIC_VIEW_LIST:
            layer = view.layer_name(1)
            bands = self.band_index.get(layer)
            if bands is None or any(band not in bands for band in view.bands):
                continue
            if layer in self.declared_layers or layer in seen:
                out.append(view)
        return out


def resolve_datasets(
    names: list[str], explicit: list[str], use_source_path: bool
) -> list[Dataset]:
    """Turn --datasets names and --ds_path overrides into Dataset objects."""
    datasets: list[Dataset] = []
    registry: dict[str, Any] = {}
    if names and REGISTRY_PATH.exists():
        registry = json.loads(REGISTRY_PATH.read_text()).get("datasets", {})

    for name in names:
        entry = registry.get(name)
        if entry is None:
            raise SystemExit(
                f"{name} is not in {REGISTRY_PATH}; pass --ds_path {name}=/path instead"
            )
        key = "source_path" if use_source_path else "weka_path"
        datasets.append(Dataset(name=name, path=Path(entry[key])))

    for spec in explicit:
        if "=" not in spec:
            raise SystemExit(f"--ds_path expects name=/path, got {spec!r}")
        name, path = spec.split("=", 1)
        datasets.append(Dataset(name=name, path=Path(path)))

    for dataset in datasets:
        try:
            dataset.load_config()
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc
        logger.info(
            "%s: %s (%d raster layers)",
            dataset.name,
            dataset.path,
            len(dataset.band_index),
        )
    return datasets


# ---------------------------------------------------------------------------
# Window index
# ---------------------------------------------------------------------------


def scan_window_dirs(dataset: Dataset, exclude_groups: str) -> list[tuple[str, str]]:
    """List (group, window name) pairs on disk, skipping excluded groups."""
    windows_root = dataset.path / "windows"
    if not windows_root.exists():
        raise FileNotFoundError(f"{dataset.name}: no windows/ under {dataset.path}")
    pattern = re.compile(exclude_groups) if exclude_groups else None
    pairs: list[tuple[str, str]] = []
    for group_entry in sorted(os.scandir(windows_root), key=lambda e: e.name):
        if not group_entry.is_dir():
            continue
        if pattern is not None and pattern.search(group_entry.name):
            logger.info("%s: skipping group %s", dataset.name, group_entry.name)
            continue
        for window_entry in os.scandir(group_entry.path):
            if window_entry.is_dir():
                pairs.append((group_entry.name, window_entry.name))
    pairs.sort()
    return pairs


def read_window_row(
    dataset: Dataset, group: str, name: str, with_labels: bool
) -> dict[str, Any]:
    """One index row: split, time range and (optionally) the centre label class."""
    row: dict[str, Any] = {"group": group, "name": name}
    root = dataset.window_root(group, name)
    try:
        metadata = json.loads((root / "metadata.json").read_text())
    except (OSError, json.JSONDecodeError):
        metadata = {}
    options = metadata.get("options") or {}
    split = options.get("eval_split") or options.get("split") or ""
    if not split and group in {"train", "val", "test"}:
        split = group
    row["split"] = split
    time_range = metadata.get("time_range")
    row["start"] = time_range[0][:10] if time_range else ""
    row["end"] = time_range[1][:10] if time_range else ""
    if with_labels:
        label, valid = center_label(dataset, group, name)
        row["label"] = label
        row["valid_frac"] = valid
    return row


def center_label(dataset: Dataset, group: str, name: str) -> tuple[int | None, float]:
    """Centre-pixel label class and the fraction of non-nodata label pixels."""
    array = read_band(dataset, group, name, "label_raster", "label")
    if array is None:
        return None, 0.0
    array = array.astype(np.int32)
    center = int(array[array.shape[0] // 2, array.shape[1] // 2])
    valid = float(np.mean(array != 255))
    return center, valid


def index_cache_path(cache_dir: Path, dataset: Dataset) -> Path:
    """Where the window index for this dataset is cached."""
    digest = hashlib.sha256(str(dataset.path).encode()).hexdigest()[:8]
    return cache_dir / f"{dataset.name}.{digest}.index.json"


def build_index(
    dataset: Dataset,
    cache_dir: Path,
    exclude_groups: str,
    workers: int,
    with_labels: bool,
    refresh: bool,
) -> None:
    """Populate ``dataset.windows``, reading from (or writing) the cache."""
    cache_path = index_cache_path(cache_dir, dataset)
    if cache_path.exists() and not refresh:
        cached = json.loads(cache_path.read_text())
        if cached.get("with_labels") or not with_labels:
            dataset.windows = cached["windows"]
            dataset.has_labels = bool(cached.get("with_labels"))
            logger.info(
                "%s: %d windows from cache %s",
                dataset.name,
                len(dataset.windows),
                cache_path,
            )
            return
        logger.info("%s: cache has no labels, rebuilding", dataset.name)

    pairs = scan_window_dirs(dataset, exclude_groups)
    logger.info(
        "%s: indexing %d windows with %d workers (labels=%s)",
        dataset.name,
        len(pairs),
        workers,
        with_labels,
    )
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(read_window_row, dataset, group, name, with_labels)
            for group, name in pairs
        ]
        for done, future in enumerate(futures, start=1):
            rows.append(future.result())
            if done % 2000 == 0:
                logger.info("%s: %d/%d", dataset.name, done, len(pairs))
    rows.sort(key=lambda r: (r["group"], r["name"]))
    dataset.windows = rows
    dataset.has_labels = with_labels
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "ds_path": str(dataset.path),
                "with_labels": with_labels,
                "windows": rows,
            }
        )
    )
    logger.info("%s: wrote %s", dataset.name, cache_path)


def filter_windows(
    dataset: Dataset, split: str, label: str, group: str, query: str
) -> list[dict[str, Any]]:
    """Apply the browser's filters to the window index."""
    rows = dataset.windows
    if split:
        rows = [r for r in rows if r.get("split") == split]
    if group:
        rows = [r for r in rows if r.get("group") == group]
    if label != "":
        want = int(label)
        rows = [r for r in rows if r.get("label") == want]
    if query:
        rows = [r for r in rows if query in r["name"]]
    return rows


# ---------------------------------------------------------------------------
# Raster reading
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8192)
def _read_band_cached(
    tif_path: str, band_idx: int
) -> np.ndarray | None:  # pragma: no cover - IO
    try:
        with rasterio.open(tif_path) as src:
            return src.read(band_idx).astype(np.float32)
    except Exception as exc:
        logger.debug("failed reading %s band %d: %s", tif_path, band_idx, exc)
        return None


@lru_cache(maxsize=65536)
def _find_tif(window_root: str, layer: str, dirname: str) -> str | None:
    layer_dir = Path(window_root) / "layers" / layer / dirname
    if not layer_dir.exists():
        return None
    tifs = sorted(layer_dir.glob("*.tif"))
    return str(tifs[0]) if tifs else None


def read_band(
    dataset: Dataset, group: str, name: str, layer: str, band: str
) -> np.ndarray | None:
    """Read one band of one layer for one window, or None if it is not there."""
    entry = dataset.band_index.get(layer, {}).get(band)
    if entry is None:
        return None
    dirname, band_idx = entry
    tif_path = _find_tif(str(dataset.window_root(group, name)), layer, dirname)
    if tif_path is None:
        return None
    array = _read_band_cached(tif_path, band_idx)
    return None if array is None else array.copy()


def layer_present(
    dataset: Dataset, group: str, name: str, view: ViewSpec, month: int
) -> bool:
    """Whether the geotiff backing this view/month exists on disk."""
    layer = view.layer_name(month)
    entry = dataset.band_index.get(layer, {}).get(view.bands[0])
    if entry is None:
        return False
    return _find_tif(str(dataset.window_root(group, name)), layer, entry[0]) is not None


def scale_array(array: np.ndarray, scale: str) -> np.ndarray:
    """Convert raw DNs to the display units the view's vmin/vmax are stated in."""
    if scale == "s2":
        out = array * S2_REFLECTANCE_SCALE
        out[array <= 0] = np.nan
        return out
    if scale == "landsat":
        out = array * LANDSAT_REFLECTANCE_MULT + LANDSAT_REFLECTANCE_ADD
        out[array <= 0] = np.nan
        return out
    return array


def upsample_to(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize a 2D float array (band sets differ in zoom)."""
    if array.shape == shape:
        return array
    image = Image.fromarray(array, mode="F")
    resized = image.resize((shape[1], shape[0]), resample=Image.NEAREST)
    return np.asarray(resized, dtype=np.float32)


def view_channels(
    dataset: Dataset, group: str, name: str, view: ViewSpec, month: int
) -> list[np.ndarray] | None:
    """Read a view's bands for one month, scaled, nodata-masked and co-shaped."""
    layer = view.layer_name(month)
    arrays = []
    for band in view.bands:
        array = read_band(dataset, group, name, layer, band)
        if array is None:
            return None
        arrays.append(array)
    if view.kind == "s1" or view.layer.startswith("sentinel1"):
        arrays = [np.where(a <= S1_NODATA_BELOW, np.nan, a) for a in arrays]
    elif view.kind in {"rgb", "ndvi", "gray", "median"}:
        arrays = [scale_array(a, view.scale) for a in arrays]
    height = max(a.shape[0] for a in arrays)
    width = max(a.shape[1] for a in arrays)
    return [upsample_to(a, (height, width)) for a in arrays]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def stretch_to_uint8(
    channel: np.ndarray, lo: float, hi: float
) -> tuple[np.ndarray, np.ndarray]:
    """Linear stretch to uint8; also return the nan mask so it can be greyed."""
    nan_mask = ~np.isfinite(channel)
    if hi <= lo:
        hi = lo + 1e-6
    scaled = np.clip((np.nan_to_num(channel, nan=lo) - lo) / (hi - lo), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8), nan_mask


def percentiles(
    channels: list[np.ndarray], low: float, high: float
) -> list[tuple[float, float]]:
    """Per-channel (low, high) percentiles ignoring nans."""
    out = []
    for channel in channels:
        finite = channel[np.isfinite(channel)]
        if finite.size == 0:
            out.append((0.0, 1.0))
        else:
            out.append(
                (float(np.percentile(finite, low)), float(np.percentile(finite, high)))
            )
    return out


@lru_cache(maxsize=4096)
def window_stretch(
    ds_name: str, group: str, name: str, view_key: str, low: float, high: float
) -> tuple[tuple[float, float], ...]:
    """Percentile stretch pooled over all twelve months of a view."""
    dataset = DATASETS[ds_name]
    view = VIEWS[view_key]
    pooled: list[list[np.ndarray]] = [[] for _ in view.bands]
    months = range(1, MONTHS + 1) if view.monthly else [1]
    for month in months:
        channels = view_channels(dataset, group, name, view, month)
        if channels is None:
            continue
        for idx, channel in enumerate(channels):
            pooled[idx].append(channel.ravel())
    if not pooled[0]:
        return tuple((0.0, 1.0) for _ in view.bands)
    joined = [np.concatenate(parts) for parts in pooled]
    return tuple(percentiles(joined, low, high))


def apply_colormap(
    normalized: np.ndarray, stops: list[tuple[float, tuple[int, int, int]]]
) -> np.ndarray:
    """Map values in [0, 1] through a piecewise-linear colormap to RGB uint8."""
    positions = np.array([s[0] for s in stops])
    rgb = np.stack(
        [np.interp(normalized, positions, [s[1][i] for s in stops]) for i in range(3)],
        axis=-1,
    )
    return rgb.astype(np.uint8)


def categorical_rgb(
    array: np.ndarray,
    colors: dict[int, tuple[int, int, int]],
    default: tuple[int, int, int],
) -> np.ndarray:
    """Colour a small-integer raster with a lookup table."""
    out = np.zeros(array.shape + (3,), dtype=np.uint8)
    out[:] = default
    values = array.astype(np.int32)
    for value, color in colors.items():
        out[values == value] = color
    return out


def grey_out(rgb: np.ndarray, nan_mask: np.ndarray) -> np.ndarray:
    """Paint nodata pixels a flat dark grey so they read as absent, not black."""
    if nan_mask.any():
        rgb = rgb.copy()
        rgb[nan_mask] = (45, 45, 52)
    return rgb


def render_view(
    dataset: Dataset,
    group: str,
    name: str,
    view: ViewSpec,
    month: int,
    mode: str,
    percentile_low: float,
    percentile_high: float,
) -> np.ndarray | None:
    """Render one grid cell to an RGB uint8 array, or None if the layer is absent."""
    if view.kind == "median":
        source = VIEWS[view.source or "s2_rgb"]
        stacks: list[np.ndarray] = []
        for m in range(1, MONTHS + 1):
            channels = view_channels(dataset, group, name, source, m)
            if channels is not None:
                stacks.append(np.stack(channels, axis=0))
        if not stacks:
            return None
        shape = max((s.shape for s in stacks), key=lambda s: s[1] * s[2])
        aligned = [
            np.stack([upsample_to(c, (shape[1], shape[2])) for c in stack], axis=0)
            for stack in stacks
        ]
        with np.errstate(all="ignore"):
            channels = list(np.nanmedian(np.stack(aligned, axis=0), axis=0))
        view = ViewSpec(
            view.key,
            view.title,
            view.layer,
            view.bands,
            "rgb",
            view.scale,
            view.vmin,
            view.vmax,
        )
        # A pooled-over-months stretch is meaningless once the months are gone.
        mode = "image" if mode == "image" else "fixed"
    else:
        channels = view_channels(dataset, group, name, view, month)
        if channels is None:
            return None

    if view.kind == "scl":
        array = channels[0]
        return categorical_rgb(array, SCL_COLORS, (200, 0, 200))

    if view.kind == "qa":
        qa = np.nan_to_num(channels[0], nan=0.0).astype(np.int64)
        rgb = np.zeros(qa.shape + (3,), dtype=np.uint8)
        rgb[:] = QA_COLORS["clear"]
        # Painted least-to-most important so cloud wins over its own dilation.
        for flag in ("snow", "shadow", "cirrus", "dilated", "cloud", "fill"):
            rgb[(qa & QA_BITS[flag]) > 0] = QA_COLORS[flag]
        return rgb

    if view.kind == "label":
        array = np.nan_to_num(channels[0], nan=255.0).astype(np.int32)
        colors = {
            int(v): TAB10[int(v) % len(TAB10)] for v in np.unique(array) if v != 255
        }
        colors[255] = (28, 28, 32)
        return categorical_rgb(array, colors, (28, 28, 32))

    if view.kind == "ndvi":
        nir, red = channels[0], channels[1]
        with np.errstate(all="ignore"):
            ndvi = (nir - red) / (nir + red)
        normalized = np.clip((ndvi - view.vmin) / (view.vmax - view.vmin), 0.0, 1.0)
        rgb = apply_colormap(np.nan_to_num(normalized, nan=0.0), COLORMAPS[view.cmap])
        return grey_out(rgb, ~np.isfinite(ndvi))

    if view.kind == "s1":
        vv = channels[0]
        vh = channels[1] if len(channels) > 1 else channels[0]
        stacked = [vv, vh, vv - vh]
        ranges = [(-25.0, 5.0), (-32.0, 0.0), (0.0, 15.0)]
        if mode == "window":
            ranges = list(
                window_stretch(
                    dataset.name, group, name, view.key, percentile_low, percentile_high
                )
            )
            ranges = [
                ranges[0],
                ranges[1] if len(ranges) > 1 else ranges[0],
                (0.0, 15.0),
            ]
        elif mode == "image":
            ranges = percentiles(stacked, percentile_low, percentile_high)
        planes, masks = zip(
            *[stretch_to_uint8(c, lo, hi) for c, (lo, hi) in zip(stacked, ranges)]
        )
        return grey_out(np.stack(planes, axis=-1), np.logical_or.reduce(masks))

    if view.kind == "gray":
        channel = channels[0]
        lo, hi = view.vmin, view.vmax
        if mode == "window":
            lo, hi = window_stretch(
                dataset.name, group, name, view.key, percentile_low, percentile_high
            )[0]
        elif mode == "image":
            lo, hi = percentiles([channel], percentile_low, percentile_high)[0]
        plane, nan_mask = stretch_to_uint8(channel, lo, hi)
        rgb = apply_colormap(plane.astype(np.float32) / 255.0, COLORMAPS[view.cmap])
        return grey_out(rgb, nan_mask)

    # kind == "rgb"
    ranges = [(view.vmin, view.vmax)] * len(channels)
    if mode == "window":
        ranges = list(
            window_stretch(
                dataset.name, group, name, view.key, percentile_low, percentile_high
            )
        )
    elif mode == "image":
        ranges = percentiles(channels, percentile_low, percentile_high)
    planes, masks = zip(
        *[stretch_to_uint8(c, lo, hi) for c, (lo, hi) in zip(channels, ranges)]
    )
    rgb = np.stack(planes[:3], axis=-1)
    return grey_out(rgb, np.logical_or.reduce(masks))


def placeholder(size: int) -> Image.Image:
    """Hatched tile standing in for a layer that was never materialized."""
    image = Image.new("RGB", (size, size), (24, 24, 28))
    draw = ImageDraw.Draw(image)
    for offset in range(-size, size, 8):
        draw.line([(offset, 0), (offset + size, size)], fill=(52, 52, 60), width=1)
    return image


def to_display_image(
    rgb: np.ndarray, size: int, mark_center: bool = False
) -> Image.Image:
    """Nearest-upsample a rendered tile to the display size."""
    image = Image.fromarray(rgb, mode="RGB")
    scale = max(1, size // max(image.size))
    if scale > 1:
        image = image.resize(
            (image.size[0] * scale, image.size[1] * scale), Image.NEAREST
        )
    if image.size[0] != size:
        image = image.resize((size, size), Image.NEAREST)
    if mark_center:
        draw = ImageDraw.Draw(image)
        pixel = size / max(rgb.shape[0], 1)
        half = rgb.shape[0] // 2
        x0, y0 = half * pixel, half * pixel
        draw.rectangle(
            [x0, y0, x0 + pixel - 1, y0 + pixel - 1], outline=(255, 255, 0), width=2
        )
    return image


def png_response(image: Image.Image) -> Response:
    """Serve a PIL image as a PNG with a long cache lifetime."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    response = Response(buffer.getvalue(), mimetype="image/png")
    response.headers["Cache-Control"] = "max-age=3600"
    return response


# ---------------------------------------------------------------------------
# Window-level metadata for the detail page
# ---------------------------------------------------------------------------


@lru_cache(maxsize=2048)
def window_items(ds_name: str, group: str, name: str) -> dict[str, dict[str, Any]]:
    """Per-layer scene info from items.json: name, date and cloud cover."""
    dataset = DATASETS[ds_name]
    items_path = dataset.window_root(group, name) / "items.json"
    try:
        entries = json.loads(items_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for entry in entries:
        groups = entry.get("serialized_item_groups") or []
        items = groups[0] if groups else []
        if not items:
            continue
        first = items[0]
        time_range = (first.get("geometry") or {}).get("time_range")
        cloud = first.get("cloud_cover")
        out[entry["layer_name"]] = {
            "names": [item.get("name", "") for item in items],
            "date": time_range[0][:10] if time_range else "",
            "cloud": cloud if isinstance(cloud, int | float) and cloud >= 0 else None,
            "count": len(items),
        }
    return out


@lru_cache(maxsize=2048)
def window_latlon(ds_name: str, group: str, name: str) -> tuple[float, float] | None:
    """Centre of the window in EPSG:4326, from its projection and bounds."""
    dataset = DATASETS[ds_name]
    try:
        metadata = json.loads(
            (dataset.window_root(group, name) / "metadata.json").read_text()
        )
    except (OSError, json.JSONDecodeError):
        return None
    projection = metadata.get("projection") or {}
    bounds = metadata.get("bounds")
    if not projection or not bounds:
        return None
    x = (bounds[0] + bounds[2]) / 2 * projection["x_resolution"]
    y = (bounds[1] + bounds[3]) / 2 * projection["y_resolution"]
    try:
        lon, lat = rasterio.warp.transform(projection["crs"], "EPSG:4326", [x], [y])
    except Exception:
        return None
    return float(lat[0]), float(lon[0])


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

STYLE = """
<style>
:root { color-scheme: dark; }
body { background:#131316; color:#e7e7ea; font:13px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; margin:0 0 48px; }
a { color:#8ab4f8; text-decoration:none; } a:hover { text-decoration:underline; }
header { position:sticky; top:0; background:#1b1b20; border-bottom:1px solid #2c2c34; padding:10px 16px; z-index:5; }
h1 { font-size:16px; margin:0 0 4px; font-weight:600; }
.meta { color:#9a9aa6; font-size:12px; }
.meta b { color:#d5d5dd; font-weight:600; }
main { padding:16px; }
form.controls { display:flex; gap:10px; align-items:flex-end; flex-wrap:wrap; margin:10px 0 0; }
label.field { display:flex; flex-direction:column; gap:3px; font-size:11px; color:#9a9aa6; }
select, input[type=text], input[type=number] { background:#22222a; color:#e7e7ea; border:1px solid #33333d; border-radius:4px; padding:4px 6px; font:12px inherit; }
button { background:#2f4f8f; color:#fff; border:0; border-radius:4px; padding:6px 12px; font:12px inherit; cursor:pointer; }
button.sec { background:#31313b; }
table.grid { border-collapse:separate; border-spacing:3px; }
table.grid th { font-size:11px; color:#9a9aa6; font-weight:600; text-align:left; vertical-align:bottom; }
table.grid th.row { text-align:right; padding-right:8px; white-space:nowrap; max-width:200px; }
table.grid th.row span { display:block; color:#71717e; font-weight:400; font-size:10px; white-space:normal; }
figure { margin:0; }
figure img { display:block; border-radius:3px; background:#1c1c22; image-rendering:pixelated; }
figcaption { font-size:10px; color:#71717e; text-align:center; margin-top:2px; }
.cards { display:flex; flex-wrap:wrap; gap:10px; }
.card { background:#1b1b20; border:1px solid #2a2a32; border-radius:6px; padding:6px; }
.card .cap { font-size:10px; color:#9a9aa6; margin-top:4px; }
.chip { display:inline-block; padding:1px 6px; border-radius:9px; font-size:10px; color:#0d0d10; font-weight:700; }
.statics { display:flex; gap:16px; flex-wrap:wrap; margin:14px 0 20px; }
.legend { display:flex; flex-wrap:wrap; gap:8px; font-size:11px; color:#9a9aa6; margin-top:6px; }
.legend span i { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:3px; vertical-align:middle; }
.views { display:flex; flex-wrap:wrap; gap:2px 12px; max-width:900px; }
.views label { font-size:11px; color:#c8c8d2; display:flex; gap:4px; align-items:center; }
.nav { display:flex; gap:8px; align-items:center; }
</style>
"""

INDEX_HTML = (
    STYLE
    + """
<header>
  <h1>Eval imagery explorer</h1>
  <div class="meta">{{ datasets|length }} dataset(s)</div>
</header>
<main>
<ul>
{% for d in datasets %}
  <li style="margin-bottom:10px">
    <a href="/ds/{{ d.name }}"><b>{{ d.name }}</b></a>
    <div class="meta">{{ d.n }} windows &middot; {{ d.path }}<br>views: {{ d.views }}</div>
  </li>
{% endfor %}
</ul>
</main>
"""
)

BROWSE_HTML = (
    STYLE
    + """
<header>
  <h1><a href="/">datasets</a> / {{ ds.name }}</h1>
  <div class="meta"><b>{{ n_filtered }}</b> of {{ ds.windows|length }} windows
    &middot; page {{ page + 1 }} / {{ n_pages }} &middot; {{ ds.path }}</div>
  <form class="controls" method="get">
    <label class="field">split
      <select name="split"><option value="">any</option>
      {% for s in splits %}<option value="{{ s }}" {% if s == split %}selected{% endif %}>{{ s }}</option>{% endfor %}
      </select></label>
    <label class="field">group
      <select name="group"><option value="">any</option>
      {% for g in groups %}<option value="{{ g }}" {% if g == group %}selected{% endif %}>{{ g }}</option>{% endfor %}
      </select></label>
    <label class="field">centre label
      <select name="label"><option value="">any</option>
      {% for c in classes %}<option value="{{ c }}" {% if c|string == label %}selected{% endif %}>{{ c }}</option>{% endfor %}
      </select></label>
    <label class="field">name contains<input type="text" name="q" value="{{ q }}" size="12"></label>
    <label class="field">thumb
      <select name="thumb">
      {% for v in thumb_views %}<option value="{{ v }}" {% if v == thumb %}selected{% endif %}>{{ v }}</option>{% endfor %}
      </select></label>
    <label class="field">month<input type="number" name="tmonth" min="0" max="12" value="{{ tmonth }}" size="3"></label>
    <label class="field">stretch
      <select name="stretch">
      {% for m in ['fixed','window','image'] %}<option value="{{ m }}" {% if m == stretch %}selected{% endif %}>{{ m }}</option>{% endfor %}
      </select></label>
    <label class="field">per page<input type="number" name="per_page" value="{{ per_page }}" size="4"></label>
    <input type="hidden" name="page" value="0">
    <button type="submit">apply</button>
    <a class="nav" href="{{ random_url }}"><button class="sec" type="button">random window</button></a>
  </form>
  <div class="nav" style="margin-top:8px">
    {% if page > 0 %}<a href="{{ page_url(page - 1) }}">&larr; prev</a>{% endif %}
    {% if page + 1 < n_pages %}<a href="{{ page_url(page + 1) }}">next &rarr;</a>{% endif %}
  </div>
</header>
<main>
  {% if not ds.has_labels %}
  <div class="meta" style="margin-bottom:10px">centre-label filter unavailable: index built with --no_labels</div>
  {% endif %}
  <div class="cards">
  {% for w in rows %}
    <div class="card">
      <a href="{{ w.url }}"><img src="{{ w.thumb }}" width="{{ size }}" height="{{ size }}" loading="lazy"></a>
      <div class="cap">{{ w.name }}<br>
        {{ w.split }}{% if w.label is not none %} &middot; <span class="chip" style="background:{{ w.color }}">{{ w.label }}</span>{% endif %}
      </div>
    </div>
  {% endfor %}
  </div>
</main>
"""
)

WINDOW_HTML = (
    STYLE
    + """
<header>
  <h1><a href="/">datasets</a> / <a href="{{ browse_url }}">{{ ds.name }}</a> / {{ group }}/{{ name }}</h1>
  <div class="meta">
    split <b>{{ row.split or '?' }}</b> &middot; window <b>{{ row.start }} &rarr; {{ row.end }}</b>
    {% if latlon %}&middot; <b>{{ '%.4f'|format(latlon[0]) }}, {{ '%.4f'|format(latlon[1]) }}</b>
      (<a target="_blank" href="https://www.google.com/maps/@{{ latlon[0] }},{{ latlon[1] }},14z/data=!3m1!1e3">map</a>){% endif %}
    {% if row.get('label') is not none %}&middot; centre label <b>{{ row.get('label') }}</b>{% endif %}
    {% if row.get('valid_frac') is not none %}&middot; labelled pixels <b>{{ '%.0f'|format(row.get('valid_frac') * 100) }}%</b>{% endif %}
  </div>
  <form class="controls" method="get">
    <label class="field">stretch
      <select name="stretch">
      {% for m in ['fixed','window','image'] %}<option value="{{ m }}" {% if m == stretch %}selected{% endif %}>{{ m }}</option>{% endfor %}
      </select></label>
    <label class="field">tile px<input type="number" name="size" value="{{ size }}" size="4"></label>
    <label class="field">pct low<input type="number" name="plow" value="{{ plow }}" size="3" step="0.5"></label>
    <label class="field">pct high<input type="number" name="phigh" value="{{ phigh }}" size="3" step="0.5"></label>
    <div class="views">
      {% for v in all_views %}
      <label><input type="checkbox" name="views" value="{{ v.key }}" {% if v.key in selected %}checked{% endif %}>{{ v.key }}</label>
      {% endfor %}
    </div>
    {% for k, v in carry.items() %}<input type="hidden" name="{{ k }}" value="{{ v }}">{% endfor %}
    <button type="submit">apply</button>
  </form>
  <div class="nav" style="margin-top:8px">
    {% if prev_url %}<a href="{{ prev_url }}">&larr; prev window</a>{% endif %}
    {% if next_url %}<a href="{{ next_url }}">next window &rarr;</a>{% endif %}
    <span class="meta">{{ position }}</span>
  </div>
</header>
<main>
  <div class="statics">
  {% for panel in statics %}
    <figure>
      <img src="{{ panel.url }}" width="{{ big }}" height="{{ big }}">
      <figcaption>{{ panel.title }}</figcaption>
    </figure>
  {% endfor %}
  </div>
  {% if label_legend %}
  <div class="legend">{% for c, colour in label_legend %}<span><i style="background:{{ colour }}"></i>class {{ c }}</span>{% endfor %}</div>
  {% endif %}

  <table class="grid">
    <tr><th class="row"></th>{% for m in months %}<th>mo{{ '%02d'|format(m) }}</th>{% endfor %}</tr>
    {% for r in rows %}
    <tr>
      <th class="row">{{ r.title }}<span>{{ r.note }}</span></th>
      {% for cell in r.cells %}
      <td><figure>
        <img src="{{ cell.url }}" width="{{ size }}" height="{{ size }}" loading="lazy"
             title="{{ cell.tooltip }}">
        <figcaption>{{ cell.caption }}</figcaption>
      </figure></td>
      {% endfor %}
    </tr>
    {% endfor %}
  </table>

  <div class="legend" style="margin-top:14px">
    <span><b>SCL</b></span>
    {% for value, cname in scl_names %}<span><i style="background:{{ scl_colors[value] }}"></i>{{ value }} {{ cname }}</span>{% endfor %}
  </div>
  <div class="legend">
    <span><b>QA_PIXEL</b></span>
    {% for flag, colour in qa_legend %}<span><i style="background:{{ colour }}"></i>{{ flag }}</span>{% endfor %}
  </div>
</main>
"""
)


def hexcolor(rgb: tuple[int, int, int]) -> str:
    """CSS hex string for an (r, g, b) triple."""
    return "#{:02x}{:02x}{:02x}".format(*rgb)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

DATASETS: dict[str, Dataset] = {}


def create_app(datasets: list[Dataset], defaults: dict[str, Any]) -> Flask:
    """Build the Flask app serving the browser and the tile renderer."""
    app = Flask(__name__)
    DATASETS.clear()
    DATASETS.update({d.name: d for d in datasets})

    def get_dataset(ds_name: str) -> Dataset:
        dataset = DATASETS.get(ds_name)
        if dataset is None:
            abort(404)
        return dataset

    def query_string(**overrides: Any) -> str:
        params = dict(request.args.items(multi=False))
        for key, value in overrides.items():
            if value is None:
                params.pop(key, None)
            else:
                params[key] = value
        pairs = [f"{k}={v}" for k, v in params.items() if v != ""]
        return ("?" + "&".join(pairs)) if pairs else ""

    @app.route("/")
    def index() -> str:
        rows = [
            {
                "name": d.name,
                "path": str(d.path),
                "n": len(d.windows),
                "views": ", ".join(v.key for v in d.available_views()),
            }
            for d in datasets
        ]
        return render_template_string(INDEX_HTML, datasets=rows)

    @app.route("/ds/<ds_name>")
    def browse(ds_name: str) -> str:
        dataset = get_dataset(ds_name)
        split = request.args.get("split", "")
        group = request.args.get("group", "")
        label = request.args.get("label", "")
        query = request.args.get("q", "")
        thumb = request.args.get("thumb", defaults["thumb_view"])
        tmonth = int(request.args.get("tmonth", defaults["thumb_month"]))
        stretch = request.args.get("stretch", defaults["stretch"])
        per_page = max(1, int(request.args.get("per_page", defaults["per_page"])))
        page = max(0, int(request.args.get("page", 0)))

        filtered = filter_windows(dataset, split, label, group, query)
        n_pages = max(1, (len(filtered) + per_page - 1) // per_page)
        page = min(page, n_pages - 1)
        chunk = filtered[page * per_page : (page + 1) * per_page]

        size = defaults["thumb_size"]
        rows = []
        for row in chunk:
            label_value = row.get("label")
            rows.append(
                {
                    "name": row["name"],
                    "split": row.get("split", ""),
                    "label": label_value,
                    "color": hexcolor(TAB10[label_value % len(TAB10)])
                    if isinstance(label_value, int) and label_value != 255
                    else "#3a3a44",
                    "thumb": f"/img/{ds_name}/{row['group']}/{row['name']}/{thumb}/{tmonth}"
                    f"?size={size}&stretch={stretch}",
                    "url": f"/w/{ds_name}/{row['group']}/{row['name']}"
                    + query_string(page=None, per_page=None, thumb=None, tmonth=None),
                }
            )

        classes = sorted(
            {
                w["label"]
                for w in dataset.windows
                if isinstance(w.get("label"), int) and w["label"] != 255
            }
        )
        random_row = random.choice(filtered) if filtered else None
        return render_template_string(
            BROWSE_HTML,
            ds=dataset,
            rows=rows,
            size=size,
            n_filtered=len(filtered),
            n_pages=n_pages,
            page=page,
            per_page=per_page,
            split=split,
            group=group,
            label=label,
            q=query,
            thumb=thumb,
            tmonth=tmonth,
            stretch=stretch,
            splits=sorted(
                {w.get("split", "") for w in dataset.windows if w.get("split")}
            ),
            groups=sorted({w["group"] for w in dataset.windows}),
            classes=classes,
            thumb_views=[v.key for v in dataset.available_views()],
            page_url=lambda p: f"/ds/{ds_name}" + query_string(page=p),
            random_url=(
                f"/w/{ds_name}/{random_row['group']}/{random_row['name']}"
                if random_row
                else "#"
            ),
        )

    @app.route("/w/<ds_name>/<group>/<name>")
    def window_view(ds_name: str, group: str, name: str) -> str:
        dataset = get_dataset(ds_name)
        stretch = request.args.get("stretch", defaults["stretch"])
        size = int(request.args.get("size", defaults["tile_size"]))
        plow = float(request.args.get("plow", defaults["percentile_low"]))
        phigh = float(request.args.get("phigh", defaults["percentile_high"]))
        selected = request.args.getlist("views") or list(defaults["views"])

        available = dataset.available_views()
        monthly_views = [v for v in available if v.monthly and v.key in selected]
        static_views = [v for v in available if not v.monthly]

        items = window_items(ds_name, group, name)
        tile_query = f"?size={size}&stretch={stretch}&plow={plow}&phigh={phigh}"
        base = f"/img/{ds_name}/{group}/{name}"

        rows = []
        for view in monthly_views:
            cells = []
            for month in range(1, MONTHS + 1):
                info = items.get(view.layer_name(month), {})
                caption = info.get("date", "")
                cloud = info.get("cloud")
                if cloud is not None:
                    caption = f"{caption} {cloud:.0f}%"
                if not layer_present(dataset, group, name, view, month):
                    caption = "absent"
                cells.append(
                    {
                        "url": f"{base}/{view.key}/{month}{tile_query}",
                        "caption": caption,
                        "tooltip": "; ".join(info.get("names", []))
                        or view.layer_name(month),
                    }
                )
            rows.append({"title": view.title, "note": view.note, "cells": cells})

        big = max(size * 2, 160)
        statics = [
            {
                "title": view.title,
                "url": f"{base}/{view.key}/0?size={big}&stretch={stretch}"
                f"&plow={plow}&phigh={phigh}",
            }
            for view in static_views
        ]

        filtered = filter_windows(
            dataset,
            request.args.get("split", ""),
            request.args.get("label", ""),
            request.args.get("group", ""),
            request.args.get("q", ""),
        )
        keys = [(w["group"], w["name"]) for w in filtered]
        try:
            position_idx = keys.index((group, name))
        except ValueError:
            position_idx = -1
        row = next(
            (w for w in dataset.windows if w["group"] == group and w["name"] == name),
            {"group": group, "name": name, "split": "", "start": "", "end": ""},
        )

        def neighbour(offset: int) -> str | None:
            if position_idx < 0:
                return None
            target = position_idx + offset
            if not 0 <= target < len(keys):
                return None
            g, n = keys[target]
            return f"/w/{ds_name}/{g}/{n}" + query_string()

        label_array = read_band(dataset, group, name, "label_raster", "label")
        label_legend = []
        if label_array is not None:
            for value in sorted(np.unique(label_array.astype(np.int32)).tolist()):
                if value == 255:
                    continue
                label_legend.append((value, hexcolor(TAB10[value % len(TAB10)])))

        return render_template_string(
            WINDOW_HTML,
            ds=dataset,
            group=group,
            name=name,
            row=row,
            latlon=window_latlon(ds_name, group, name),
            months=list(range(1, MONTHS + 1)),
            rows=rows,
            statics=statics,
            size=size,
            big=big,
            stretch=stretch,
            plow=plow,
            phigh=phigh,
            all_views=[v for v in available if v.monthly],
            selected=set(selected),
            carry={
                k: v
                for k, v in request.args.items(multi=False)
                if k in {"split", "label", "q", "group"}
            },
            browse_url=f"/ds/{ds_name}" + query_string(stretch=None, size=None),
            prev_url=neighbour(-1),
            next_url=neighbour(1),
            position=(
                f"{position_idx + 1} / {len(keys)} in filter"
                if position_idx >= 0
                else ""
            ),
            label_legend=label_legend,
            scl_names=sorted(SCL_CLASS_NAMES.items()),
            scl_colors={k: hexcolor(v) for k, v in SCL_COLORS.items()},
            qa_legend=[(flag, hexcolor(colour)) for flag, colour in QA_COLORS.items()],
        )

    @app.route("/img/<ds_name>/<group>/<name>/<view_key>/<int:month>")
    def img(ds_name: str, group: str, name: str, view_key: str, month: int) -> Response:
        dataset = get_dataset(ds_name)
        view = VIEWS.get(view_key)
        if view is None:
            abort(404)
        size = int(request.args.get("size", defaults["tile_size"]))
        stretch = request.args.get("stretch", defaults["stretch"])
        plow = float(request.args.get("plow", defaults["percentile_low"]))
        phigh = float(request.args.get("phigh", defaults["percentile_high"]))
        rgb = render_view(
            dataset, group, name, view, max(1, month), stretch, plow, phigh
        )
        if rgb is None:
            return png_response(placeholder(size))
        return png_response(
            to_display_image(rgb, size, mark_center=(view.kind == "label"))
        )

    @app.route("/random/<ds_name>")
    def random_window(ds_name: str) -> Response:
        dataset = get_dataset(ds_name)
        row = random.choice(dataset.windows)
        return redirect(f"/w/{ds_name}/{row['group']}/{row['name']}")

    return app


# ---------------------------------------------------------------------------
# Headless dump
# ---------------------------------------------------------------------------


def dump_window(
    dataset: Dataset, spec: str, out_dir: Path, defaults: dict[str, Any]
) -> None:
    """Render every view of one window to PNGs and print a layer availability table."""
    parts = spec.split(":")
    if len(parts) == 3:
        group, name = parts[1], parts[2]
    else:
        if not dataset.windows:
            raise SystemExit(f"{dataset.name}: index is empty, nothing to dump")
        group, name = dataset.windows[0]["group"], dataset.windows[0]["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{dataset.name}  {group}/{name}  -> {out_dir}")
    print(f"{'view':22s} " + " ".join(f"{m:>2d}" for m in range(1, MONTHS + 1)))
    for view in dataset.available_views():
        marks = []
        months = range(1, MONTHS + 1) if view.monthly else [1]
        for month in months:
            rgb = render_view(
                dataset,
                group,
                name,
                view,
                month,
                defaults["stretch"],
                defaults["percentile_low"],
                defaults["percentile_high"],
            )
            if rgb is None:
                marks.append(" .")
                continue
            marks.append(" x")
            suffix = f"_mo{month:02d}" if view.monthly else ""
            to_display_image(
                rgb, defaults["tile_size"] * 2, mark_center=(view.kind == "label")
            ).save(out_dir / f"{view.key}{suffix}.png")
        print(f"{view.key:22s} " + " ".join(marks))
    items = window_items(dataset.name, group, name)
    print("\nscenes:")
    for layer in sorted(items):
        info = items[layer]
        cloud = "" if info["cloud"] is None else f"  cloud={info['cloud']:.0f}%"
        print(
            f"  {layer:24s} {info['date']}  n={info['count']}{cloud}  {info['names'][:1]}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def summarize_layers(layers: set[str]) -> str:
    """Collapse the twelve monthly siblings of a layer into one 'name_moNN x12' entry."""
    monthly: dict[str, int] = {}
    plain: list[str] = []
    for layer in layers:
        match = re.fullmatch(r"(.+)_mo(\d{2})", layer)
        if match:
            monthly[match.group(1)] = monthly.get(match.group(1), 0) + 1
        else:
            plain.append(layer)
    parts = [f"{prefix}_moNN x{count}" for prefix, count in sorted(monthly.items())]
    return ", ".join(parts + sorted(plain))


def main() -> None:
    """Parse arguments, build the window indexes, then serve (or dump)."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DEFAULT_DATASETS),
        help="registry dataset names to serve",
    )
    parser.add_argument(
        "--ds_path",
        action="append",
        default=[],
        help="extra dataset as name=/path/to/rslearn/dataset (repeatable)",
    )
    parser.add_argument(
        "--use_source_path",
        action="store_true",
        help="read the staging tree (source_path) instead of the registered weka_path",
    )
    parser.add_argument(
        "--cache_dir",
        default=str(Path.home() / ".cache" / "olmoearth_imagery_explorer"),
        help="where window indexes are cached",
    )
    parser.add_argument("--refresh", action="store_true", help="rebuild window indexes")
    parser.add_argument(
        "--no_labels",
        action="store_true",
        help="skip reading label rasters while indexing (faster; disables label filter)",
    )
    parser.add_argument(
        "--exclude_groups",
        default=r"_tessera_v2$",
        help="regex of window groups to skip (the tessera fetch groups by default)",
    )
    parser.add_argument("--index_workers", type=int, default=32)
    parser.add_argument("--tile_size", type=int, default=112)
    parser.add_argument("--thumb_size", type=int, default=96)
    parser.add_argument(
        "--thumb_view",
        default="s2_rgb",
        help="view used for browser thumbnails; s2_median looks better but reads 12x",
    )
    parser.add_argument("--thumb_month", type=int, default=7)
    parser.add_argument("--per_page", type=int, default=60)
    parser.add_argument(
        "--stretch",
        default="fixed",
        choices=["fixed", "window", "image"],
        help="default stretch: fixed reflectance range, per-window or per-image percentiles",
    )
    parser.add_argument("--percentile_low", type=float, default=2.0)
    parser.add_argument("--percentile_high", type=float, default=98.0)
    parser.add_argument("--views", nargs="*", default=list(DEFAULT_VIEWS))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--list",
        action="store_true",
        help="print dataset/window/layer summary and exit",
    )
    parser.add_argument(
        "--dump",
        default=None,
        help="render one window headlessly: <dataset>[:<group>:<name>]",
    )
    parser.add_argument("--dump_dir", default="./explorer_dump")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    datasets = resolve_datasets(args.datasets, args.ds_path, args.use_source_path)
    DATASETS.clear()
    DATASETS.update({d.name: d for d in datasets})
    cache_dir = Path(args.cache_dir)
    for dataset in datasets:
        build_index(
            dataset,
            cache_dir,
            args.exclude_groups,
            args.index_workers,
            not args.no_labels,
            args.refresh,
        )

    defaults = {
        "tile_size": args.tile_size,
        "thumb_size": args.thumb_size,
        "thumb_view": args.thumb_view,
        "thumb_month": args.thumb_month,
        "per_page": args.per_page,
        "stretch": args.stretch,
        "percentile_low": args.percentile_low,
        "percentile_high": args.percentile_high,
        "views": args.views,
    }

    if args.list:
        for dataset in datasets:
            splits: dict[str, int] = {}
            labels: dict[Any, int] = {}
            for row in dataset.windows:
                splits[row.get("split", "")] = splits.get(row.get("split", ""), 0) + 1
                if "label" in row:
                    labels[row["label"]] = labels.get(row["label"], 0) + 1
            print(f"\n{dataset.name}  {dataset.path}")
            print(f"  windows: {len(dataset.windows)}")
            print(f"  splits:  {splits}")
            if labels:
                print(
                    f"  centre labels: {dict(sorted(labels.items(), key=lambda kv: str(kv[0])))}"
                )
            print(f"  views:   {', '.join(v.key for v in dataset.available_views())}")
            seen = dataset.seen_layers()
            inferred = seen - dataset.declared_layers
            undermaterialized = {
                layer
                for layer in dataset.declared_layers
                if layer not in seen and not layer.startswith(("gse", "tessera"))
            }
            if inferred:
                print(f"  on disk but NOT in config.json: {summarize_layers(inferred)}")
            if undermaterialized:
                print(
                    "  in config.json but not on disk (sampled): "
                    f"{summarize_layers(undermaterialized)}"
                )
        return

    if args.dump:
        name = args.dump.split(":")[0]
        if name not in DATASETS:
            raise SystemExit(f"--dump names unknown dataset {name}")
        dump_window(DATASETS[name], args.dump, Path(args.dump_dir), defaults)
        return

    app = create_app(datasets, defaults)
    logger.info("serving on http://%s:%d", args.host, args.port)
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()

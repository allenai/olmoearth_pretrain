"""Every-capture S1 + S2 inputs for eval windows, from the Tessera v2 fetch groups.

Each AEF year-aligned eval dataset (and PASTIS) has a twin window per eval window,
same name, projection and bounds, in a fetch group of its staging dataset (built by
``tessera_v2_export.create_windows``): ``sentinel2_l2a_all`` (every S2 L2A capture
of the year, 10/20 m band sets + SCL, no B01/B09) and ``sentinel1_ascending_all`` /
``sentinel1_descending_all`` (every S1 RTC VV/VH capture, linear).

:func:`read_every_capture` turns one twin (cropped like the eval sample) into raw
per-modality capture series:

* captures of a modality on the same calendar day are merged (the INTERSECTS fetch
  returns one scene per overlapping S2 tile / S1 frame): earlier scenes win, later
  ones fill their empty pixels;
* a capture with less than ``min_valid_fraction`` valid pixels in the crop is
  dropped (scene edges, partial tiles);
* S1 ascending and descending are one ``sentinel1`` series.

:func:`union_timeline_sample` puts the series on one chronological timeline (S2 and
S1 captures on the same day share a step): ``[H, W, T_u, C]`` per modality with a
per-pixel validity mask, MISSING where the modality has no capture or a pixel is
nodata. This dense layout is what the eval path consumes (masked tokens are removed
before attention, so it is equivalent to per-modality time axes).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime

import h5py
import hdf5plugin
import numpy as np
from rslearn.dataset import Window
from upath import UPath

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.evals.datasets.tessera_v2_export import (
    DATASETS,
    S1_BAND_SET,
    S1_LAYERS,
    S2_BAND_SETS,
    S2_LAYER,
    _acquisition_times,
)
from olmoearth_pretrain.evals.embedding_materializer.providers import (
    RslearnWindowProvider,
)

logger = logging.getLogger(__name__)

LANDSAT_L2_LAYER = "landsat_l2_all"
LANDSAT_L2_BAND_SET = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B10"]

# Canonical S2 bands read from the fetch (B01 / B09 are not fetched).
S2_FETCHED_BANDS = [b for bands in S2_BAND_SETS for b in bands]
S1_NODATA = -32768.0

# Per-window h5 copies of the twins (build_every_capture_cache.py): one file instead
# of a few hundred GeoTIFFs per window. They hold each modality's same-day mosaics,
# unfiltered -- the mostly-empty-capture filter depends on the crop, so it runs at
# read time and cached reads equal GeoTIFF reads. A modality missing from a file
# (e.g. Landsat fetched after the file was built) is read from the GeoTIFFs.
CACHE_ROOT = "/weka/dfive-default/olmoearth_pretrain/every_capture_cache"


def _s2_valid(a: np.ndarray) -> np.ndarray:
    return (a > 0).any(axis=-1)


def _s1_valid(a: np.ndarray) -> np.ndarray:
    return (np.isfinite(a) & (a > 0) & (a != S1_NODATA)).all(axis=-1)


def cache_path(source: EveryCaptureSource, window_name: str) -> UPath:
    """The cached h5 of one twin window."""
    return (
        UPath(CACHE_ROOT) / UPath(source.root).name / source.group / f"{window_name}.h5"
    )


def _read_cached(
    source: EveryCaptureSource,
    window_name: str,
    modality: str,
    crop: tuple[slice, slice],
) -> tuple[list[np.ndarray], list[np.ndarray], list[datetime]] | None:
    """A modality's cropped same-day mosaics from the cache (None: not cached)."""
    path = cache_path(source, window_name)
    if not path.exists():
        return None
    with path.open("rb") as f, h5py.File(f, "r") as h5:
        if modality not in h5:
            return None
        group = h5[modality]
        days = [date.fromordinal(int(o)) for o in group["days"][()]]
        if not days:
            return [], [], []
        values = group["values"][(slice(None), *crop)].astype(np.float32)
        valid = group["valid"][(slice(None), *crop)].astype(bool)
    keep = [t for t in range(len(days)) if valid[t].mean() >= source.min_valid_fraction]
    return (
        [values[t] for t in keep],
        [valid[t] for t in keep],
        [datetime(days[t].year, days[t].month, days[t].day) for t in keep],
    )


def write_cache(window: Window, source: EveryCaptureSource) -> UPath:
    """Write one twin window's cache file (all modalities whose layers exist)."""
    series: dict[str, tuple[list[np.ndarray], list[np.ndarray], list[datetime]]] = {}
    s2_scenes, s2_times = _read_layer(window, S2_LAYER, S2_BAND_SETS)
    series[Modality.SENTINEL2_L2A.name] = _merge_same_day(
        s2_scenes, s2_times, _s2_valid, 0.0
    )
    s1_scenes, s1_times = [], []
    for layer in S1_LAYERS:
        scenes, times = _read_layer(window, layer, [S1_BAND_SET])
        s1_scenes += scenes
        s1_times += times
    series[Modality.SENTINEL1.name] = _merge_same_day(
        s1_scenes, s1_times, _s1_valid, 0.0
    )
    if any(name == LANDSAT_L2_LAYER for name, _ in window.list_completed_layers()):
        scenes, times = _read_layer(window, LANDSAT_L2_LAYER, [LANDSAT_L2_BAND_SET])
        series[Modality.LANDSAT_L2.name] = _merge_same_day(
            scenes, times, _s2_valid, 0.0
        )
    path = cache_path(source, window.name)
    write_series(path, series)
    return path


def write_series(
    path: UPath,
    series: dict[str, tuple[list[np.ndarray], list[np.ndarray], list[datetime]]],
) -> None:
    """Write modality -> unfiltered same-day mosaics to ``path`` (atomically)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f, h5py.File(f, "w") as h5:
        for modality, (values, valid, days) in series.items():
            group = h5.create_group(modality)
            group["days"] = np.array([d.toordinal() for d in days], dtype=np.int64)
            if not values:
                continue
            stacked = np.stack(values)
            chunks = (1, min(stacked.shape[1], 32), min(stacked.shape[2], 32))
            group.create_dataset(
                "values",
                data=stacked,
                chunks=(*chunks, stacked.shape[3]),
                **hdf5plugin.Zstd(clevel=3),
            )
            group.create_dataset(
                "valid", data=np.stack(valid), chunks=chunks, **hdf5plugin.Zstd()
            )
    tmp.rename(path)


@dataclass
class EveryCaptureSource:
    """Where an eval dataset's every-capture twins live.

    Args:
        root: the staging rslearn dataset holding the fetch group.
        group: the fetch group (e.g. ``ethiopia_crops_tessera_v2``).
        min_valid_fraction: drop captures with fewer valid crop pixels than this.
    """

    root: str
    group: str
    min_valid_fraction: float = 0.5


# PASTIS's fetch group lives in the original pastis_rslearn staging dataset.
PASTIS_STAGING = "/weka/dfive-default/rslearn-eai/datasets/pastis_rslearn"


def every_capture_source(dataset: str, source_path: str | None) -> EveryCaptureSource:
    """The every-capture twin group of a registry eval dataset.

    The ``*_year_aligned`` AEF datasets keep their fetch group in their staging
    dataset (the registry ``source_path``); PASTIS keeps it in ``pastis_rslearn``.
    """
    if dataset not in DATASETS:
        raise ValueError(f"{dataset} has no every-capture fetch group")
    group = DATASETS[dataset].fetch_group
    if dataset.startswith("pastis"):
        return EveryCaptureSource(root=PASTIS_STAGING, group=group)
    if not source_path:
        raise ValueError(f"{dataset} has no staging source_path in the registry")
    return EveryCaptureSource(root=source_path, group=group)


class EveryCaptureReader:
    """Name -> twin window lookup plus raster reading (one per dataset/worker)."""

    def __init__(self, source: EveryCaptureSource) -> None:
        """Initialize; windows are listed lazily on first use."""
        self.source = source
        self._windows: dict[str, Window] | None = None

    def window(self, name: str) -> Window:
        """The twin of the eval window ``name``."""
        if self._windows is None:
            provider = RslearnWindowProvider(
                UPath(self.source.root), groups=[self.source.group]
            )
            self._windows = {w.name: w for w in provider.load_windows()}
            logger.info(
                f"every-capture: {len(self._windows)} windows in "
                f"{self.source.root}/{self.source.group}"
            )
        if name not in self._windows:
            raise KeyError(f"no every-capture twin for window {name!r}")
        return self._windows[name]


def _read_layer(
    window: Window, layer: str, band_sets: tuple[list[str], ...] | list[list[str]]
) -> tuple[list[np.ndarray], list[datetime]]:
    """Every materialized scene of ``layer``: [(H, W, C)] float32, start times."""
    from rasterio.enums import Resampling
    from rslearn.utils.raster_format import GeotiffRasterFormat

    completed = sorted(
        idx for name, idx in window.list_completed_layers() if name == layer
    )
    if not completed:
        return [], []
    times = _acquisition_times(window, layer)
    raster_format = GeotiffRasterFormat()
    scenes = []
    for idx in completed:
        arrays = [
            raster_format.decode_raster(
                window.get_raster_dir(layer, bands, idx),
                window.projection,
                window.bounds,
                resampling=Resampling.bilinear,
            ).get_chw_array()
            for bands in band_sets
        ]
        scenes.append(np.concatenate(arrays, axis=0).transpose(1, 2, 0))
    return [s.astype(np.float32) for s in scenes], [times[i] for i in completed]


def _merge_same_day(
    scenes: list[np.ndarray],
    times: list[datetime],
    valid_fn: Callable[[np.ndarray], np.ndarray],
    min_valid_fraction: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[datetime]]:
    """Mosaic same-day scenes (first valid pixel wins); drop mostly-empty days."""
    order = np.argsort(np.array(times, dtype="datetime64[us]"), kind="stable")
    by_day: dict[tuple[int, int, int], list[int]] = {}
    for i in order:
        t = times[i]
        by_day.setdefault((t.year, t.month, t.day), []).append(int(i))
    out, valid_out, days = [], [], []
    for day, members in by_day.items():
        mosaic = scenes[members[0]].copy()
        valid = valid_fn(mosaic)
        for j in members[1:]:
            fill = valid_fn(scenes[j]) & ~valid
            mosaic[fill] = scenes[j][fill]
            valid |= fill
        if valid.mean() < min_valid_fraction:
            continue
        out.append(mosaic)
        valid_out.append(valid)
        days.append(datetime(*day))
    return out, valid_out, days


def read_every_capture(
    reader: EveryCaptureReader,
    window_name: str,
    rows: slice | None,
    cols: slice | None,
) -> dict[str, tuple[np.ndarray, np.ndarray, list[datetime]]]:
    """Raw every-capture series of one eval window, cropped like the eval sample.

    Returns modality -> ``(values [H, W, T, C], valid [H, W, T], dates)``:
    ``sentinel2_l2a`` in the canonical 12-band order (B01/B09 zero, flagged in
    :data:`S2_ABSENT_BANDS`), raw DN; ``sentinel1`` VV/VH, raw linear.
    """
    crop = (rows or slice(None), cols or slice(None))
    src = reader.source

    cached = _read_cached(src, window_name, Modality.SENTINEL2_L2A.name, crop)
    if cached is not None:
        s2, s2_valid, s2_days = cached
    else:
        window = reader.window(window_name)
        s2_scenes, s2_times = _read_layer(window, S2_LAYER, S2_BAND_SETS)
        s2_scenes = [s[crop] for s in s2_scenes]
        s2, s2_valid, s2_days = _merge_same_day(
            s2_scenes, s2_times, _s2_valid, src.min_valid_fraction
        )

    cached = _read_cached(src, window_name, Modality.SENTINEL1.name, crop)
    if cached is not None:
        s1, s1_valid, s1_days = cached
    else:
        window = reader.window(window_name)
        s1_scenes, s1_times = [], []
        for layer in S1_LAYERS:
            scenes, times = _read_layer(window, layer, [S1_BAND_SET])
            s1_scenes += [s[crop] for s in scenes]
            s1_times += times
        s1, s1_valid, s1_days = _merge_same_day(
            s1_scenes, s1_times, _s1_valid, src.min_valid_fraction
        )

    canonical = Modality.SENTINEL2_L2A.band_order
    out: dict[str, tuple[np.ndarray, np.ndarray, list[datetime]]] = {}
    if s2:
        stacked = np.stack(s2, axis=2)  # H W T C(fetched)
        full = np.zeros((*stacked.shape[:3], len(canonical)), dtype=np.float32)
        for k, band in enumerate(S2_FETCHED_BANDS):
            full[..., canonical.index(band)] = stacked[..., k]
        out[Modality.SENTINEL2_L2A.name] = (full, np.stack(s2_valid, axis=2), s2_days)
    if s1:
        out[Modality.SENTINEL1.name] = (
            np.stack(s1, axis=2),
            np.stack(s1_valid, axis=2),
            s1_days,
        )
    return out


def read_landsat_l2(
    reader: EveryCaptureReader,
    window_name: str,
    rows: slice | None,
    cols: slice | None,
) -> tuple[np.ndarray, np.ndarray, list[datetime]] | None:
    """Every Landsat C2 L2 capture of the twin (``landsat_l2_all``), cropped.

    Returns ``(values [H, W, T, 8], valid [H, W, T], dates)`` (raw DN, same-day
    scenes merged, mostly-empty captures dropped) or None if the window has none.
    """
    crop = (rows or slice(None), cols or slice(None))
    cached = _read_cached(reader.source, window_name, Modality.LANDSAT_L2.name, crop)
    if cached is not None:
        merged, valid, days = cached
    else:
        window = reader.window(window_name)
        scenes, times = _read_layer(window, LANDSAT_L2_LAYER, [LANDSAT_L2_BAND_SET])
        scenes = [s[crop] for s in scenes]
        merged, valid, days = _merge_same_day(
            scenes, times, _s2_valid, reader.source.min_valid_fraction
        )
    if not merged:
        return None
    return np.stack(merged, axis=2), np.stack(valid, axis=2), days


def slot_medians(
    values: np.ndarray,
    valid: np.ndarray,
    dates: list[datetime],
    slot_starts: list[datetime],
    slot_days: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel median of the captures falling in each slot ``[start, start + slot_days)``.

    The same per-pixel median the training composites use (Landsat has no QA band).
    Returns ``(composites [H, W, n_slots, C], valid [H, W, n_slots])``; a pixel
    with no valid capture in its slot is invalid.
    """
    out = np.zeros((*values.shape[:2], len(slot_starts), values.shape[3]), np.float32)
    ok = np.zeros((*values.shape[:2], len(slot_starts)), dtype=bool)
    for k, start in enumerate(slot_starts):
        members = [i for i, d in enumerate(dates) if 0 <= (d - start).days < slot_days]
        if not members:
            continue
        chunk = np.where(valid[:, :, members][..., None], values[:, :, members], np.nan)
        any_valid = valid[:, :, members].any(axis=2)
        with np.errstate(all="ignore"):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(chunk, axis=2)
        out[:, :, k] = np.where(any_valid[..., None], med, 0.0)
        ok[:, :, k] = any_valid
    return out, ok


S2_ABSENT_BANDS = [
    i
    for i, band in enumerate(Modality.SENTINEL2_L2A.band_order)
    if band not in S2_FETCHED_BANDS
]


def union_timeline_sample(
    series: dict[str, tuple[np.ndarray, np.ndarray, list[datetime]]],
    modalities: list[str],
    height: int,
    width: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[datetime]]:
    """Place each modality's captures on one chronological day timeline.

    Returns ``(values, valid, dates)``: modality -> ``[H, W, T_u, C]`` (zeros where
    absent) and ``[H, W, T_u]`` validity, and the ``T_u`` dates. A modality with no
    captures (or a sample with none at all) is all-invalid; ``T_u >= 1``.
    """
    all_days = sorted({d for m in modalities if m in series for d in series[m][2]})
    if not all_days:
        all_days = [datetime(1970, 1, 1)]
    step = {d: i for i, d in enumerate(all_days)}
    values, valid = {}, {}
    for m in modalities:
        num_bands = len(Modality.get(m).band_order)
        v = np.zeros((height, width, len(all_days), num_bands), dtype=np.float32)
        ok = np.zeros((height, width, len(all_days)), dtype=bool)
        if m in series:
            arr, arr_valid, days = series[m]
            idx = [step[d] for d in days]
            v[:, :, idx] = arr
            ok[:, :, idx] = arr_valid
        values[m], valid[m] = v, ok
    return values, valid, all_days

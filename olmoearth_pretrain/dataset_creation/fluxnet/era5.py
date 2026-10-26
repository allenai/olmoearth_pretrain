"""Stage 3: one ERA5-Land daily series per flux site, from the weka Zarr mirror.

The mirror (``common.ERA5_MIRROR``) holds the EarthDataHub ERA5-Land daily UTC
product that the pretraining corpus and the other ERA5 evals were ingested from,
rechunked to (14 bands, 3650 days, 10 lat, 10 lon), so a site's whole record is
five chunk reads instead of the remote store's 75-day x 15 x 30 deg chunks.

Each site takes the nearest ERA5-Land cell; when that cell is not land in
ERA5-Land (coastal and lakeside towers) the nearest land cell within
``MAX_CELL_OFFSET`` cells is used instead, and the offset is recorded.

Values are the cell's own ERA5-Land values (no spatial interpolation).
``validate`` checks the mirror against the remote store value for value
(identical on 2026-10-26, except that the mirror lacks every 29 February;
those days are filled with the mean of 28 February and 1 March). Windows materialized by rslearn from UTM windows
(DFMC, the pretraining corpus) carry resampled values instead: they correlate
> 0.99 with the nearest cell in flat terrain and less in mountains, and match
neither the nearest cell nor plain bilinear interpolation exactly.

Outputs under ``<root>/meta``: ``era5/<site_id>.npy`` ``(T_all, 14)`` float32 in
raw ECMWF units with ``NODATA`` gaps, ``era5_dates.npy`` (days since 1970-01-01)
and ``era5_cells.parquet``.

Usage::

    python -m olmoearth_pretrain.dataset_creation.fluxnet.era5 validate
    python -m olmoearth_pretrain.dataset_creation.fluxnet.era5 extract --workers 32
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zarr
from rslearn.data_sources.earthdatahub import ERA5LandDailyUTCv1

from .common import (
    DEFAULT_ROOT,
    ERA5_MIRROR,
    ERA5L_BANDS,
    NODATA,
    WINDOW_DAYS,
    haversine_km,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MAX_CELL_OFFSET = 3


class Era5Mirror:
    """Read access to the ERA5-Land daily mirror."""

    def __init__(self, path: Path = ERA5_MIRROR) -> None:
        """Open the store and cache its coordinates.

        Args:
            path: Mirror Zarr path.
        """
        group = zarr.open_group(str(path), mode="r")
        self.data = group["data"]
        bands = [str(b) for b in group["band"][:]]
        self.band_index = [bands.index(b) for b in ERA5L_BANDS]
        self.lat = np.asarray(group["latitude"][:], dtype=np.float64)
        self.lon = np.asarray(group["longitude"][:], dtype=np.float64)
        # Read lazily: the mask is stored in 10x10 chunks (53k files).
        self.valid = group["valid_mask"]
        times = np.asarray(group["time"][:]).astype("datetime64[D]")
        # The mirror has no 29 February (12 days over 1980-2026); its other
        # days equal the remote store. ``dates`` is the complete daily axis
        # (days since 1970-01-01) that ``series`` returns.
        self.native_dates = times.astype(np.int64)
        self.dates = np.arange(self.native_dates[0], self.native_dates[-1] + 1)
        self._native_rows = np.searchsorted(self.dates, self.native_dates)

    def nearest_cell(self, lat: float, lon: float) -> dict[str, Any] | None:
        """Nearest valid (land) cell to a point.

        Args:
            lat: Latitude (deg).
            lon: Longitude (deg, any convention).

        Returns:
            ``ilat, ilon, cell_lat, cell_lon, dist_km, offset`` (offset in cells
            from the nearest cell), or None if no valid cell is within
            ``MAX_CELL_OFFSET``.
        """
        i0 = int(np.abs(self.lat - lat).argmin())
        j0 = int(np.abs(self.lon - lon % 360.0).argmin())
        offsets = np.arange(-MAX_CELL_OFFSET, MAX_CELL_OFFSET + 1)
        rows = np.clip(i0 + offsets, 0, len(self.lat) - 1)
        cols = (j0 + offsets) % len(self.lon)
        valid = np.asarray(
            self.valid.get_orthogonal_selection((rows, cols)), dtype=bool
        )
        best: dict[str, Any] | None = None
        for a, di in enumerate(offsets):
            for b, dj in enumerate(offsets):
                i, j = int(rows[a]), int(cols[b])
                if i != i0 + di or not valid[a, b]:
                    continue
                dist = haversine_km(lat, lon, self.lat[i], self.lon[j])
                if best is None or dist < best["dist_km"]:
                    best = dict(
                        ilat=i, ilon=j, cell_lat=float(self.lat[i]),
                        cell_lon=float((self.lon[j] + 180.0) % 360.0 - 180.0),
                        dist_km=float(dist), offset=max(abs(di), abs(dj)),
                    )  # fmt: skip
        return best

    def native_series(self, ilat: int, ilon: int) -> np.ndarray:
        """One cell's record on the mirror's own (leap-day-free) axis.

        Args:
            ilat: Latitude index.
            ilon: Longitude index.

        Returns:
            ``(len(native_dates), 14)`` float32 in ``ERA5L_BANDS`` order, NaN
            as ``NODATA``.
        """
        block = np.asarray(self.data[:, :, ilat, ilon], dtype=np.float32)  # (C, T)
        block = block[self.band_index].T
        return np.where(np.isfinite(block), block, NODATA).astype(np.float32)

    def series(self, ilat: int, ilon: int) -> np.ndarray:
        """One cell's complete daily record, 29 February filled in.

        Each missing 29 February is the mean of 28 February and 1 March (NODATA
        if either is). At most one day of a 448-day window is filled this way.

        Args:
            ilat: Latitude index.
            ilon: Longitude index.

        Returns:
            ``(len(dates), 14)`` float32 in ``ERA5L_BANDS`` order.
        """
        native = self.native_series(ilat, ilon)
        full = np.full((len(self.dates), native.shape[1]), NODATA, np.float32)
        full[self._native_rows] = native
        missing = np.setdiff1d(np.arange(len(self.dates)), self._native_rows)
        before, after = full[missing - 1], full[missing + 1]
        valid = (before != NODATA) & (after != NODATA)
        full[missing] = np.where(valid, (before + after) / 2, NODATA)
        return full


# --- validation against the remote store -------------------------------------
def validate(num_checks: int, seed: int) -> None:
    """Check the mirror against the remote EarthDataHub store, value for value.

    Needs ``EARTHDATAHUB_TOKEN`` and zarr >= 3 (the remote store is zarr v3).

    Args:
        num_checks: Random (land cell, 30-day span) pairs to compare.
        seed: Sampling seed.
    """
    mirror = Era5Mirror()
    remote = ERA5LandDailyUTCv1(band_names=list(ERA5L_BANDS))._get_dataset()
    remote_dates = remote["valid_time"].values.astype("datetime64[D]").astype(np.int64)
    rng = np.random.default_rng(seed)
    exact = 0
    for _ in range(num_checks):
        # Random land cell between 60S and 70N, random 30-day span.
        while True:
            i, j = int(rng.integers(200, 1400)), int(rng.integers(0, len(mirror.lon)))
            if bool(mirror.valid[i, j]):
                break
        k = int(rng.integers(0, len(mirror.native_dates) - 30))
        span = mirror.native_dates[k : k + 30]
        r0 = int(np.searchsorted(remote_dates, span[0]))
        theirs = np.stack(
            [remote[b].isel(valid_time=slice(r0, r0 + 31), latitude=i, longitude=j).values
             for b in ERA5L_BANDS],
            axis=1,
        )  # fmt: skip
        # Compare on the mirror's own days (a 29 February in the span is skipped).
        theirs = theirs[np.searchsorted(remote_dates[r0 : r0 + 31], span)]
        theirs = np.where(np.isfinite(theirs), theirs, NODATA)
        ours = mirror.native_series(i, j)[k : k + 30]
        max_abs = float(np.abs(theirs - ours).max())
        exact += int(max_abs == 0.0)
        logger.info("cell (%.1f, %.1f) from %s: max|diff| %.3g", mirror.lat[i], mirror.lon[j],
                    np.datetime64(int(span[0]), "D"), max_abs)  # fmt: skip
    logger.info(
        "validation: %d/%d spans identical to the remote store", exact, num_checks
    )


# --- extraction ---------------------------------------------------------------
_MIRROR: Era5Mirror | None = None


def _init_worker() -> None:
    """Open the mirror once per worker process."""
    global _MIRROR
    _MIRROR = Era5Mirror()


def extract_site(job: tuple[str, float, float, str]) -> dict[str, Any]:
    """Worker: write one site's ERA5 series.

    Args:
        job: ``(site_id, lat, lon, output dir)``.

    Returns:
        The site's cell row for ``era5_cells.parquet`` (``ilat`` -1 when no
        land cell was found).
    """
    site_id, lat, lon, out_dir = job
    assert _MIRROR is not None
    cell = _MIRROR.nearest_cell(lat, lon)
    if cell is None:
        return dict(site_id=site_id, ilat=-1, ilon=-1)
    series = _MIRROR.series(cell["ilat"], cell["ilon"])
    np.save(Path(out_dir) / f"{site_id}.npy", series)
    return dict(
        site_id=site_id, **cell, n_nodata_days=int((series == NODATA).any(axis=1).sum())
    )


def extract(root: Path, workers: int) -> None:
    """Write every site's series, the shared date axis and ``era5_cells.parquet``.

    Args:
        root: Dataset root.
        workers: Reader processes.
    """
    sites = pd.read_parquet(root / "meta" / "sites.parquet")
    sites = sites[sites["lat"].notna() & sites["lon"].notna()]
    out_dir = root / "meta" / "era5"
    out_dir.mkdir(parents=True, exist_ok=True)
    mirror = Era5Mirror()
    np.save(root / "meta" / "era5_dates.npy", mirror.dates)
    jobs = [
        (r.site_id, float(r.lat), float(r.lon), str(out_dir))
        for r in sites.itertuples()
    ]
    with multiprocessing.get_context("spawn").Pool(
        workers, initializer=_init_worker
    ) as pool:
        rows = list(pool.imap_unordered(extract_site, jobs))
    cells = pd.DataFrame(rows).sort_values("site_id")
    cells.to_parquet(root / "meta" / "era5_cells.parquet")
    found = cells["ilat"] >= 0
    logger.info(
        "ERA5 cells: %d/%d sites; offset counts %s; dates %s..%s (window %d days)",
        int(found.sum()), len(cells), cells.loc[found, "offset"].value_counts().to_dict(),
        datetime.fromtimestamp(int(mirror.dates[0]) * 86400, UTC).date(),
        datetime.fromtimestamp(int(mirror.dates[-1]) * 86400, UTC).date(), WINDOW_DAYS,
    )  # fmt: skip


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("step", choices=["validate", "extract"])
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--num-checks", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.step == "validate":
        validate(args.num_checks, args.seed)
    else:
        extract(Path(args.root), args.workers)


if __name__ == "__main__":
    main()

"""Stage 1: MODIS 8-day maximum snow extent -> snow counts per 0.1 deg cell.

Reads the MOD10A2 collection 6.1 ``Maximum_Snow_Extent`` COGs that the
Planetary Computer serves (anonymous, SAS token; 2000-02-18 onward) and bins
every 500 m pixel into the 0.1 deg ERA5-Land cell holding its centre. Per cell
and 8-day composite, three counts: land pixels with snow (any snow day within
the 8 days), land pixels seen snow-free, and land pixels of unknown state
(cloud on every day, polar night, no decision, missing). Water (inland water,
ocean, lake ice) is left out, so snow cover fraction = snow / (snow + no snow).

``screen`` keeps the sinusoidal tiles with seasonal snow: every land tile of
the rows 70-20 N (v01-v06) and 20-60 S (v11-v14) is read on two late-winter
composites of 2011 and 2018 (NH 10 and 26 February, SH 5 and 21 August) and
kept when any of them has ``MIN_SNOW_CELLS`` cells at least half snow-covered.

``counts`` then reads every composite of every kept tile, one job per
(tile, year); a job's output is the year's 46 fixed composite slots (start days
1, 9, .., 361 of the year; a missing composite leaves zeros).

Outputs under ``<root>/raw/modis``: ``tiles.csv`` (screening) and
``parts/h<HH>v<VV>_<YYYY>.npz`` with ``lat10``, ``lon10`` (cell centres in
tenths of a degree), ``npix`` (pixels of the tile in each cell, water
included), ``slot_start`` (days since 1970-01-01), ``slot_days`` and
``counts`` (``(46, n_cells, 3)`` uint16: snow, no snow, unknown).

Usage::

    python -m olmoearth_pretrain.dataset_creation.snow.modis screen --workers 48
    python -m olmoearth_pretrain.dataset_creation.snow.modis counts --workers 48
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import multiprocessing
import random
import re
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from rasterio.io import MemoryFile

from .common import (
    ASSET,
    BLOB_ROOT,
    DEFAULT_ROOT,
    FILL,
    GRID_X_MIN,
    GRID_Y_MAX,
    NO_SNOW,
    PRODUCT,
    SAS_URL,
    SNOW,
    SPHERE_RADIUS,
    TILE_PX,
    TILE_SIZE_M,
    WATER,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HTTP_TIMEOUT = (30, 180)
MAX_ATTEMPTS = 10
TOKEN_CACHE = Path(tempfile.gettempdir()) / "snow_modis_sas_token.json"
TOKEN_WAIT_S = 3600
# One connection pool per process: keep-alive also spares the DNS lookups that
# fail intermittently under load.
SESSION = requests.Session()
FIRST_YEAR = 2000
SLOTS_PER_YEAR = 46
NH_ROWS = range(1, 7)
SH_ROWS = range(11, 15)
SCREEN_YEARS = (2011, 2018)
SCREEN_DOYS = {"nh": (41, 57), "sh": (217, 233)}
MIN_SNOW_CELLS = 3
MIN_CLEAR_PIXELS = 20

# Pixel value -> 0 ignored (water, fill), 1 snow, 2 no snow, 3 unknown land.
CLASS_LUT = np.full(256, 3, np.int64)
CLASS_LUT[SNOW] = 1
CLASS_LUT[NO_SNOW] = 2
CLASS_LUT[list(WATER)] = 0
CLASS_LUT[FILL] = 0

BLOB_RE = re.compile(
    rf"{PRODUCT}\.A(\d{{4}})(\d{{3}})\.h\d\dv\d\d\.061\.(\d+)_{ASSET}\.tif$"
)


class SasToken:
    """Planetary Computer SAS token for the MODIS COG container, refreshed before expiry.

    The token endpoint rate-limits per client IP, and every Beaker job leaves
    through the same address, so tokens come first from a relay file that a
    process outside the cluster refreshes (``--token-file``, written by
    ``scripts/era5_supervised/v0/snow/token_relay.py``), then from a per-job
    cache, and only then from the endpoint. The job's worker processes share
    one token under a lock.
    """

    def __init__(self, relay: Path | None = None, cache: Path = TOKEN_CACHE) -> None:
        """Start without a token.

        Args:
            relay: Token file refreshed from outside the cluster (optional).
            cache: Token cache file shared by the job's processes.
        """
        self.relay = relay
        self.cache = cache
        self.token = ""
        self.expiry = 0.0

    def _load(self, path: Path | None) -> bool:
        """Take the token stored in ``path`` when it is valid for ten more minutes.

        Args:
            path: Token file (``token``, ``expiry`` in epoch seconds).

        Returns:
            Whether a valid token was loaded.
        """
        if path is None or not path.exists():
            return False
        try:
            body = json.loads(path.read_text())
        except (OSError, ValueError):
            return False
        if float(body["expiry"]) < time.time() + 600:
            return False
        self.token, self.expiry = body["token"], float(body["expiry"])
        return True

    def get(self) -> str:
        """A token valid for at least ten more minutes.

        Returns:
            The SAS query string.
        """
        if time.time() < self.expiry - 600:
            return self.token
        with open(self.cache.with_suffix(".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            deadline = time.time() + TOKEN_WAIT_S
            while not (self._load(self.relay) or self._load(self.cache)):
                try:
                    response = requests.get(SAS_URL, timeout=HTTP_TIMEOUT)
                    response.raise_for_status()
                except requests.RequestException as exc:
                    if time.time() > deadline:
                        raise
                    logger.warning(
                        "SAS token unavailable (%s); waiting for the relay", exc
                    )
                    time.sleep(30 * random.uniform(0.8, 1.2))
                    continue
                body = response.json()
                expiry = datetime.fromisoformat(
                    body["msft:expiry"].replace("Z", "+00:00")
                ).timestamp()
                self.cache.write_text(
                    json.dumps(dict(token=body["token"], expiry=expiry))
                )
        return self.token


def http_get(url: str) -> requests.Response:
    """GET with retries, backing off longer on rate limits.

    Args:
        url: URL.

    Returns:
        The successful response.
    """
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = SESSION.get(url, timeout=HTTP_TIMEOUT)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            status = getattr(exc.response, "status_code", None)
            wait = min(120.0, 2.0 * 2**attempt) * random.uniform(0.5, 1.5)
            if status == 429:
                wait = max(wait, float(exc.response.headers.get("Retry-After", 10)))
            # The exception text holds the full URL; keep the SAS token out of logs.
            logger.warning("GET failed (%s, HTTP %s), retry in %.0f s: %s",
                           type(exc).__name__, status, wait, url.split("?")[0])  # fmt: skip
            time.sleep(wait)
    raise AssertionError("unreachable")


def composite_blobs(h: int, v: int, year: int, sas: SasToken) -> dict[int, str]:
    """``Maximum_Snow_Extent`` COGs of one tile and year.

    Args:
        h: Horizontal tile index.
        v: Vertical tile index.
        year: Calendar year.
        sas: Token source.

    Returns:
        ``{day of year of the composite start: blob name}``; when a composite
        was produced more than once, the latest production is kept.
    """
    prefix = f"{PRODUCT}/{h:02d}/{v:02d}/{year}"
    found: dict[int, tuple[str, str]] = {}
    marker = ""
    while True:
        url = (f"{BLOB_ROOT}?restype=container&comp=list&prefix={prefix}"
               f"&maxresults=5000&{sas.get()}" + (f"&marker={marker}" if marker else ""))  # fmt: skip
        root = ET.fromstring(http_get(url).content)
        for blob in root.iter("Blob"):
            name = blob.findtext("Name") or ""
            match = BLOB_RE.search(name)
            if match is None:
                continue
            doy, produced = int(match.group(2)), match.group(3)
            if doy not in found or produced > found[doy][0]:
                found[doy] = (produced, name)
        marker = root.findtext("NextMarker") or ""
        if not marker:
            break
    return {doy: name for doy, (_, name) in found.items()}


def read_composite(name: str, sas: SasToken) -> np.ndarray:
    """One COG, downloaded whole (one request) and decoded in memory.

    Args:
        name: Blob name.
        sas: Token source.

    Returns:
        ``(2400, 2400)`` uint8 class values.
    """
    content = http_get(f"{BLOB_ROOT}/{name}?{sas.get()}").content
    with MemoryFile(content) as memfile, memfile.open() as src:
        array = src.read(1)
    if array.shape != (TILE_PX, TILE_PX):
        raise ValueError(f"{name}: unexpected shape {array.shape}")
    return array


def tile_cells(h: int, v: int) -> dict[str, np.ndarray]:
    """Map every pixel of a tile to its 0.1 deg cell.

    Args:
        h: Horizontal tile index.
        v: Vertical tile index.

    Returns:
        ``pixels`` (flat indices of the pixels on the globe), ``index`` (their
        cell, into the cell arrays), ``lat10``, ``lon10`` (cell centres in
        tenths of a degree, longitude in [-1800, 1800)) and ``npix``.
    """
    step = TILE_SIZE_M / TILE_PX
    centres = (np.arange(TILE_PX) + 0.5) * step
    x = GRID_X_MIN + h * TILE_SIZE_M + centres
    y = GRID_Y_MAX - v * TILE_SIZE_M - centres
    lat = y / SPHERE_RADIUS
    lon = x[None, :] / (SPHERE_RADIUS * np.cos(lat)[:, None])
    on_globe = (np.abs(lon) <= np.pi).ravel()
    lat10 = np.broadcast_to(np.rint(np.degrees(lat) * 10)[:, None], lon.shape).ravel()
    lon10 = np.rint(np.degrees(lon) * 10).ravel()
    lon10 = np.where(lon10 >= 1800, lon10 - 3600, lon10)
    pixels = np.flatnonzero(on_globe)
    key = (
        (lat10[pixels].astype(np.int64) + 900) * 3600
        + lon10[pixels].astype(np.int64)
        + 1800
    )
    keys, index, npix = np.unique(key, return_inverse=True, return_counts=True)
    return dict(
        pixels=pixels, index=index.astype(np.int64),
        lat10=(keys // 3600 - 900).astype(np.int16), lon10=(keys % 3600 - 1800).astype(np.int16),
        npix=npix.astype(np.int32),
    )  # fmt: skip


def cell_counts(array: np.ndarray, cells: dict[str, np.ndarray]) -> np.ndarray:
    """Snow / no-snow / unknown land pixel counts per cell of one composite.

    Args:
        array: Composite class values.
        cells: Output of :func:`tile_cells`.

    Returns:
        ``(n_cells, 3)`` uint16.
    """
    n = len(cells["lat10"])
    cls = CLASS_LUT[array.ravel()[cells["pixels"]]]
    counts = np.bincount(cells["index"] * 4 + cls, minlength=n * 4).reshape(n, 4)
    return counts[:, 1:].astype(np.uint16)


def year_slots(year: int) -> tuple[np.ndarray, np.ndarray]:
    """The fixed composite slots of a year.

    Args:
        year: Calendar year.

    Returns:
        ``(start day of year [46], length in days [46])``.
    """
    doys = np.arange(1, 366, 8)
    year_days = (date(year + 1, 1, 1) - date(year, 1, 1)).days
    return doys, np.minimum(8, year_days - doys + 1)


# --- screening ------------------------------------------------------------------
_SAS: SasToken | None = None


def _init_worker(relay: str) -> None:
    """One token source per worker process.

    Args:
        relay: Relay token file.
    """
    global _SAS
    _SAS = SasToken(Path(relay))


def screen_tile(job: tuple[int, int]) -> dict[str, Any]:
    """Worker: snow-covered cells of one tile on its late-winter screening composites.

    Args:
        job: ``(h, v)``.

    Returns:
        ``h, v, land`` (whether the tile exists), ``snow_cells`` (max over the
        composites of cells with snow cover fraction >= 0.5).
    """
    h, v = job
    assert _SAS is not None
    hemisphere = "nh" if v <= 8 else "sh"
    cells: dict[str, np.ndarray] | None = None
    best, land = 0, False
    for year in SCREEN_YEARS:
        blobs = composite_blobs(h, v, year, _SAS)
        land |= bool(blobs)
        for doy in SCREEN_DOYS[hemisphere]:
            if doy not in blobs:
                continue
            if cells is None:
                cells = tile_cells(h, v)
            counts = cell_counts(read_composite(blobs[doy], _SAS), cells).astype(
                np.int64
            )
            clear = counts[:, 0] + counts[:, 1]
            snowy = (clear >= MIN_CLEAR_PIXELS) & (2 * counts[:, 0] >= clear)
            best = max(best, int(snowy.sum()))
    return dict(h=h, v=v, land=land, snow_cells=best)


def screen(root: Path, workers: int, relay: str) -> pd.DataFrame:
    """Screen every candidate tile; writes ``raw/modis/tiles.csv``.

    Args:
        root: Dataset root.
        workers: Processes.
        relay: Relay token file.

    Returns:
        The tile table (``kept`` column).
    """
    out = root / "raw" / "modis" / "tiles.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    jobs = [(h, v) for v in (*NH_ROWS, *SH_ROWS) for h in range(36)]
    with multiprocessing.get_context("spawn").Pool(
        workers, initializer=_init_worker, initargs=(relay,)
    ) as pool:
        rows = list(pool.imap_unordered(screen_tile, jobs))
    tiles = pd.DataFrame(rows).sort_values(["v", "h"]).reset_index(drop=True)
    tiles["kept"] = tiles["snow_cells"] >= MIN_SNOW_CELLS
    tiles.to_csv(out, index=False)
    logger.info("%d land tiles, %d kept (>= %d snow cells)", int(tiles["land"].sum()),
                int(tiles["kept"].sum()), MIN_SNOW_CELLS)  # fmt: skip
    return tiles


# --- counts ---------------------------------------------------------------------
def part_path(root: Path, h: int, v: int, year: int) -> Path:
    """Output file of one (tile, year) job.

    Args:
        root: Dataset root.
        h: Horizontal tile index.
        v: Vertical tile index.
        year: Calendar year.

    Returns:
        The path.
    """
    return root / "raw" / "modis" / "parts" / f"h{h:02d}v{v:02d}_{year}.npz"


def count_year(job: tuple[int, int, int, str]) -> tuple[str, int]:
    """Worker: per-cell counts of every composite of one tile and year.

    Args:
        job: ``(h, v, year, dataset root)``.

    Returns:
        ``(part name, composites read)``.
    """
    h, v, year, root = job
    assert _SAS is not None
    out = part_path(Path(root), h, v, year)
    cells = tile_cells(h, v)
    doys, lengths = year_slots(year)
    counts = np.zeros((SLOTS_PER_YEAR, len(cells["lat10"]), 3), np.uint16)
    blobs = composite_blobs(h, v, year, _SAS)
    for k, doy in enumerate(doys):
        if int(doy) in blobs:
            counts[k] = cell_counts(read_composite(blobs[int(doy)], _SAS), cells)
    start = np.array([(date(year, 1, 1) + timedelta(days=int(d) - 1) - date(1970, 1, 1)).days
                      for d in doys], np.int32)  # fmt: skip
    # Written to a side file, then renamed, so a killed job never leaves a
    # truncated part behind.
    part = out.with_suffix(".part.npz")
    np.savez_compressed(
        part, lat10=cells["lat10"], lon10=cells["lon10"], npix=cells["npix"],
        slot_start=start, slot_days=lengths.astype(np.int8), counts=counts,
    )  # fmt: skip
    part.rename(out)
    return out.name, len(blobs)


def count_all(
    root: Path, workers: int, last_year: int, tiles_arg: str | None, relay: str
) -> None:
    """Every (kept tile, year) part that is not on disk yet.

    Args:
        root: Dataset root.
        workers: Processes.
        last_year: Last calendar year read.
        tiles_arg: Optional comma-separated ``hHHvVV`` list overriding the screening.
        relay: Relay token file.
    """
    if tiles_arg:
        tiles = [(int(t[1:3]), int(t[4:6])) for t in tiles_arg.split(",")]
    else:
        table = pd.read_csv(root / "raw" / "modis" / "tiles.csv")
        tiles = [(int(r.h), int(r.v)) for r in table[table["kept"]].itertuples()]
    (root / "raw" / "modis" / "parts").mkdir(parents=True, exist_ok=True)
    jobs = [(h, v, year, str(root)) for year in range(FIRST_YEAR, last_year + 1)
            for h, v in tiles if not part_path(root, h, v, year).exists()]  # fmt: skip
    logger.info("%d tiles, %d (tile, year) parts to read", len(tiles), len(jobs))
    started = time.time()
    total = 0
    with multiprocessing.get_context("spawn").Pool(
        workers, initializer=_init_worker, initargs=(relay,)
    ) as pool:
        for k, (name, n) in enumerate(pool.imap_unordered(count_year, jobs), start=1):
            total += n
            if k % 50 == 0 or k == len(jobs):
                rate = total / (time.time() - started)
                logger.info("[%d/%d parts] %d composites, %.1f/s; last %s",
                            k, len(jobs), total, rate, name)  # fmt: skip


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("step", choices=["screen", "counts"])
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--last-year", type=int, default=2026)
    ap.add_argument(
        "--tiles", default=None, help="hHHvVV list (counts; default: screened)"
    )
    ap.add_argument(
        "--token-file",
        default=None,
        help="relay token file (default <root>/raw/modis/sas_token.json)",
    )
    args = ap.parse_args()
    root = Path(args.root)
    relay = args.token_file or str(root / "raw" / "modis" / "sas_token.json")
    if args.step == "screen":
        screen(root, args.workers, relay)
    else:
        count_all(root, args.workers, args.last_year, args.tiles, relay)


if __name__ == "__main__":
    main()

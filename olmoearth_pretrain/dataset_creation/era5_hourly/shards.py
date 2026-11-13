"""Stage 2: build one shard of hourly series from the EarthDataHub store.

For each store tile of the shard (``shards.json``), every band is read one
(1440 h, 50, 100) chunk at a time over the tile's time chunks (in parallel,
with retries), the tile's cells are kept, accumulated bands are turned into
hourly amounts (:func:`common.deaccumulate`) and each window's 8,760 rows are
cut from its cell's series (``NODATA`` where the store has no value).

The shard is ``shards/shard_XXX.npy.partial`` until every tile is written,
then renamed to ``shards/shard_XXX.npy``. Rows go to their byte offset with
``pwrite`` and ``fsync`` and are read back before the tile's marker
(``progress/shard_XXX/tile_*.json``) is written: on weka, rows written through
``np.memmap`` by a restarted job were seen to read back as zeros. A rerun skips
the tiles that have a marker, so retries and preemptions resume.

Usage::

    python -m olmoearth_pretrain.dataset_creation.era5_hourly.shards --shard 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import (
    BANDS,
    DEFAULT_ROOT,
    LAT_CHUNK,
    LON_CHUNK,
    N_LAT,
    NODATA,
    TIME_CHUNK,
    WINDOW_HOURS,
    deaccumulate,
    open_store,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROW_BYTES = WINDOW_HOURS * len(BANDS) * 4
READ_ATTEMPTS = 8


def create_shard_file(path: Path, n_windows: int) -> None:
    """Create an ``.npy`` file of the shard's shape (contents unwritten).

    Args:
        path: File to create.
        n_windows: Rows.
    """
    header = {
        "descr": np.lib.format.dtype_to_descr(np.dtype("<f4")),
        "fortran_order": False,
        "shape": (n_windows, WINDOW_HOURS, len(BANDS)),
    }
    with open(path, "wb") as f:
        np.lib.format.write_array_header_1_0(f, header)
        f.truncate(f.tell() + n_windows * ROW_BYTES)


def write_rows(path: Path, at: int, block: np.ndarray) -> None:
    """Write ``block`` at byte ``at``, sync it, and check its ends read back.

    Args:
        path: Shard file.
        at: Byte offset.
        block: Contiguous float32 rows.
    """
    data = memoryview(np.ascontiguousarray(block, dtype="<f4")).cast("B")
    fd = os.open(path, os.O_WRONLY)
    try:
        written = 0
        while written < len(data):
            written += os.pwrite(fd, data[written : written + (1 << 30)], at + written)
        os.fsync(fd)
    finally:
        os.close(fd)
    span = min(len(data), 1 << 20)
    fd = os.open(path, os.O_RDONLY)
    try:
        head = os.pread(fd, span, at)
        tail = os.pread(fd, span, at + len(data) - span)
    finally:
        os.close(fd)
    if head != bytes(data[:span]) or tail != bytes(data[len(data) - span :]):
        raise OSError(f"rows written at byte {at} of {path} did not read back")


def read_chunk(array: Any, selection: tuple[slice, slice, slice]) -> np.ndarray:
    """Read one store chunk, retrying with backoff.

    Args:
        array: Zarr array.
        selection: Chunk-aligned selection.

    Returns:
        The block as float32.
    """
    for attempt in range(READ_ATTEMPTS):
        try:
            return np.asarray(array[selection], dtype=np.float32)
        except Exception as exc:
            if attempt == READ_ATTEMPTS - 1:
                raise
            wait = min(300, 10 * 2**attempt)
            logger.warning(
                "read %s %s failed (%r); retry in %ds", array.path, selection, exc, wait
            )
            time.sleep(wait)
    raise AssertionError("unreachable")


def build_tile(
    group: Any, tile: dict[str, Any], windows: pd.DataFrame, fetch_workers: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """Every band of one tile's windows.

    Args:
        group: Hourly store root.
        tile: The tile's ``shards.json`` entry.
        windows: The tile's index rows, in row order.
        fetch_workers: Concurrent chunk reads.

    Returns:
        ``(n_windows, WINDOW_HOURS, n_bands)`` float32 and tile stats.
    """
    lat0, lon0 = tile["tile_lat"] * LAT_CHUNK, tile["tile_lon"] * LON_CHUNK
    lat1 = min(lat0 + LAT_CHUNK, N_LAT)
    cells, cell_of = np.unique(
        windows[["ilat", "ilon"]].to_numpy(), axis=0, return_inverse=True
    )
    cell_of = cell_of.reshape(-1)
    ci, cj = cells[:, 0] - lat0, cells[:, 1] - lon0
    chunks = tile["time_chunks"]
    t0 = chunks[0] * TIME_CHUNK
    span = (chunks[-1] + 1) * TIME_CHUNK - t0
    take = (windows["first_hour"].to_numpy() - t0)[:, None] + np.arange(WINDOW_HOURS)[
        None, :
    ]
    out = np.empty((len(windows), WINDOW_HOURS, len(BANDS)), np.float32)
    nodata = {}
    window_has_nodata = np.zeros(len(windows), dtype=bool)
    # Bands are read together when a tile has few time chunks, so about
    # fetch_workers chunks are always in flight.
    per_group = max(1, fetch_workers // len(chunks))
    for g0 in range(0, len(BANDS), per_group):
        members = range(g0, min(g0 + per_group, len(BANDS)))
        started = time.time()
        arrays = {b: group[BANDS[b].source] for b in members}
        series = {b: np.full((span, len(cells)), np.nan, np.float32) for b in members}

        def fetch(job: tuple[int, int]) -> None:
            b, tc = job
            lo, hi = tc * TIME_CHUNK, min((tc + 1) * TIME_CHUNK, arrays[b].shape[0])
            block = read_chunk(
                arrays[b],
                (slice(lo, hi), slice(lat0, lat1), slice(lon0, lon0 + LON_CHUNK)),
            )
            series[b][lo - t0 : hi - t0] = block[:, ci, cj]

        with ThreadPoolExecutor(fetch_workers) as ex:
            list(ex.map(fetch, [(b, tc) for b in members for tc in chunks]))
        for b in members:
            band = BANDS[b]
            full = (
                deaccumulate(series.pop(b), t0) if band.accumulated else series.pop(b)
            )
            values = full[take, cell_of[:, None]]
            missing = ~np.isfinite(values)
            out[:, :, b] = np.where(missing, NODATA, values)
            nodata[band.name] = int(missing.sum())
            window_has_nodata |= missing.any(axis=1)
        logger.info("  %s: %d chunks in %.1fs", " ".join(BANDS[b].name for b in members),
                    len(members) * len(chunks), time.time() - started)  # fmt: skip
    stats = dict(
        n_windows=len(windows), n_cells=len(cells), chunks_per_band=len(chunks),
        nodata_values=nodata, windows_with_nodata=int(window_has_nodata.sum()),
    )  # fmt: skip
    return out, stats


def run(root: Path, shard: int, fetch_workers: int) -> None:
    """Build one shard (resuming from its tile markers).

    Args:
        root: Output root (``plan`` outputs).
        shard: Shard number.
        fetch_workers: Concurrent chunk reads.
    """
    final = root / "shards" / f"shard_{shard:03d}.npy"
    if final.exists():
        logger.info("%s exists; nothing to do", final)
        return
    spec = json.loads((root / "shards.json").read_text())["shards"][shard]
    assert spec["shard"] == shard
    index = pd.read_parquet(root / "index.parquet")
    index = index[index["shard"] == shard].sort_values("row")
    assert (
        len(index) == spec["n_windows"]
        and (index["row"].to_numpy() == np.arange(len(index))).all()
    )
    partial = final.with_name(final.name + ".partial")
    markers = root / "progress" / f"shard_{shard:03d}"
    markers.mkdir(parents=True, exist_ok=True)
    final.parent.mkdir(parents=True, exist_ok=True)
    if not partial.exists():
        create_shard_file(partial, spec["n_windows"])
    offset = np.load(partial, mmap_mode="r").offset
    group = open_store()
    started = time.time()
    tiles_stats = []
    for k, tile in enumerate(spec["tiles"]):
        marker = (
            markers
            / f"tile_{tile['tile_lat']:02d}_{tile['tile_lon']:02d}_{tile['row_start']:05d}.json"
        )
        if marker.exists():
            tiles_stats.append(json.loads(marker.read_text()))
            continue
        tile_started = time.time()
        logger.info(
            "shard %d tile %d/%d (lat %d, lon %d): %d windows, %d time chunks",
            shard, k + 1, len(spec["tiles"]), tile["tile_lat"], tile["tile_lon"],
            tile["row_end"] - tile["row_start"], len(tile["time_chunks"]),
        )  # fmt: skip
        rows = index.iloc[tile["row_start"] : tile["row_end"]]
        block, stats = build_tile(group, tile, rows, fetch_workers)
        write_rows(partial, offset + tile["row_start"] * ROW_BYTES, block)
        stats.update(tile=tile, seconds=round(time.time() - tile_started, 1))
        marker.write_text(json.dumps(stats))
        tiles_stats.append(stats)
        logger.info(
            "  tile done in %.0fs (%d windows with no-data values)",
            stats["seconds"], stats["windows_with_nodata"],
        )  # fmt: skip
    os.replace(partial, final)
    summary = dict(
        shard=shard, n_windows=spec["n_windows"], n_tiles=len(spec["tiles"]),
        seconds_this_run=round(time.time() - started, 1),
        seconds_all_tiles=round(sum(t["seconds"] for t in tiles_stats), 1),
        chunks=sum(t["chunks_per_band"] for t in tiles_stats) * len(BANDS),
        windows_with_nodata=sum(t["windows_with_nodata"] for t in tiles_stats),
    )  # fmt: skip
    (root / "progress" / f"shard_{shard:03d}.json").write_text(
        json.dumps(summary, indent=2)
    )
    logger.info("shard %d done: %s", shard, summary)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--fetch-workers", type=int, default=32)
    args = ap.parse_args()
    run(Path(args.root), args.shard, args.fetch_workers)


if __name__ == "__main__":
    main()

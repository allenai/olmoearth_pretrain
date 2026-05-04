"""Create rslearn windows from a corpus CSV of (sample_id, lon, lat, start_time[, end_time]).

This is the entry point for building an OlmoEarth Pretrain dataset from an arbitrary
list of samples where each sample carries its own location, time, and stable identifier.
Unlike `from_lon_lat_list`, this flow does not sample timestamps from NAIP/Sentinel-2
availability: it trusts the caller's `start_time` and emits sample-id-mode windows so
downstream exporters can write per-sample metadata (bounds, CRS, per-axis resolution,
tile time, start/end time) keyed by `sample_id`.

Input CSV columns (with header row):
    sample_id, lon, lat, start_time [, end_time]

- sample_id: stable string id, used as the rslearn window name and the OlmoEarth
  `example_id`. Must be unique within the dataset.
- lon, lat: WGS84 decimal degrees.
- start_time: ISO-8601 timestamp (UTC if no tz).
- end_time: optional. Defaults to start_time + WINDOW_DURATION.

Windows are created in a single group `res_{resolution}` with `WINDOW_SIZE x WINDOW_SIZE`
pixel bounds centered on the reprojected point, in the local UTM/UPS zone. The
contract is carried on `window.options` via the `use_grid_reference=false` +
`sample_id=...` flags read by `olmoearth_pretrain.dataset_creation.util.get_window_metadata`.

If `--config-path` is provided, the command also writes `ds_path/config.json` from that
file. The default is a **copy** so the dataset stays self-contained if the repo branch
moves or the source file disappears. Use `--config-mode symlink` if you prefer a symlink
to a stable canonical path on the same filesystem.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import shapely
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Window
from rslearn.dataset.storage.storage import WindowStorage
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from upath import UPath

from olmoearth_pretrain.dataset_creation.constants import (
    SAMPLE_ID_OPTION,
    USE_GRID_REFERENCE_OPTION,
    WINDOW_DURATION,
    WINDOW_SIZE,
)

from rslearn.dataset.storage.file import FileWindowStorage

from .util import star_imap

DEFAULT_RESOLUTION = 10.0
DEFAULT_CONFIG_MODE = 'copy'


@dataclass(frozen=True)
class CorpusEntry:
    """A single row from the corpus CSV."""

    sample_id: str
    lon: float
    lat: float
    start_time: datetime
    end_time: datetime

    def __post_init__(self) -> None:  # noqa: D105
        if not self.sample_id:
            raise ValueError('sample_id must be non-empty')
        if self.end_time <= self.start_time:
            raise ValueError(
                f'end_time ({self.end_time}) must be strictly after start_time '
                f'({self.start_time}) for sample {self.sample_id!r}'
            )


def _parse_time(raw: str) -> datetime:
    """Parse an ISO-8601 timestamp, defaulting to UTC if no tz is present."""
    value = raw.strip()
    if not value:
        raise ValueError('empty timestamp')
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def attach_dataset_config(
    ds_path: UPath | str,
    config_path: UPath | str,
    mode: str = DEFAULT_CONFIG_MODE,
    force: bool = False,
) -> Path:
    """Attach `config.json` into the dataset root.

    Args:
        ds_path: dataset root directory.
        config_path: source config file to expose as `config.json`.
        mode: one of `symlink` or `copy`.
        force: replace an existing config if needed.

    Returns:
        filesystem path to the attached config inside the dataset root.
    """
    if mode not in {'symlink', 'copy'}:
        raise ValueError(f'unsupported config attach mode {mode!r}')

    ds_root = Path(str(ds_path))
    src = Path(str(config_path)).resolve()
    dst = ds_root / 'config.json'

    if not src.exists():
        raise FileNotFoundError(src)

    ds_root.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        if not force:
            if dst.is_symlink() and dst.resolve() == src:
                return dst
            if dst.exists() and not dst.is_symlink() and dst.resolve() == src:
                return dst
            raise FileExistsError(
                f'{dst} already exists; use force=True to replace it'
            )
        if dst.is_dir() and not dst.is_symlink():
            raise IsADirectoryError(dst)
        dst.unlink()

    if mode == 'symlink':
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)
    return dst


def read_corpus_csv(path: UPath | str) -> list[CorpusEntry]:
    """Read a corpus CSV into a list of `CorpusEntry`.

    Args:
        path: path to the corpus CSV (local or fsspec).

    Returns:
        parsed entries, in file order. Duplicate sample_ids raise.
    """
    csv_path = UPath(path)
    entries: list[CorpusEntry] = []
    seen: set[str] = set()
    with csv_path.open('r') as f:
        reader = csv.DictReader(f)
        required = {'sample_id', 'lon', 'lat', 'start_time'}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f'corpus CSV missing required columns {required}; '
                f'got {reader.fieldnames}'
            )
        for row_idx, row in enumerate(reader):
            sample_id = row['sample_id'].strip()
            if sample_id in seen:
                raise ValueError(f'duplicate sample_id {sample_id!r} at row {row_idx}')
            seen.add(sample_id)

            start_time = _parse_time(row['start_time'])
            end_raw = (row.get('end_time') or '').strip()
            end_time = _parse_time(end_raw) if end_raw else start_time + WINDOW_DURATION

            entries.append(
                CorpusEntry(
                    sample_id=sample_id,
                    lon=float(row['lon']),
                    lat=float(row['lat']),
                    start_time=start_time,
                    end_time=end_time,
                )
            )
    return entries


def _entry_to_window_spec(
    entry: CorpusEntry,
    resolution: float,
    window_size: int,
) -> tuple[Projection, tuple[int, int, int, int], tuple[datetime, datetime]]:
    """Compute projection/bounds/time_range for one corpus entry.

    Bounds are a square of `window_size` pixels centered on the reprojected point, in
    the local UTM/UPS zone at the given resolution.
    """
    projection = get_utm_ups_projection(entry.lon, entry.lat, resolution, -resolution)
    src_geom = STGeometry(WGS84_PROJECTION, shapely.Point(entry.lon, entry.lat), None)
    dst_geom = src_geom.to_projection(projection)
    cx = int(round(dst_geom.shp.x))
    cy = int(round(dst_geom.shp.y))
    half = window_size // 2
    bounds = (cx - half, cy - half, cx - half + window_size, cy - half + window_size)
    time_range = (entry.start_time, entry.end_time)
    return projection, bounds, time_range


def _create_one_corpus_window(
    entry: CorpusEntry,
    storage: WindowStorage,
    group: str,
    resolution: float,
    window_size: int,
) -> None:
    projection, bounds, time_range = _entry_to_window_spec(entry, resolution, window_size)
    window = Window(
        storage=storage,
        group=group,
        name=entry.sample_id,
        projection=projection,
        bounds=bounds,
        time_range=time_range,
        options={
            USE_GRID_REFERENCE_OPTION: 'false',
            SAMPLE_ID_OPTION: entry.sample_id,
        },
    )
    window.save()


def create_corpus_windows(
    ds_path: UPath,
    entries: list[CorpusEntry],
    resolution: float = DEFAULT_RESOLUTION,
    window_size: int = WINDOW_SIZE,
    workers: int = 32,
) -> None:
    """Create one rslearn window per corpus entry at the given resolution.

    All windows land in group `res_{resolution}` and are named by `sample_id`. Adding
    new modalities to the dataset later does not require recreating these windows: the
    single `ds_path` can carry any number of rslearn layers over the same windows.

    Args:
        ds_path: rslearn dataset path. A `config.json` may exist (in which case its
            storage backend is honored) but is not required for window creation.
        entries: corpus rows to materialize as windows. Must have unique sample_ids.
        resolution: meters/pixel for the window projection (default 10 m).
        window_size: square tile size in pixels (default WINDOW_SIZE).
        workers: number of parallel worker processes for window creation.
    """
    storage = FileWindowStorage(ds_path)
    group = f'res_{resolution}'

    jobs = [
        dict(
            entry=entry,
            storage=storage,
            group=group,
            resolution=resolution,
            window_size=window_size,
        )
        for entry in entries
    ]

    if workers <= 1 or len(jobs) <= 1:
        for job in tqdm.tqdm(jobs, desc='Create corpus windows'):
            _create_one_corpus_window(**job)
        return

    with multiprocessing.Pool(workers) as p:
        for _ in tqdm.tqdm(
            star_imap(p, _create_one_corpus_window, jobs),
            desc='Create corpus windows',
            total=len(jobs),
        ):
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Create rslearn windows for an OlmoEarth Pretrain corpus dataset from a CSV '
            'of (sample_id, lon, lat, start_time[, end_time]).'
        )
    )
    parser.add_argument(
        '--ds_path', type=str, required=True, help='Target rslearn dataset path.'
    )
    parser.add_argument(
        '--fname',
        type=str,
        required=True,
        help='Corpus CSV path (local or fsspec-supported URI).',
    )
    parser.add_argument(
        '--resolution',
        type=float,
        default=DEFAULT_RESOLUTION,
        help='Meters per pixel (default 10).',
    )
    parser.add_argument(
        '--window_size',
        type=int,
        default=WINDOW_SIZE,
        help=f'Square tile size in pixels (default {WINDOW_SIZE}).',
    )
    parser.add_argument(
        '--config-path',
        type=str,
        default=None,
        help='Optional config to expose as ds_path/config.json.',
    )
    parser.add_argument(
        '--config-mode',
        choices=['symlink', 'copy'],
        default=DEFAULT_CONFIG_MODE,
        help='How to attach --config-path into the dataset root (default copy).',
    )
    parser.add_argument(
        '--force-config',
        action='store_true',
        help='Replace an existing ds_path/config.json if it differs.',
    )
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()

    if args.config_path:
        attach_dataset_config(
            ds_path=UPath(args.ds_path),
            config_path=UPath(args.config_path),
            mode=args.config_mode,
            force=args.force_config,
        )

    entries = read_corpus_csv(args.fname)
    create_corpus_windows(
        UPath(args.ds_path),
        entries,
        resolution=args.resolution,
        window_size=args.window_size,
        workers=args.workers,
    )


__all__ = [
    'CorpusEntry',
    'DEFAULT_CONFIG_MODE',
    'DEFAULT_RESOLUTION',
    'attach_dataset_config',
    'create_corpus_windows',
    'read_corpus_csv',
]

"""Tests for corpus-driven rslearn window creation."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from upath import UPath

from olmoearth_pretrain.dataset_creation.constants import (
    SAMPLE_ID_OPTION,
    USE_GRID_REFERENCE_OPTION,
    WINDOW_DURATION,
    WINDOW_SIZE,
)
from olmoearth_pretrain.dataset_creation.create_windows.from_corpus import (
    CorpusEntry,
    attach_dataset_config,
    create_corpus_windows,
    read_corpus_csv,
)
from olmoearth_pretrain.dataset_creation.util import get_window_metadata

SAMPLE_ROWS = [
    ('sample_sf', -122.41, 37.77, '2023-06-15T00:00:00Z'),
    ('sample_paris', 2.35, 48.86, '2022-03-01T00:00:00+00:00'),
    ('sample_sydney', 151.21, -33.87, '2024-01-10T00:00:00Z'),
]


def _write_corpus_csv(path: Path, include_end_time: bool = False) -> None:
    header = ['sample_id', 'lon', 'lat', 'start_time']
    if include_end_time:
        header.append('end_time')
    lines = [','.join(header)]
    for sid, lon, lat, start in SAMPLE_ROWS:
        row = [sid, f'{lon}', f'{lat}', start]
        if include_end_time:
            end = (
                datetime.fromisoformat(start.replace('Z', '+00:00'))
                + timedelta(days=30)
            ).isoformat()
            row.append(end)
        lines.append(','.join(row))
    path.write_text("\n".join(lines) + "\n")


def test_read_corpus_csv_defaults_end_time(tmp_path: Path) -> None:
    csv_path = tmp_path / 'corpus.csv'
    _write_corpus_csv(csv_path)

    entries = read_corpus_csv(csv_path)

    assert [e.sample_id for e in entries] == [r[0] for r in SAMPLE_ROWS]
    for entry, (_, lon, lat, start) in zip(entries, SAMPLE_ROWS):
        assert entry.lon == pytest.approx(lon)
        assert entry.lat == pytest.approx(lat)
        expected_start = datetime.fromisoformat(start.replace('Z', '+00:00'))
        assert entry.start_time == expected_start
        assert entry.end_time == expected_start + WINDOW_DURATION


def test_read_corpus_csv_honors_explicit_end_time(tmp_path: Path) -> None:
    csv_path = tmp_path / 'corpus.csv'
    _write_corpus_csv(csv_path, include_end_time=True)

    entries = read_corpus_csv(csv_path)
    for entry in entries:
        assert entry.end_time == entry.start_time + timedelta(days=30)


def test_read_corpus_csv_rejects_duplicates(tmp_path: Path) -> None:
    csv_path = tmp_path / 'corpus.csv'
    csv_path.write_text(
        'sample_id,lon,lat,start_time\n'
        'dup,-122.4,37.7,2023-01-01T00:00:00Z\n'
        'dup,-122.4,37.7,2023-02-01T00:00:00Z\n'
    )
    with pytest.raises(ValueError, match='duplicate sample_id'):
        read_corpus_csv(csv_path)


def test_read_corpus_csv_requires_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / 'corpus.csv'
    csv_path.write_text('lon,lat,start_time\n-122.4,37.7,2023-01-01T00:00:00Z\n')
    with pytest.raises(ValueError, match='missing required columns'):
        read_corpus_csv(csv_path)


def test_corpus_entry_validates_time_order() -> None:
    t = datetime(2023, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match='end_time'):
        CorpusEntry(sample_id='s', lon=0.0, lat=0.0, start_time=t, end_time=t)


def test_attach_dataset_config_copies_by_default(tmp_path: Path) -> None:
    ds_path = tmp_path / 'ds'
    src = tmp_path / 'config_corpus.json'
    payload = '{"layers": {}, "tile_store": {"class_path": "x"}}'
    src.write_text(payload)

    dst = attach_dataset_config(ds_path, src)

    assert dst == ds_path / 'config.json'
    assert not dst.is_symlink()
    assert dst.read_text() == payload


def test_attach_dataset_config_symlink_mode(tmp_path: Path) -> None:
    ds_path = tmp_path / 'ds'
    src = tmp_path / 'config_corpus.json'
    src.write_text('{"layers": {}, "tile_store": {"class_path": "x"}}')

    dst = attach_dataset_config(ds_path, src, mode='symlink')

    assert dst == ds_path / 'config.json'
    assert dst.is_symlink()
    assert dst.resolve() == src.resolve()


def test_attach_dataset_config_rejects_existing_without_force(tmp_path: Path) -> None:
    ds_path = tmp_path / 'ds'
    ds_path.mkdir()
    src = tmp_path / 'config_corpus.json'
    src.write_text('{"layers": {}, "tile_store": {"class_path": "x"}}')
    (ds_path / 'config.json').write_text('different')

    with pytest.raises(FileExistsError):
        attach_dataset_config(ds_path, src)


def test_attach_dataset_config_force_replaces_existing(tmp_path: Path) -> None:
    ds_path = tmp_path / 'ds'
    ds_path.mkdir()
    src = tmp_path / 'config_corpus.json'
    src.write_text('{"layers": {}, "tile_store": {"class_path": "x"}}')
    (ds_path / 'config.json').write_text('different')

    dst = attach_dataset_config(ds_path, src, force=True)

    assert not dst.is_symlink()
    assert dst.read_text() == src.read_text()


def test_create_corpus_windows_sample_id_mode(tmp_path: Path) -> None:
    csv_path = tmp_path / 'corpus.csv'
    _write_corpus_csv(csv_path)
    entries = read_corpus_csv(csv_path)

    ds_path = UPath(tmp_path / 'ds')
    create_corpus_windows(ds_path, entries, resolution=10.0, workers=1)

    group_dir = ds_path / 'windows' / 'res_10.0'
    assert group_dir.exists()
    assert {p.name for p in group_dir.iterdir()} == {e.sample_id for e in entries}

    for entry in entries:
        meta_fname = group_dir / entry.sample_id / 'metadata.json'
        with meta_fname.open() as f:
            meta = json.load(f)

        assert meta['options'][USE_GRID_REFERENCE_OPTION] == 'false'
        assert meta['options'][SAMPLE_ID_OPTION] == entry.sample_id

        bounds = meta['bounds']
        assert bounds[2] - bounds[0] == WINDOW_SIZE
        assert bounds[3] - bounds[1] == WINDOW_SIZE

        time_range = meta['time_range']
        assert datetime.fromisoformat(time_range[0]) == entry.start_time
        assert datetime.fromisoformat(time_range[1]) == entry.end_time

        assert meta['projection']['x_resolution'] == 10.0
        assert meta['projection']['y_resolution'] == -10.0


def test_window_metadata_roundtrips_through_hadrien_contract(tmp_path: Path) -> None:
    csv_path = tmp_path / 'corpus.csv'
    _write_corpus_csv(csv_path, include_end_time=True)
    entries = read_corpus_csv(csv_path)

    ds_path = UPath(tmp_path / 'ds')
    create_corpus_windows(ds_path, entries, resolution=10.0, workers=1)

    from rslearn.dataset.storage.file import FileWindowStorage

    storage = FileWindowStorage(ds_path)
    windows = storage.get_windows(groups=['res_10.0'])
    assert len(windows) == len(entries)

    by_name = {w.name: w for w in windows}
    for entry in entries:
        window = by_name[entry.sample_id]
        wm = get_window_metadata(window)
        assert wm.use_grid_reference is False
        assert wm.sample_id == entry.sample_id
        assert wm.col is None and wm.row is None
        assert wm.bounds is not None
        assert wm.bounds[2] - wm.bounds[0] == WINDOW_SIZE
        assert wm.time == entry.start_time + WINDOW_DURATION // 2

"""Validate an every-capture (allcap) OlmoEarth dataset end to end.

Checks the consolidated per-modality CSVs and a random sample of the generated h5
files: dataset presence and shapes, per-modality timestamp alignment and monotonic
ordering, capture-count distributions, and the SCL-derived cloud-fraction
distribution (which should span ~0-1 if no cloud filtering happened).

Usage:
    python -m scripts.data.validate_allcap \
        --olmoearth_path /weka/.../dataset/osm_allcaptures_pilot1k \
        --h5_dir /weka/.../h5py_data_.../<modalities>/<N> \
        --num_h5 200
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict

import h5py
import numpy as np
from upath import UPath

from olmoearth_pretrain.data.constants import Modality, TimeSpan
from olmoearth_pretrain.dataset_creation.util import get_modality_dir

ALLCAP_MODALITIES = [
    Modality.SENTINEL2_L2A,
    Modality.SENTINEL2_SCL,
    Modality.SENTINEL1,
    Modality.LANDSAT_L2,
]


def check_csvs(olmoearth_path: UPath) -> None:
    """Check the consolidated allcap CSVs: capture counts, ordering, scene ids."""
    for modality in ALLCAP_MODALITIES:
        modality_dir = get_modality_dir(olmoearth_path, modality, TimeSpan.ALL)
        csv_fname = olmoearth_path / f"{modality_dir.name}.csv"
        if not csv_fname.exists():
            print(f"[csv] {modality.name}: MISSING {csv_fname}")
            continue

        captures_per_tile: dict[str, list[str]] = defaultdict(list)
        missing_scene_ids = 0
        with csv_fname.open() as f:
            for row in csv.DictReader(f):
                key = f"{row['crs']}_{row['col']}_{row['row']}" if row["col"] else row["sample_id"]
                captures_per_tile[key].append(row["start_time"])
                if not row.get("scene_id"):
                    missing_scene_ids += 1

        counts = np.array([len(v) for v in captures_per_tile.values()])
        unsorted_tiles = sum(1 for v in captures_per_tile.values() if v != sorted(v))
        print(
            f"[csv] {modality.name}: {len(counts)} tiles, captures/tile "
            f"min={counts.min()} p50={int(np.median(counts))} mean={counts.mean():.1f} "
            f"max={counts.max()}; unsorted_tiles={unsorted_tiles}; "
            f"missing_scene_ids={missing_scene_ids}"
        )
        if unsorted_tiles:
            print(f"[csv] {modality.name}: ERROR - timestamps not chronological")


def check_h5(h5_dir: UPath, num_h5: int, seed: int) -> None:
    """Open a random sample of h5 files and validate shapes, timestamps, cloud stats."""
    sample_files = sorted(h5_dir.glob("sample_*.h5"))
    print(f"[h5] {len(sample_files)} sample files in {h5_dir}")
    if not sample_files:
        return
    rng = random.Random(seed)
    picked = rng.sample(sample_files, min(num_h5, len(sample_files)))

    t_counts: dict[str, list[int]] = defaultdict(list)
    cloud_fracs: list[np.ndarray] = []
    bytes_per_file: list[int] = []
    errors = 0

    for fname in picked:
        bytes_per_file.append(fname.stat().st_size)
        with fname.open("rb") as f, h5py.File(f, "r") as h5:
            keys = set(h5.keys())
            if "latlon" not in keys:
                print(f"[h5] {fname.name}: ERROR - missing latlon")
                errors += 1
            for modality in ALLCAP_MODALITIES:
                if modality.name not in keys:
                    continue
                data = h5[modality.name]
                ts_key = f"timestamps_{modality.name}"
                if ts_key not in keys:
                    print(f"[h5] {fname.name}: ERROR - {modality.name} without {ts_key}")
                    errors += 1
                    continue
                ts = h5[ts_key][:]
                t = data.shape[2]
                if ts.shape[0] != t:
                    print(
                        f"[h5] {fname.name}: ERROR - {modality.name} T={t} but "
                        f"{ts_key} has {ts.shape[0]} rows"
                    )
                    errors += 1
                if data.shape[3] != modality.num_bands:
                    print(
                        f"[h5] {fname.name}: ERROR - {modality.name} has "
                        f"{data.shape[3]} bands, expected {modality.num_bands}"
                    )
                    errors += 1
                # (day, month0, year) rows -> comparable ints; must be non-decreasing.
                ordinal = ts[:, 2] * 10000 + ts[:, 1] * 100 + ts[:, 0]
                if np.any(np.diff(ordinal) < 0):
                    print(f"[h5] {fname.name}: ERROR - {modality.name} timestamps not sorted")
                    errors += 1
                t_counts[modality.name].append(t)
            if "sentinel2_scl_cloud_fraction" in keys:
                cf = h5["sentinel2_scl_cloud_fraction"][:]
                if cf.min() < 0 or cf.max() > 1:
                    print(f"[h5] {fname.name}: ERROR - cloud fraction outside [0,1]")
                    errors += 1
                cloud_fracs.append(cf)
            elif Modality.SENTINEL2_SCL.name in keys:
                print(f"[h5] {fname.name}: ERROR - SCL present but no cloud fraction")
                errors += 1

    for name, counts_list in sorted(t_counts.items()):
        arr = np.array(counts_list)
        print(
            f"[h5] {name}: T min={arr.min()} p50={int(np.median(arr))} "
            f"mean={arr.mean():.1f} max={arr.max()} (n={len(arr)})"
        )
    if cloud_fracs:
        all_cf = np.concatenate(cloud_fracs)
        pcts = np.percentile(all_cf, [0, 10, 25, 50, 75, 90, 100])
        print(
            f"[h5] cloud fraction over {len(all_cf)} captures: "
            + " ".join(f"p{p}={v:.2f}" for p, v in zip([0, 10, 25, 50, 75, 90, 100], pcts))
        )
        print(f"[h5] captures >50% cloudy: {(all_cf > 0.5).mean() * 100:.1f}%")
    mb = np.array(bytes_per_file) / 1e6
    print(f"[h5] file size MB: min={mb.min():.1f} p50={np.median(mb):.1f} max={mb.max():.1f}")
    print(f"[h5] checked {len(picked)} files, {errors} errors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate an allcap OlmoEarth dataset")
    parser.add_argument("--olmoearth_path", type=str, required=True)
    parser.add_argument("--h5_dir", type=str, default=None, help="h5 sample directory")
    parser.add_argument("--num_h5", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    check_csvs(UPath(args.olmoearth_path))
    if args.h5_dir:
        check_h5(UPath(args.h5_dir), args.num_h5, args.seed)

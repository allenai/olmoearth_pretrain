"""Generate 96x96 JPEG thumbnails from H5 sample files.

Extracts the Sentinel-2 RGB channels from each H5 file and saves a
thumbnail image suitable for the embedding map click-to-view feature.

Usage:
    python generate_thumbnails.py \
        --h5-dir /path/to/h5_files \
        --output-dir /path/to/_viz/thumbnails \
        --workers 8

    # Only generate thumbnails for samples in a metadata parquet:
    python generate_thumbnails.py \
        --h5-dir /path/to/h5_files \
        --output-dir /path/to/_viz/thumbnails \
        --metadata /path/to/metadata.parquet \
        --workers 8

Thumbnail naming: each H5 file ``sample_123.h5`` produces
``sample_123.jpg`` (stem preserved, extension replaced).  This matches
the metadata ``filename`` column used by the embedding map.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


def inspect_h5(path: str) -> None:
    """Print the full structure of an H5 file (useful for discovery)."""
    import h5py

    with h5py.File(path, "r") as f:

        def _show(name: str, obj: object) -> None:
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
            else:
                print(f"  {name}/")

        print(f"Structure of {path}:")
        f.visititems(_show)


# ---- Adapt this section to your H5 layout --------------------------------
# The functions below assume a common multi-sensor H5 layout.  Override
# S2_KEY, RGB_BANDS, and TIMESTEP as needed.

S2_KEY = "sentinel2_l2a"  # H5 group/dataset key for Sentinel-2
RGB_BANDS = (2, 1, 0)  # B4 (Red), B3 (Green), B2 (Blue) — 10m bands first ordering
THUMBNAIL_SIZE = 128

# Preferred summer months (0-indexed) per hemisphere.
_SUMMER_MONTHS_NH = {5, 6, 7}  # June, July, August
_SUMMER_MONTHS_SH = {11, 0, 1}  # December, January, February


def _pick_timestep(months: np.ndarray, lat: float) -> int:
    """Return the stored-array index of the best summer timestep.

    *months* is a 1-D array of 0-indexed month values for each stored
    timestep.  Prefers the middle summer month (July / January) when
    available, otherwise any summer month, otherwise the middle timestep.
    """
    target = _SUMMER_MONTHS_NH if lat >= 0 else _SUMMER_MONTHS_SH
    preferred = 6 if lat >= 0 else 0  # July / January

    candidates = [i for i, m in enumerate(months) if int(m) in target]
    if not candidates:
        return len(months) // 2

    for i in candidates:
        if int(months[i]) == preferred:
            return i
    return candidates[len(candidates) // 2]


def extract_rgb(h5_path: str) -> np.ndarray | None:
    """Extract an (H, W, 3) uint8 RGB array from an H5 sample file.

    Picks the best summer timestep based on hemisphere (June-August in
    the north, December-February in the south).  Falls back to the
    middle timestep when no summer image is available.

    Returns None if the expected keys are not found.
    """
    import h5py
    import hdf5plugin  # noqa: F401 – registers zstd/blosc filters
    import numpy as np

    with h5py.File(h5_path, "r") as f:
        if S2_KEY not in f:
            return None
        data = f[S2_KEY]
        if isinstance(data, h5py.Group):
            if "data" in data:
                data = data["data"]
            else:
                return None
        arr = data[()]  # (H, W, T, C)

        lat = float(f["latlon"][()][0])
        timestamps = f["timestamps"][()]  # (12, 3) [day, month, year]
        mask_key = f"missing_timesteps_masks/{S2_KEY}"
        if mask_key in f:
            present = np.array(f[mask_key][()], dtype=bool)  # True = stored
            stored_months = timestamps[present, 1]
        else:
            stored_months = timestamps[:, 1]

    t = _pick_timestep(stored_months, lat)
    rgb = arr[:, :, t, :][:, :, list(RGB_BANDS)]  # (H, W, 3)

    # Per-channel percentile stretch to uint8
    rgb = rgb.astype(np.float32)
    for c in range(3):
        band = rgb[:, :, c]
        p2, p98 = np.percentile(band, [2, 98])
        if p98 > p2:
            rgb[:, :, c] = (band - p2) / (p98 - p2) * 255
        else:
            rgb[:, :, c] = band / max(band.max(), 1) * 255
    rgb = np.clip(rgb, 0, 255)

    return rgb.astype(np.uint8)


def process_one(h5_path: str, output_dir: str, quality: int) -> str | None:
    """Process a single H5 file -> JPEG thumbnail. Returns output path or None."""
    from PIL import Image

    rgb = extract_rgb(h5_path)
    if rgb is None:
        return None

    img = Image.fromarray(rgb)
    if img.size != (THUMBNAIL_SIZE, THUMBNAIL_SIZE):
        img = img.resize((THUMBNAIL_SIZE, THUMBNAIL_SIZE), Image.LANCZOS)

    name = Path(h5_path).name
    out_path = os.path.join(output_dir, f"{name}.jpg")
    img.save(out_path, "JPEG", quality=quality)
    return out_path


def main() -> None:
    """Generate thumbnails for the requested H5 files."""
    p = argparse.ArgumentParser(
        description="Generate JPEG thumbnails from H5 sample files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--h5-dir", required=True, help="Directory containing sample_*.h5 files."
    )
    p.add_argument(
        "--output-dir", required=True, help="Where to write thumbnail JPEGs."
    )
    p.add_argument("--quality", type=int, default=85, help="JPEG quality (1-100).")
    p.add_argument("--workers", type=int, default=4, help="Number of parallel workers.")
    p.add_argument(
        "--metadata",
        default=None,
        help="Optional parquet file. If provided, only generates "
        "thumbnails for filenames present in the parquet.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most this many files (0 = all).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for shuffling. If omitted, uses a random seed "
        "(good for launching multiple machines independently).",
    )
    p.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling (process in sorted order).",
    )
    p.add_argument(
        "--inspect",
        action="store_true",
        help="Inspect the first H5 file and exit (useful for "
        "discovering the internal structure).",
    )
    args = p.parse_args()

    h5_dir = args.h5_dir
    h5_files = sorted(f for f in os.listdir(h5_dir) if f.endswith(".h5"))
    if not h5_files:
        print(f"[error] No .h5 files found in {h5_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[thumbgen] Found {len(h5_files):,} H5 files in {h5_dir}")

    if args.inspect:
        inspect_h5(os.path.join(h5_dir, h5_files[0]))
        return

    if args.metadata:
        import pandas as pd

        meta = pd.read_parquet(args.metadata, columns=["filename"])
        wanted = set(meta["filename"].dropna().astype(str))
        h5_files = [f for f in h5_files if Path(f).stem in wanted or f in wanted]
        print(f"[thumbgen] Filtered to {len(h5_files):,} files matching metadata")

    if not args.no_shuffle:
        random.seed(args.seed)
        random.shuffle(h5_files)
        seed_msg = f"seed={args.seed}" if args.seed is not None else "random seed"
        print(f"[thumbgen] Shuffled file list ({seed_msg})")

    if args.limit > 0:
        h5_files = h5_files[: args.limit]

    os.makedirs(args.output_dir, exist_ok=True)

    # Skip files whose thumbnails already exist on disk.
    before = len(h5_files)
    h5_files = [
        f
        for f in h5_files
        if not os.path.exists(os.path.join(args.output_dir, f"{f}.jpg"))
    ]
    already_done = before - len(h5_files)
    if already_done:
        print(f"[thumbgen] Skipping {already_done:,} already-generated thumbnails")
    print(f"[thumbgen] {len(h5_files):,} files to process")

    paths = [os.path.join(h5_dir, f) for f in h5_files]
    done = 0
    skipped = 0
    total_files = len(paths)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_one, p, args.output_dir, args.quality): p for p in paths
        }
        for fut in as_completed(futures):
            result = fut.result()
            if result:
                done += 1
            else:
                skipped += 1
            total = done + skipped
            if total % 1000 == 0 or total == total_files:
                print(
                    f"[thumbgen] {total:,}/{total_files:,} "
                    f"({done:,} ok, {skipped:,} skipped)"
                )

    print(f"[thumbgen] Done. {done:,} new thumbnails in {args.output_dir}")


if __name__ == "__main__":
    main()

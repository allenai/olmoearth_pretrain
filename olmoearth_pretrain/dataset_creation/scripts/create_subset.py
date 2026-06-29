"""Create a subset of an OlmoEarth-formatted dataset.

Selects a random subset of tiles at a given resolution and copies their files
across all modality directories, plus filters the summary CSVs to match.

Usage:
    python -m olmoearth_pretrain.dataset_creation.scripts.create_subset \
        --src_path /path/to/full_dataset \
        --dst_path /path/to/subset \
        --num_samples 100 \
        --modalities sentinel1 sentinel2_l2a worldcover srtm eurocrops cdl
"""

import argparse
import csv
import random
import shutil

from upath import UPath

from olmoearth_pretrain.data.constants import Modality, ModalitySpec, TimeSpan
from olmoearth_pretrain.dataset.parse import GridTile, ModalityTile, parse_modality_csv
from olmoearth_pretrain.dataset.utils import get_modality_dir


def discover_modality_csvs(
    src: UPath, modalities: list[ModalitySpec], resolution_factor: int
) -> list[tuple[ModalitySpec, TimeSpan, UPath]]:
    """Find which (modality, time_span) CSVs exist for the requested modalities.

    Args:
        src: the dataset root.
        modalities: the modalities to include.
        resolution_factor: the tile resolution factor to subset at.

    Returns:
        list of (modality, time_span, csv_path) tuples.

    Raises:
        ValueError: if a modality's tile_resolution_factor doesn't match.
    """
    available = []
    for modality in modalities:
        if modality.tile_resolution_factor != resolution_factor:
            raise ValueError(
                f"Modality {modality.name} has resolution_factor="
                f"{modality.tile_resolution_factor}, expected {resolution_factor}"
            )
        for ts in [TimeSpan.STATIC, TimeSpan.YEAR, TimeSpan.TWO_WEEK]:
            csv_path = (
                src
                / f"{modality.get_tile_resolution()}_{modality.name}{ts.get_suffix()}.csv"
            )
            if csv_path.exists():
                available.append((modality, ts, csv_path))
    return available


def create_subset(
    src: UPath,
    dst: UPath,
    num_samples: int,
    modalities: list[ModalitySpec],
    resolution_factor: int,
) -> None:
    """Create a subset dataset."""
    dst.mkdir(parents=True, exist_ok=True)

    # Discover available modality CSVs and parse them.
    # We use these to find all tiles in the dataset.
    print(f"Parsing modality CSVs (resolution_factor={resolution_factor})...")
    available = discover_modality_csvs(src, modalities, resolution_factor)
    if not available:
        print("No modality CSVs found for the requested modalities.")
        return

    parsed: dict[tuple[ModalitySpec, TimeSpan], list[ModalityTile]] = {}
    all_grid_tiles: set[GridTile] = set()

    for modality, ts, csv_path in available:
        tiles = parse_modality_csv(src, modality, ts, csv_path)
        parsed[(modality, ts)] = tiles
        for tile in tiles:
            all_grid_tiles.add(tile.grid_tile)

    print(
        f"Found {len(all_grid_tiles)} unique tiles "
        f"across {len(available)} modality CSVs"
    )

    if num_samples >= len(all_grid_tiles):
        print(
            f"Requested {num_samples} but only {len(all_grid_tiles)} exist, using all"
        )
        selected = all_grid_tiles
    else:
        selected = set(random.sample(list(all_grid_tiles), num_samples))
    print(f"Selected {len(selected)} tiles")

    # Copy files for selected tiles using paths from parsed ModalityTiles.
    for (modality, ts), tiles in parsed.items():
        modality_dir = get_modality_dir(dst, modality, ts)
        modality_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for tile in tiles:
            if tile.grid_tile not in selected:
                continue
            for src_fname in tile.band_sets.values():
                if not src_fname.exists():
                    continue
                dst_fname = dst / src_fname.relative_to(src)
                shutil.copy2(str(src_fname), str(dst_fname))
                copied += 1
        print(f"  {modality_dir.name}: copied {copied} files")

    # Filter and write subset CSVs.
    for modality, ts, csv_path in available:
        dst_csv = dst / csv_path.name
        with csv_path.open() as f_in, dst_csv.open("w") as f_out:
            reader = csv.DictReader(f_in)
            if reader.fieldnames is None:
                continue
            writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
            writer.writeheader()
            kept = 0
            total = 0
            for row in reader:
                total += 1

                # See if this row is in the selected tiles.
                gt = GridTile(
                    crs=row["crs"],
                    resolution_factor=resolution_factor,
                    col=int(row["col"]),
                    row=int(row["row"]),
                )
                if gt not in selected:
                    continue

                writer.writerow(row)
                kept += 1
        print(f"  {csv_path.name}: kept {kept}/{total} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a subset of an OlmoEarth-formatted dataset",
    )
    parser.add_argument(
        "--src_path", type=str, required=True, help="Source dataset path"
    )
    parser.add_argument(
        "--dst_path", type=str, required=True, help="Destination subset path"
    )
    parser.add_argument(
        "--num_samples", type=int, required=True, help="Number of tiles to select"
    )
    parser.add_argument(
        "--modalities",
        type=str,
        nargs="+",
        required=True,
        help="Modality names to include (e.g. sentinel1 sentinel2_l2a worldcover)",
    )
    parser.add_argument(
        "--resolution_factor",
        type=int,
        default=16,
        help="Tile resolution factor to subset (default: 16 = 10 m/pixel)",
    )
    args = parser.parse_args()

    modalities = [Modality.get(name) for name in args.modalities]

    create_subset(
        UPath(args.src_path),
        UPath(args.dst_path),
        args.num_samples,
        modalities,
        args.resolution_factor,
    )

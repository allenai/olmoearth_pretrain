"""Pre-tile OpenStreetMap PBF files into a 1-degree WGS84 grid of GeoJSON tiles.

One-time preprocessing step. Parses country/region PBF files and writes per-tile
GeoJSON using the same categories and format as rslearn's OpenStreetMap data source.
Output is shared across all dataset creation runs.

Usage:
    # Single PBF (for testing):
    python -m olmoearth_pretrain.dataset_creation.openstreetmap.pretile_osm \
        --pbf-dir /weka/.../source_data/openstreetmap \
        --bounds-json /weka/.../source_data/openstreetmap/pbf_bounds.json \
        --categories-json /weka/.../categories.json \
        --output-dir /weka/.../osm_pretiled \
        --only andorra-latest.osm.pbf

    # All PBFs:
    python -m olmoearth_pretrain.dataset_creation.openstreetmap.pretile_osm \
        --pbf-dir /weka/.../source_data/openstreetmap \
        --bounds-json /weka/.../source_data/openstreetmap/pbf_bounds.json \
        --categories-json /weka/.../categories.json \
        --output-dir /weka/.../osm_pretiled \
        --workers 8
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing
import os
import time
from pathlib import Path

import shapely
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.openstreetmap import Filter, OsmHandler
from rslearn.utils import Feature, STGeometry

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

TILE_SIZE_DEG = 1.0


def _parse_categories(raw: dict) -> dict[str, Filter]:
    """Parse category dict from JSON config into Filter objects."""
    categories = {}
    for name, spec in raw.items():
        from rslearn.data_sources.openstreetmap import FeatureType

        feature_types = None
        if "feature_types" in spec:
            feature_types = [FeatureType(ft.lower()) for ft in spec["feature_types"]]
        categories[name] = Filter(
            feature_types=feature_types,
            tag_conditions=spec.get("tag_conditions"),
            tag_properties=spec.get("tag_properties"),
            to_geometry=spec.get("to_geometry"),
        )
    return categories


def _tile_key(lon: int, lat: int) -> str:
    return f"lon_{lon}_lat_{lat}"


def _tile_path(output_dir: Path, lon: int, lat: int) -> Path:
    return output_dir / f"{_tile_key(lon, lat)}.geojson"


def _features_to_geojson(features: list[Feature]) -> dict:
    """Serialize rslearn Features to a GeoJSON FeatureCollection."""
    geojson_features = []
    for feat in features:
        shp = feat.geometry.to_wgs84().shp
        geojson_features.append(
            {
                "type": "Feature",
                "properties": feat.properties,
                "geometry": shapely.geometry.mapping(shp),
            }
        )
    return {"type": "FeatureCollection", "features": geojson_features}


def _grid_cells_for_bounds(
    bounds: tuple[float, float, float, float],
) -> list[tuple[int, int]]:
    """Return (lon_floor, lat_floor) for all 1-degree cells overlapping bounds."""
    min_lon, min_lat, max_lon, max_lat = bounds
    lon_start = int(math.floor(min_lon))
    lon_end = int(math.floor(max_lon))
    lat_start = int(math.floor(min_lat))
    lat_end = int(math.floor(max_lat))
    cells = []
    for lon in range(lon_start, lon_end + 1):
        for lat in range(lat_start, lat_end + 1):
            cells.append((lon, lat))
    return cells


def _assign_tiles_to_pbfs(
    pbf_names: list[str], pbf_bounds: list[tuple[float, float, float, float]]
) -> dict[str, list[tuple[int, int]]]:
    """Assign each 1-degree tile to the smallest PBF that covers it.

    This avoids redundant parsing of continent-level files for tiles that are
    already covered by a country-level file.
    """
    pbf_areas = []
    for b in pbf_bounds:
        area = (b[2] - b[0]) * (b[3] - b[1])
        pbf_areas.append(area)

    # Sort PBFs by area (smallest first) so smaller files "claim" tiles first
    order = sorted(range(len(pbf_names)), key=lambda i: pbf_areas[i])

    claimed: set[tuple[int, int]] = set()
    assignments: dict[str, list[tuple[int, int]]] = {}

    for idx in order:
        name = pbf_names[idx]
        bounds = pbf_bounds[idx]
        cells = _grid_cells_for_bounds(bounds)
        unclaimed = [c for c in cells if c not in claimed]
        if not unclaimed:
            logger.info(f"Skipping {name} -- all {len(cells)} tiles already claimed")
            continue
        assignments[name] = unclaimed
        claimed.update(unclaimed)
        logger.info(
            f"{name}: {len(unclaimed)} tiles assigned "
            f"({len(cells) - len(unclaimed)} already claimed)"
        )

    logger.info(f"Total unique tiles: {len(claimed)}")
    return assignments


def pretile_one_pbf(
    pbf_path: Path,
    tiles: list[tuple[int, int]],
    categories: dict[str, Filter],
    output_dir: Path,
) -> dict:
    """Parse one PBF file and write GeoJSON tiles for assigned grid cells.

    Returns a summary dict with counts.
    """
    t0 = time.time()
    pbf_name = pbf_path.name

    # Skip tiles that already exist on disk
    remaining = [(lon, lat) for lon, lat in tiles if not _tile_path(output_dir, lon, lat).exists()]
    if not remaining:
        logger.info(f"Skipping {pbf_name} -- all {len(tiles)} tiles already on disk")
        return {
            "pbf": pbf_name,
            "tiles_assigned": len(tiles),
            "tiles_written": 0,
            "tiles_empty": 0,
            "tiles_skipped": len(tiles),
            "features_total": 0,
            "elapsed_seconds": 0,
        }
    if len(remaining) < len(tiles):
        logger.info(f"{pbf_name}: {len(tiles) - len(remaining)} tiles already on disk, {len(remaining)} remaining")
    tiles = remaining

    logger.info(f"Processing {pbf_name} ({len(tiles)} tiles)...")

    # Build geometries for all assigned tiles
    geometries = []
    for lon, lat in tiles:
        box = shapely.box(lon, lat, lon + TILE_SIZE_DEG, lat + TILE_SIZE_DEG)
        geometries.append(STGeometry(WGS84_PROJECTION, box, None))

    handler = OsmHandler(categories, geometries)
    handler.apply_file(str(pbf_path))

    # Now bin features into tiles by checking which tile each feature falls in
    tile_features: dict[tuple[int, int], list[Feature]] = {t: [] for t in tiles}
    for feat in handler.features:
        wgs84_shp = feat.geometry.to_wgs84().shp
        centroid = wgs84_shp.centroid
        lon_idx = int(math.floor(centroid.x))
        lat_idx = int(math.floor(centroid.y))
        key = (lon_idx, lat_idx)
        if key in tile_features:
            tile_features[key].append(feat)

    written = 0
    empty = 0
    for (lon, lat), features in tile_features.items():
        if not features:
            empty += 1
            continue
        out_path = _tile_path(output_dir, lon, lat)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        geojson = _features_to_geojson(features)
        with open(out_path, "w") as f:
            json.dump(geojson, f)
        written += 1

    elapsed = time.time() - t0
    logger.info(
        f"Done {pbf_name}: {written} tiles written, "
        f"{empty} empty, {elapsed:.1f}s"
    )
    return {
        "pbf": pbf_name,
        "tiles_assigned": len(tiles),
        "tiles_written": written,
        "tiles_empty": empty,
        "features_total": len(handler.features),
        "elapsed_seconds": round(elapsed, 1),
    }


def _worker_wrapper(args: tuple) -> dict:
    """Wrapper for multiprocessing."""
    return pretile_one_pbf(*args)


def build_manifest(output_dir: Path) -> dict:
    """Scan the output directory and build a manifest of all tiles."""
    tiles = {}
    for f in sorted(output_dir.glob("lon_*_lat_*.geojson")):
        name = f.stem
        parts = name.split("_")
        lon = int(parts[1])
        lat = int(parts[3])
        tiles[name] = {
            "bounds": [lon, lat, lon + TILE_SIZE_DEG, lat + TILE_SIZE_DEG],
            "path": f.name,
        }
    return {"tile_size_deg": TILE_SIZE_DEG, "num_tiles": len(tiles), "tiles": tiles}


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-tile OSM PBF files")
    parser.add_argument(
        "--pbf-dir",
        required=True,
        help="Directory containing PBF files",
    )
    parser.add_argument(
        "--bounds-json",
        required=True,
        help="Path to pbf_bounds.json (or will be created)",
    )
    parser.add_argument(
        "--categories-json",
        required=True,
        help="Path to JSON file with OSM category filters",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for pre-tiled GeoJSON",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (each processes one PBF)",
    )
    parser.add_argument(
        "--rslearn-config",
        help="Path to rslearn config.json (to resolve pbf_fnames ordering for bounds)",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        help="Only process these PBF filenames (for testing)",
    )
    args = parser.parse_args()

    pbf_dir = Path(args.pbf_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.categories_json) as f:
        raw_categories = json.load(f)
    categories = _parse_categories(raw_categories)
    logger.info(f"Loaded {len(categories)} categories")

    with open(args.bounds_json) as f:
        all_bounds = json.load(f)

    # Build the pbf name -> bounds mapping. The bounds file is an ordered list
    # matching the pbf_fnames list in the rslearn config.
    pbf_files = sorted(pbf_dir.glob("*-latest.osm.pbf"))

    config_path = Path(args.rslearn_config) if args.rslearn_config else None
    if config_path is None or not config_path.exists():
        for candidate in [
            pbf_dir.parent.parent / "config.json",
            pbf_dir.parent.parent / "rslearn_corpus_30" / "config.json",
            Path("/weka/dfive-default/helios/dataset_creation/studio_corpus/rslearn_corpus_30/config.json"),
        ]:
            if candidate.exists():
                config_path = candidate
                break
    if config_path is None or not config_path.exists():
        raise FileNotFoundError("Cannot find rslearn config.json. Pass --rslearn-config.")

    with open(config_path) as f:
        cfg = json.load(f)
    config_fnames = cfg["layers"]["openstreetmap"]["data_source"]["init_args"]["pbf_fnames"]
    fname_to_bounds = {}
    for fname, b in zip(config_fnames, all_bounds):
        fname_to_bounds[os.path.basename(fname)] = tuple(b)

    # Filter to requested PBFs
    pbf_names = []
    pbf_bounds_list = []
    for pf in pbf_files:
        if pf.name not in fname_to_bounds:
            continue
        if args.only and pf.name not in args.only:
            continue
        pbf_names.append(pf.name)
        pbf_bounds_list.append(fname_to_bounds[pf.name])

    logger.info(f"Processing {len(pbf_names)} PBF files")

    assignments = _assign_tiles_to_pbfs(pbf_names, pbf_bounds_list)

    # Build work items
    work = []
    for name, tiles in assignments.items():
        work.append((pbf_dir / name, tiles, categories, output_dir))

    if not work:
        logger.info("No work to do")
        return

    if args.workers <= 1:
        results = [pretile_one_pbf(*w) for w in work]
    else:
        with multiprocessing.Pool(args.workers) as pool:
            results = pool.map(_worker_wrapper, work)

    # Build and write manifest
    manifest = build_manifest(output_dir)
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(
        f"Manifest written: {manifest['num_tiles']} tiles -> {manifest_path}"
    )

    # Print summary
    total_features = sum(r["features_total"] for r in results)
    total_written = sum(r["tiles_written"] for r in results)
    total_elapsed = sum(r["elapsed_seconds"] for r in results)
    logger.info(
        f"Summary: {total_written} tiles, {total_features} features, "
        f"{total_elapsed:.0f}s total compute"
    )


if __name__ == "__main__":
    main()

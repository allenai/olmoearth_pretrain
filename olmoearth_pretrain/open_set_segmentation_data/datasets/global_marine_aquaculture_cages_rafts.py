"""Process the reglab/aquaculture "marine aquaculture" dataset into detection label tiles.

Source: Ubina et al. / reglab, "Remote sensing and computer vision for marine
aquaculture", Science Advances 2024 (DOI 10.1126/sciadv.adn4944). Code + data at
github.com/reglab/aquaculture, archived on Zenodo record 10933921 (v1.0.0). We download
the Zenodo repo snapshot and use two georeferenced GeoJSON layers (both EPSG:3857):

  * ``output/humanlabels.geojson`` -- 4142 MANUAL cage bounding-box annotations
    (validation rounds) on French-Mediterranean aerial ortho-imagery, 2002-2021. Each is a
    small square/circle finfish net-pen cage footprint (~12 m median, 1-3 px @ 10 m), with
    a ``year`` and cage ``type`` (circle_cage / square_cage). This is the reference GT.
  * ``output/ocean_detections.geojson`` -- 17252 MODEL detections (YOLOv5) over the whole
    French-Med coast, 2000-2021, each with a ``det_conf``. Used as a high-confidence
    (det_conf >= 0.7) fallback map to expand spatial/temporal coverage beyond the small
    manual validation set.

IMPORTANT provenance/scope notes (judgment calls, see summary):
  * The manifest lists this as "Global ... cages & rafts" with classes {finfish cage,
    bivalve/algae raft}. The actual DOI-matched dataset is FRENCH-MEDITERRANEAN FINFISH
    CAGES ONLY -- there are NO rafts and it is not global. We therefore emit a single
    foreground class ``finfish_cage`` (no raft class can be fabricated).
  * Labels span 2000-2021; per the Sentinel-era rule we keep only year >= 2016.

Encoding (detection, spec section 4 bboxes -> detection): cages are sub-/near-resolution
objects marking aquaculture presence. We grid-snap features to 64x64 (640 m) UTM tiles
keyed by (year, utm_epsg, cell); within a cell-year, manual GT takes precedence over
detections. Per tile we rasterize the cage footprints as class 1 (all_touched, so tiny
cages keep >=1 px), ring each footprint with a 10 px nodata (255) buffer -- coordinates are
ortho-derived and may be a couple px off the Sentinel grid -- and leave the rest of the
tile as background 0. In-tile background provides spatially-meaningful negatives (finfish
farms never fill a 640 m tile); we do NOT fabricate separate all-ocean negative tiles since
"confirmed-empty ocean" cannot be reliably derived from this release. Time range = the
label's aerial-image year (1-year window); cages are persistent structures.

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_marine_aquaculture_cages_rafts
"""

import argparse
import json
import multiprocessing
import os
from collections import Counter
from typing import Any

import numpy as np
import scipy.ndimage as ndi
import shapely
import tqdm
from pyproj import Transformer
from rasterio.crs import CRS
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)

SLUG = "global_marine_aquaculture_cages_rafts"
NAME = "Global Marine Aquaculture (cages & rafts)"
ZENODO_RECORD = "10933921"
ZENODO_URL = (
    "https://zenodo.org/api/records/10933921/files/"
    "reglab/aquaculture-v1.0.0.zip/content"
)
REPO_DIR = "reglab-aquaculture-399a078"

TILE = io.MAX_TILE  # 64 px @ 10 m = 640 m
BUFFER = 10  # nodata ring (px) around each cage footprint (spec >= 10)
MIN_YEAR = 2016  # Sentinel era
DET_CONF_MIN = 0.7  # high-confidence filter for the model-detection fallback map
PER_CLASS = 1000  # cap on finfish_cage tiles (spec section 5)

BACKGROUND_ID = 0
CAGE_ID = 1
CLASS_NAMES = {BACKGROUND_ID: "background", CAGE_ID: "finfish_cage"}


def _utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def _load_features() -> list[dict[str, Any]]:
    """Read both GeoJSON layers, reproject footprints EPSG:3857 -> WGS84 lon/lat.

    Returns a list of feature dicts: {rings: [[(lon,lat),...]], clon, clat, year,
    is_human, source}. Only year >= MIN_YEAR (and det_conf >= DET_CONF_MIN for detections).
    """
    base = os.path.join(io.raw_dir(SLUG).path, REPO_DIR)
    to_wgs = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    out: list[dict[str, Any]] = []

    def add(path: str, is_human: bool) -> None:
        d = json.load(open(path))
        for f in d["features"]:
            p = f["properties"]
            year = p.get("year")
            if year is None or year < MIN_YEAR:
                continue
            if not is_human and p.get("det_conf", 0.0) < DET_CONF_MIN:
                continue
            rings_ll = []
            for ring in f["geometry"]["coordinates"]:
                xs = [c[0] for c in ring]
                ys = [c[1] for c in ring]
                lons, lats = to_wgs.transform(xs, ys)
                rings_ll.append(list(zip(lons, lats)))
            outer = rings_ll[0]
            clon = sum(c[0] for c in outer) / len(outer)
            clat = sum(c[1] for c in outer) / len(outer)
            out.append(
                {
                    "rings": rings_ll,
                    "clon": clon,
                    "clat": clat,
                    "year": int(year),
                    "is_human": is_human,
                    "source": "human" if is_human else "detection",
                }
            )

    add(os.path.join(base, "output", "humanlabels.geojson"), True)
    add(os.path.join(base, "output", "ocean_detections.geojson"), False)
    return out


def _write_tile(rec: dict[str, Any]) -> str | None:
    """Rasterize one tile's cage footprints with buffer, write tif + json. Idempotent."""
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return rec["present_key"]
    proj = Projection(CRS.from_epsg(rec["epsg"]), io.RESOLUTION, -io.RESOLUTION)
    bounds = tuple(rec["bounds"])

    shapes = []
    for rings in rec["rings"]:
        poly = shapely.Polygon(rings[0], rings[1:])
        px = geom_to_pixels(poly, WGS84_PROJECTION, proj)
        if not px.is_empty:
            shapes.append((px, CAGE_ID))
    pos = (
        rasterize_shapes(shapes, bounds, fill=0, dtype="uint8", all_touched=True)[0]
        == 1
        if shapes
        else np.zeros((TILE, TILE), dtype=bool)
    )
    buf = ndi.binary_dilation(pos, structure=np.ones((3, 3), bool), iterations=BUFFER)
    out = np.zeros((TILE, TILE), dtype=np.uint8)
    out[buf & ~pos] = io.CLASS_NODATA
    out[pos] = CAGE_ID

    present = sorted(int(v) for v in np.unique(out) if v != io.CLASS_NODATA)
    io.write_label_geotiff(SLUG, sample_id, out, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(rec["year"]),
        source_id=rec["source_id"],
        classes_present=present,
    )
    return "|".join(str(c) for c in present)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    # 1. Download + unzip the Zenodo repo snapshot (idempotent).
    from olmoearth_pretrain.open_set_segmentation_data import download

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    zip_path = raw / "aquaculture-v1.0.0.zip"
    download.download_http(ZENODO_URL, zip_path)
    if not (raw / REPO_DIR).exists():
        import zipfile

        with zipfile.ZipFile(zip_path.path) as z:
            z.extractall(raw.path)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "reglab/aquaculture (Science Advances 2024, DOI 10.1126/sciadv.adn4944).\n"
            f"Zenodo record {ZENODO_RECORD} v1.0.0 -> {REPO_DIR}/.\n"
            "output/humanlabels.geojson: 4142 manual finfish-cage bbox annotations "
            "(French Med, 2002-2021, EPSG:3857).\n"
            "output/ocean_detections.geojson: 17252 YOLOv5 detections w/ det_conf.\n"
            f"Used: year>={MIN_YEAR}; detections filtered det_conf>={DET_CONF_MIN}.\n"
        )

    # 2. Load + reproject features.
    feats = _load_features()
    n_h = sum(1 for f in feats if f["is_human"])
    print(
        f"features (year>={MIN_YEAR}): human={n_h} detection={len(feats) - n_h}",
        flush=True,
    )

    # 3. Grid-snap into (year, epsg, cellx, celly) tiles; human GT precedence per cell-year.
    tiles: dict[tuple, dict[str, Any]] = {}
    for f in feats:
        epsg = _utm_epsg(f["clon"], f["clat"])
        proj = Projection(CRS.from_epsg(epsg), io.RESOLUTION, -io.RESOLUTION)
        _, col, row = io.lonlat_to_utm_pixel(f["clon"], f["clat"], proj)
        cx, cy = col // TILE, row // TILE
        key = (f["year"], epsg, cx, cy)
        t = tiles.get(key)
        if t is None:
            t = tiles[key] = {
                "year": f["year"],
                "epsg": epsg,
                "bounds": [cx * TILE, cy * TILE, cx * TILE + TILE, cy * TILE + TILE],
                "human": [],
                "det": [],
            }
        (t["human"] if f["is_human"] else t["det"]).append(f["rings"])

    # Build tile records: manual GT if present in the cell-year, else detections.
    records: list[dict[str, Any]] = []
    for key, t in tiles.items():
        use_human = len(t["human"]) > 0
        rings = t["human"] if use_human else t["det"]
        records.append(
            {
                "year": t["year"],
                "epsg": t["epsg"],
                "bounds": t["bounds"],
                "rings": rings,
                "is_human": use_human,
                "n_obj": len(rings),
                "source_id": (
                    f"{'human' if use_human else f'detection>={DET_CONF_MIN}'}:"
                    f"{t['year']}:{t['epsg']}:{key[2]}:{key[3]}"
                ),
            }
        )

    n_human_tiles = sum(1 for r in records if r["is_human"])
    print(
        f"tiles: {len(records)} (human={n_human_tiles}, "
        f"detection={len(records) - n_human_tiles})",
        flush=True,
    )

    # 4. Cap at PER_CLASS finfish_cage tiles; prioritize manual GT tiles.
    records.sort(
        key=lambda r: (not r["is_human"], r["year"], r["epsg"], tuple(r["bounds"]))
    )
    if len(records) > PER_CLASS:
        print(f"capping {len(records)} -> {PER_CLASS} (human first)", flush=True)
        records = records[:PER_CLASS]
    for i, r in enumerate(records):
        r["sample_id"] = f"{i:06d}"
        r["present_key"] = ""

    # 5. Write tiles in parallel.
    io.check_disk()
    class_counts: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for present in tqdm.tqdm(
            star_imap_unordered(p, _write_tile, [dict(rec=r) for r in records]),
            total=len(records),
        ):
            for c in (present or "").split("|"):
                if c != "":
                    class_counts[int(c)] += 1

    src_counts = Counter("human" if r["is_human"] else "detection" for r in records)
    year_counts = dict(sorted(Counter(r["year"] for r in records).items()))
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "reglab/aquaculture (Science Advances 2024, DOI 10.1126/sciadv.adn4944)",
            "license": "MIT (code+labels repo) / CC-BY-4.0 (Zenodo archive)",
            "provenance": {
                "url": "https://doi.org/10.5281/zenodo.10933921",
                "have_locally": False,
                "annotation_method": (
                    "manual bounding-box annotation (validation rounds) + high-confidence "
                    "YOLOv5 detections as fallback map"
                ),
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {
                    "id": BACKGROUND_ID,
                    "name": "background",
                    "description": "Open water / non-cage surface within the detection tile.",
                },
                {
                    "id": CAGE_ID,
                    "name": "finfish_cage",
                    "description": (
                        "Marine finfish net-pen surface cage (circular or square), "
                        "French Mediterranean; footprint ~12 m median (1-3 px @ 10 m). "
                        "Rasterized footprint = positive, ringed by a 10 px nodata buffer."
                    ),
                },
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(records),
            "class_tile_counts": {
                CLASS_NAMES[c]: class_counts[c] for c in (BACKGROUND_ID, CAGE_ID)
            },
            "source_tile_counts": dict(src_counts),
            "year_tile_counts": year_counts,
            "tile_size": TILE,
            "detection_encoding": {"positive": "footprint", "buffer_px": BUFFER},
            "notes": (
                "reglab/aquaculture: French-Mediterranean finfish net-pen cages only "
                "(manifest's 'global' region and 'bivalve/algae raft' class are NOT present "
                "in the DOI-matched source; single finfish_cage class emitted). Labels "
                f"filtered to year>={MIN_YEAR} (Sentinel era); model detections filtered to "
                f"det_conf>={DET_CONF_MIN}. Detection encoding: cage footprints rasterized as "
                "class 1 (all_touched), 10 px nodata buffer ring, background elsewhere; "
                "64x64 UTM tiles grid-snapped per (year, zone, cell) with manual GT taking "
                "precedence over detections in a cell-year. In-tile background supplies "
                "negatives; no separate all-ocean negative tiles fabricated. Time range = "
                "aerial-image year (persistent structures)."
            ),
        },
    )
    print(f"class tile counts: {dict(class_counts)}")
    print(f"source: {dict(src_counts)}  years: {year_counts}")
    print(f"done: {len(records)} tiles")

    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(records)
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()

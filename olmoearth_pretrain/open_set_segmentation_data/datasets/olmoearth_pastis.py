"""Process PASTIS(-R) into open-set-segmentation crop-type label patches (dense_raster).

Source: PASTIS / PASTIS-R (Garnot & Landrieu, ICCV 2021; Zenodo 5735646 / 5012942),
staged locally at ``/weka/dfive-default/rslearn-eai/artifacts/PASTIS-R``. It is a
crop-type **semantic segmentation** benchmark over four Sentinel-2 tiles in France
(zones 30/31/32), 2433 patches of 128x128 px at 10 m. Labels come from the French RPG
(Registre Parcellaire Graphique) farmer declarations for the 2019 campaign, distributed
as per-patch ``ANNOTATIONS/TARGET_{id}.npy`` (channel 0 = semantic class). Per-patch
georeferencing lives in ``metadata.geojson`` (EPSG:2154 / Lambert-93 footprints, S2 tile
id, acquisition dates). Licensed open for research.

Task: per-pixel **classification** (crop type). We use ONLY the labels (pretraining
supplies its own S2/S1 imagery), so we do not touch the DATA_S2/S1 arrays.

Georeferencing: the local rslearn copy of PASTIS uses a *dummy* EPSG:3857 origin (its
convert.py notes it is "difficult to get the actual correct one"), so we instead recover
true geolocation from ``metadata.geojson``. Each patch footprint, transformed from
EPSG:2154 into its own Sentinel-2 UTM zone (from the ``TILE`` field), is an exact
1280x1280 m axis-aligned square, so the 128x128 native grid maps 1:1 onto the UTM 10 m
grid with no resampling. We split each patch into four 64x64 UTM tiles (quadrants) and
snap the origin to the nearest 10 m pixel (sub-pixel <~0.3 px offset from the 2154->UTM
transform, negligible for pretraining co-location).

Class scheme (native PASTIS semantic ids, kept as-is): 0 = background (non-declared /
non-crop land, a real observed negative class), 1..18 = the 18 crop types, 19 = void
(parcels touching patch borders / unresolved) -> mapped to nodata=255 (dropped). 19
usable classes (ids 0-18), uint8, well under the 254 cap.

Sampling: tiles-per-class balanced (spec 5) with per_class=1000 and the 25k cap; a tile
counts toward every class present in it and rare crops are prioritized.

Time range: PASTIS labels are the 2019 RPG crop campaign; imagery spans Sep 2018-Nov
2019 (> 1 yr). We assign a fixed 360-day growing-season window [2019-01-01, 2019-12-27)
per tile (<= 1 year, post-2016), change_time=null (this is state classification, not a
dated change event).

Run (idempotent; skips already-written {sample_id}.tif):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_pastis
"""

import argparse
import json
import multiprocessing
from collections import Counter
from datetime import UTC, datetime
from typing import Any

import numpy as np
import shapely
import tqdm
from pyproj import Transformer
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered
from shapely.ops import transform as shp_transform

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    select_tiles_per_class,
)

SLUG = "olmoearth_pastis"
NAME = "OlmoEarth PASTIS"
SRC = "/weka/dfive-default/rslearn-eai/artifacts/PASTIS-R"

VOID_ID = 19  # PASTIS "void" label -> nodata (255)
TILE_SIZE = 64  # split each 128x128 patch into four 64x64 quadrants
PER_CLASS = 1000

# Fixed 360-day growing-season window for the 2019 RPG campaign (<= 1 year, post-2016).
GROWING_SEASON = (
    datetime(2019, 1, 1, tzinfo=UTC),
    datetime(2019, 12, 27, tzinfo=UTC),
)

# Native PASTIS semantic classes: (name, description). Ids kept as-is (0..18).
CLASSES: list[tuple[str, str]] = [
    (
        "background",
        "Non-declared / non-agricultural land (roads, built-up, water, forest, "
        "unclassified) — the observed negative class outside declared crop parcels.",
    ),
    ("meadow", "Permanent or temporary grassland / meadow used for grazing or fodder."),
    (
        "soft winter wheat",
        "Soft (common) winter wheat (Triticum aestivum), autumn-sown.",
    ),
    ("corn", "Maize / corn (Zea mays), spring-sown summer cereal."),
    ("winter barley", "Autumn-sown winter barley (Hordeum vulgare)."),
    (
        "winter rapeseed",
        "Winter oilseed rape / canola (Brassica napus), yellow spring bloom.",
    ),
    ("spring barley", "Spring-sown barley (Hordeum vulgare)."),
    ("sunflower", "Sunflower (Helianthus annuus), summer oilseed crop."),
    ("grapevine", "Vineyards / grapevine (Vitis vinifera), perennial row crop."),
    ("beet", "Sugar / fodder beet (Beta vulgaris)."),
    (
        "winter triticale",
        "Winter triticale (x Triticosecale), wheat-rye hybrid cereal.",
    ),
    ("winter durum wheat", "Winter durum / hard wheat (Triticum durum)."),
    (
        "fruits/vegetables/flowers",
        "Mixed horticulture: orchards' understory, market-garden "
        "vegetables, flowers and small fruit plots.",
    ),
    ("potatoes", "Potato (Solanum tuberosum) fields."),
    (
        "leguminous fodder",
        "Leguminous fodder crops (alfalfa/lucerne, clover, sainfoin).",
    ),
    ("soybeans", "Soybean (Glycine max), summer legume."),
    ("orchard", "Fruit orchards (apple, stone fruit, nuts) — perennial tree crops."),
    ("mixed cereal", "Mixed / associated cereals grown together."),
    ("sorghum", "Sorghum (Sorghum bicolor), summer cereal."),
]

_WGS84 = None
_TRANSFORMERS: dict[int, Transformer] = {}


def _utm_epsg_for_tile(tile: str) -> int:
    """S2 tile id like 't30uxv' -> UTM EPSG (northern hemisphere; France)."""
    zone = int(tile[1:3])
    return 32600 + zone


def _transformer(epsg: int) -> Transformer:
    tr = _TRANSFORMERS.get(epsg)
    if tr is None:
        tr = Transformer.from_crs(2154, epsg, always_xy=True)
        _TRANSFORMERS[epsg] = tr
    return tr


def _patch_origin(feature: dict[str, Any], epsg: int) -> tuple[int, int]:
    """Return (col_min, row_min) integer pixel origin (top-left) of the 128x128 patch
    in the tile's UTM projection at 10 m (north-up, y_res=-10).
    """
    tr = _transformer(epsg)
    geom = shapely.geometry.shape(feature["geometry"])
    geom_utm = shp_transform(lambda x, y, z=None: tr.transform(x, y), geom)
    minx, _miny, _maxx, maxy = geom_utm.bounds
    col_min = int(round(minx / io.RESOLUTION))
    row_min = int(round(-maxy / io.RESOLUTION))  # world_y = row * (-10) => row = -y/10
    return col_min, row_min


def scan_patch(feature: dict[str, Any]) -> list[dict[str, Any]]:
    """Load one patch's semantic label, split into quadrant tile records."""
    example_id = feature["id"]
    tile = feature["properties"]["TILE"]
    fold = feature["properties"]["Fold"]
    epsg = _utm_epsg_for_tile(tile)
    target = np.load(f"{SRC}/ANNOTATIONS/TARGET_{example_id}.npy")
    label = target[0].astype(np.uint8)
    label[label == VOID_ID] = io.CLASS_NODATA  # drop void -> nodata

    col_min, row_min = _patch_origin(feature, epsg)
    h, w = label.shape  # 128, 128
    out: list[dict[str, Any]] = []
    for qr in range(0, h, TILE_SIZE):
        for qc in range(0, w, TILE_SIZE):
            sub = np.ascontiguousarray(label[qr : qr + TILE_SIZE, qc : qc + TILE_SIZE])
            present = sorted(set(np.unique(sub).tolist()) - {io.CLASS_NODATA})
            if not present:  # all-nodata quadrant: nothing to learn
                continue
            bounds = (
                col_min + qc,
                row_min + qr,
                col_min + qc + sub.shape[1],
                row_min + qr + sub.shape[0],
            )
            out.append(
                {
                    "epsg": epsg,
                    "bounds": bounds,
                    "array": sub,
                    "classes_present": present,
                    "source_id": f"{tile}/{example_id}/q{qr // TILE_SIZE}{qc // TILE_SIZE}",
                    "fold": fold,
                }
            )
    return out


def write_tile(rec: dict[str, Any]) -> tuple[str, str]:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return sample_id, "skip"
    try:
        proj = Projection(CRS.from_epsg(rec["epsg"]), io.RESOLUTION, -io.RESOLUTION)
        io.write_label_geotiff(
            SLUG, sample_id, rec["array"], proj, rec["bounds"], nodata=io.CLASS_NODATA
        )
        io.write_sample_json(
            SLUG,
            sample_id,
            proj,
            rec["bounds"],
            GROWING_SEASON,
            change_time=None,
            source_id=rec["source_id"],
            classes_present=rec["classes_present"],
        )
        return sample_id, "ok"
    except Exception as e:  # noqa: BLE001
        print(f"error on {sample_id}: {e}")
        return sample_id, "error"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    # Record source pointer (labels are staged locally; do not copy).
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "PASTIS-R (Garnot & Landrieu 2021), staged locally.\n"
            f"Source labels: {SRC}/ANNOTATIONS/TARGET_*.npy (channel 0 = semantic).\n"
            f"Georeferencing: {SRC}/metadata.geojson (EPSG:2154 footprints, TILE, dates).\n"
            "Zenodo: https://zenodo.org/records/5735646 (PASTIS-R) / 5012942 (PASTIS).\n"
            "Only labels used; pretraining supplies imagery.\n"
        )

    with open(f"{SRC}/metadata.geojson") as f:
        features = json.load(f)["features"]
    print(f"{len(features)} PASTIS patches")

    # ---- Scan phase: patch -> quadrant tile records (parallel) --------------------
    records: list[dict[str, Any]] = []
    with multiprocessing.Pool(args.workers) as p:
        for recs in tqdm.tqdm(
            star_imap_unordered(p, scan_patch, [dict(feature=ft) for ft in features]),
            total=len(features),
            desc="scan",
        ):
            records.extend(recs)
    print(f"{len(records)} candidate 64x64 tiles")
    io.check_disk()

    # ---- Tiles-per-class balanced selection (rare crops prioritized, 25k cap) ------
    selected = select_tiles_per_class(
        records, classes_key="classes_present", per_class=PER_CLASS, total_cap=25000
    )
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(f"selected {len(selected)} tiles")

    # ---- Write phase (parallel) ---------------------------------------------------
    results: Counter = Counter()
    written_by_class: Counter = Counter()
    id_to_rec = {r["sample_id"]: r for r in selected}
    with multiprocessing.Pool(args.workers) as p:
        for sample_id, res in tqdm.tqdm(
            star_imap_unordered(p, write_tile, [dict(rec=r) for r in selected]),
            total=len(selected),
            desc="write",
        ):
            results[res] += 1
            if res in ("ok", "skip"):
                for c in id_to_rec[sample_id]["classes_present"]:
                    written_by_class[c] += 1
    print("write results:", dict(results))
    io.check_disk()

    num_written = int(results.get("ok", 0) + results.get("skip", 0))

    # ---- Metadata -----------------------------------------------------------------
    classes = [
        {"id": i, "name": name, "description": desc}
        for i, (name, desc) in enumerate(CLASSES)
    ]
    class_counts = {
        CLASSES[i][0]: int(written_by_class.get(i, 0)) for i in range(len(CLASSES))
    }
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "olmoearth (PASTIS-R, staged locally)",
            "license": "open (research)",
            "provenance": {
                "url": "https://zenodo.org/records/5735646",
                "have_locally": True,
                "local_path": SRC,
                "annotation_method": "farmer declaration (French RPG 2019 campaign)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": classes,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": num_written,
            "class_counts": class_counts,
            "notes": (
                "PASTIS crop-type semantic segmentation over 4 Sentinel-2 tiles in France "
                "(UTM zones 30/31/32). Labels = French RPG 2019 farmer declarations "
                "(ANNOTATIONS/TARGET_*.npy channel 0). Native class ids kept: 0=background "
                "(real non-crop negative), 1-18 = crop types; PASTIS void (19) -> nodata "
                "255. Each 128x128 @10 m patch split into four 64x64 UTM 10 m tiles; "
                "all-nodata quadrants dropped. Georeferencing recovered from metadata.geojson "
                "(EPSG:2154 -> per-tile UTM; patches are exact 1280 m axis-aligned squares, "
                "no resampling). Tiles-per-class balanced (per_class=1000, 25k cap). "
                "Time range = fixed 360-day 2019 growing-season window; change_time=null."
            ),
        },
    )

    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=num_written
    )
    print(f"done: {num_written} samples")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()

"""Process OlmoEarth Maldives ecosystem mapping into open-set-segmentation patches.

Source: local rslearn project ``maldives_ecosystem_mapping/dataset_v1/20240924``. The
``crops`` group holds 91 manually-annotated coastal/marine ecosystem-type crops
(IUCN Global Ecosystem Typology classes) rasterized over Maxar VHR imagery
(~0.35-0.49 m/pixel) in local UTM (EPSG:32643, zone 43N). Each crop covers a small
(~1 km) patch with a ~2-minute Maxar acquisition time_range.

VHR handling (per the task spec): the VHR categorical label is resampled to 10 m
(NEAREST -- never bilinear) and each crop is tiled into <=64x64 patches. The label
value 0 ("unknown"/unannotated) is treated as nodata (255); the 16 real IUCN GET
classes (source ids 1..16) are remapped to output ids 0..15. Time range is a 1-year
window centered on the Maxar image date.

Suitability note: all 16 classes survive 10 m nearest resampling. Broad shallow-water
benthic classes (seagrass, coral reef, subtidal sand/rocky reef) are well mappable from
Sentinel-2 at 10 m in the clear Maldivian atolls. The narrow linear shoreline classes
(rocky / sandy / artificial shorelines) and the rare saltmarsh/rocky-reef classes are
marginal at 10 m -- they are kept but flagged low-confidence; see the summary.

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_maldives_ecosystem_mapping
"""

import argparse
import json
import multiprocessing
import os
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import tqdm
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import GeotiffRasterFormat
from upath import UPath

from olmoearth_pretrain.open_set_segmentation_data import io

SLUG = "olmoearth_maldives_ecosystem_mapping"
NAME = "OlmoEarth Maldives ecosystem mapping"
SOURCE = "/weka/dfive-default/rslearn-eai/datasets/maldives_ecosystem_mapping/dataset_v1/20240924"
CROPS_ROOT = os.path.join(SOURCE, "windows", "crops")
TARGET_RES = 10.0
TILE = io.MAX_TILE  # 64
SOURCE_UNKNOWN = 0  # source label value for unannotated / "unknown" -> nodata

# Output classes (output id -> IUCN GET code, name, description). These are the source
# CATEGORIES[1..16] (index 0 "unknown" is dropped to nodata), remapped id = source-1.
CLASSES = [
    (
        "FM1.3",
        "Intermittently closed and open lakes and lagoons",
        "IUCN GET FM1.3. Coastal water bodies periodically connected to the sea; brackish lagoons whose inlets open and close.",
    ),
    (
        "F2.2",
        "Small permanent freshwater lakes",
        "IUCN GET F2.2. Standing permanent freshwater bodies on the islands.",
    ),
    (
        "MFT1.2",
        "Intertidal forests and shrublands (mangroves)",
        "IUCN GET MFT1.2. Mangrove forests/shrublands in the intertidal zone.",
    ),
    (
        "MFT1.3",
        "Coastal saltmarshes and reedbeds",
        "IUCN GET MFT1.3. Herbaceous salt-tolerant marsh/reed vegetation of sheltered coasts.",
    ),
    (
        "MT1.1",
        "Rocky shorelines",
        "IUCN GET MT1.1. Wave-exposed hard rocky intertidal shores. Narrow linear feature.",
    ),
    (
        "MT1.3",
        "Sandy shorelines",
        "IUCN GET MT1.3. Sandy beaches of the intertidal zone. Narrow linear feature.",
    ),
    (
        "MT2.1",
        "Coastal shrublands and grasslands",
        "IUCN GET MT2.1. Supralittoral coastal vegetation of shrubs and grasses above the shoreline.",
    ),
    (
        "MT3.1",
        "Artificial shorelines",
        "IUCN GET MT3.1. Human-made shoreline structures (seawalls, harbours, reclaimed edges). Narrow linear feature.",
    ),
    (
        "M1.1",
        "Seagrass meadows",
        "IUCN GET M1.1. Subtidal/intertidal seagrass beds in shallow soft sediment.",
    ),
    (
        "M1.3",
        "Photic coral reefs",
        "IUCN GET M1.3. Shallow, light-dependent coral reef ecosystems.",
    ),
    (
        "M1.6",
        "Subtidal rocky reefs",
        "IUCN GET M1.6. Submerged hard-substrate reefs below the low-tide mark.",
    ),
    (
        "M1.7",
        "Subtidal sand beds",
        "IUCN GET M1.7. Submerged unconsolidated sand/soft-sediment beds.",
    ),
    (
        "TF1.3",
        "Permanent marshes",
        "IUCN GET TF1.3. Permanently waterlogged palustrine herbaceous wetlands.",
    ),
    ("T7.1", "Annual croplands", "IUCN GET T7.1. Cultivated land under annual crops."),
    (
        "T7.3",
        "Plantations",
        "IUCN GET T7.3. Woody plantation agriculture (e.g. coconut/other tree crops).",
    ),
    (
        "T7.4",
        "Urban and industrial ecosystems",
        "IUCN GET T7.4. Built-up urban, settlement, and industrial land.",
    ),
]
NUM_CLASSES = len(CLASSES)  # 16


def _remap(arr: np.ndarray) -> np.ndarray:
    """Map source category ids to output ids; unknown/unmapped -> nodata (255)."""
    out = np.full(arr.shape, io.CLASS_NODATA, dtype=np.uint8)
    # source 1..16 -> 0..15
    for src in range(1, NUM_CLASSES + 1):
        out[arr == src] = src - 1
    return out


def _centered_year(ts: datetime) -> tuple[datetime, datetime]:
    """1-year window centered on ts (<=360 days: +/-180 days)."""
    return (ts - timedelta(days=180), ts + timedelta(days=180))


def _tiles_for_crop(crop_name: str) -> list[dict[str, Any]]:
    """Decode one crop's VHR label at 10 m (nearest) and split into <=64x64 tiles.

    Returns non-empty tile records (those with >=1 labeled pixel), each carrying the
    remapped uint8 array so the writer need not re-decode.
    """
    wdir = os.path.join(CROPS_ROOT, crop_name)
    try:
        with open(os.path.join(wdir, "metadata.json")) as f:
            md = json.load(f)
    except FileNotFoundError:
        return []
    b = md["bounds"]
    crs = CRS.from_string(md["projection"]["crs"])
    src_res = md["projection"]["x_resolution"]
    tr = md.get("time_range")
    # Image acquisition midpoint.
    if tr:
        t0 = datetime.fromisoformat(tr[0])
        t1 = datetime.fromisoformat(tr[1])
        ts = t0 + (t1 - t0) / 2
    else:
        ts = datetime.fromisoformat("2024-08-01T00:00:00+00:00")

    proj = Projection(crs, TARGET_RES, -TARGET_RES)
    f = src_res / TARGET_RES
    # Full crop footprint in 10 m pixel coords (same CRS, reused since already UTM).
    tx0 = int(np.floor(b[0] * f))
    ty0 = int(np.floor(b[1] * f))
    tx1 = int(np.ceil(b[2] * f))
    ty1 = int(np.ceil(b[3] * f))

    raster_dir = UPath(wdir) / "layers" / "label" / "label"
    fmt = GeotiffRasterFormat()

    records: list[dict[str, Any]] = []
    for r0 in range(ty0, ty1, TILE):
        th = min(TILE, ty1 - r0)
        for c0 in range(tx0, tx1, TILE):
            tw = min(TILE, tx1 - c0)
            bounds = (c0, r0, c0 + tw, r0 + th)
            ra = fmt.decode_raster(
                raster_dir, proj, bounds, resampling=Resampling.nearest
            )
            src_arr = np.asarray(ra.array).reshape(th, tw).astype(np.uint8)
            out = _remap(src_arr)
            present = sorted(int(v) for v in np.unique(out) if v != io.CLASS_NODATA)
            if not present:
                continue  # no labeled pixels -> no signal
            records.append(
                {
                    "crop": crop_name,
                    "r0": r0,
                    "c0": c0,
                    "bounds": bounds,
                    "crs": crs.to_string(),
                    "res": TARGET_RES,
                    "ts": ts.isoformat(),
                    "classes_present": present,
                    "array": out,
                }
            )
    return records


def _write_one(rec: dict[str, Any]) -> None:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return
    proj = Projection(CRS.from_string(rec["crs"]), rec["res"], -rec["res"])
    bounds = tuple(rec["bounds"])
    io.write_label_geotiff(
        SLUG, sample_id, rec["array"], proj, bounds, nodata=io.CLASS_NODATA
    )
    ts = datetime.fromisoformat(rec["ts"])
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        _centered_year(ts),
        source_id=rec["crop"],
        classes_present=rec["classes_present"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "local rslearn dataset (have_locally=true, not copied):\n"
            f"{SOURCE}\n"
            "group 'crops' = manually annotated Maldives ecosystem crops over Maxar VHR.\n"
            "Legend: rslp.maldives_ecosystem_mapping.config.CATEGORIES "
            "(0=unknown->nodata, 1..16 = IUCN GET classes).\n"
        )

    crop_names = sorted(os.listdir(CROPS_ROOT))
    print(f"scanning {len(crop_names)} crops (decode @10m nearest + tile)")
    with multiprocessing.Pool(args.workers) as p:
        all_records: list[dict[str, Any]] = []
        for recs in tqdm.tqdm(
            star_imap_unordered(
                p, _tiles_for_crop, [dict(crop_name=c) for c in crop_names]
            ),
            total=len(crop_names),
        ):
            all_records.extend(recs)

    # Deterministic ordering for stable/idempotent sample ids.
    all_records.sort(key=lambda r: (r["crop"], r["r0"], r["c0"]))
    for i, r in enumerate(all_records):
        r["sample_id"] = f"{i:06d}"
    print(f"non-empty tiles: {len(all_records)}")

    with multiprocessing.Pool(args.workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in all_records]),
            total=len(all_records),
        ):
            pass

    # Per-class tile counts (a tile counts toward every class present in it).
    tile_counts = {i: 0 for i in range(NUM_CLASSES)}
    for r in all_records:
        for cid in r["classes_present"]:
            tile_counts[cid] += 1

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "olmoearth",
            "license": "internal",
            "provenance": {
                "url": SOURCE,
                "have_locally": True,
                "annotation_method": "manual annotation (Kili) over Maxar VHR imagery",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {
                    "id": i,
                    "name": name,
                    "description": f"{desc}",
                    "iucn_get_code": code,
                }
                for i, (code, name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(all_records),
            "class_tile_counts": {
                CLASSES[i][1]: tile_counts[i] for i in range(NUM_CLASSES)
            },
            "notes": (
                "VHR (~0.35-0.49 m) IUCN GET ecosystem labels resampled to 10 m with "
                "NEAREST and tiled into <=64x64 patches; source label 0 (unknown/"
                "unannotated) -> nodata 255; source ids 1..16 -> output ids 0..15. "
                "Time range: 1-year window centered on the Maxar image date "
                "(dates span 2023-2024). All source splits used. "
                "Low-confidence at 10 m (narrow/rare): Rocky shorelines, Sandy "
                "shorelines, Artificial shorelines, Coastal saltmarshes and reedbeds, "
                "Subtidal rocky reefs -- retained but noisy after resampling."
            ),
        },
    )
    print("class tile counts:")
    for i in range(NUM_CLASSES):
        print(f"  {i:>2} {CLASSES[i][1]:42} {tile_counts[i]}")
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()

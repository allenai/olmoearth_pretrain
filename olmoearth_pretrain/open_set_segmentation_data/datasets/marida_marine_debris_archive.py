"""MARIDA (Marine Debris Archive) -> open-set-segmentation classification.

MARIDA (Kikaki et al., PLOS ONE 2022; Zenodo 5151941) is a benchmark of manually
photo-interpreted Sentinel-2 pixel annotations distinguishing marine debris from
co-occurring sea-surface features. The release ships 1381 patches, each a 256x256
Sentinel-2 crop already georeferenced in local UTM at 10 m/pixel, with a companion
``*_cl.tif`` class raster (float32; value 0 = unlabeled, 1-15 = the 15 MARIDA classes)
and a ``*_conf.tif`` confidence raster. Annotations are sparse within each patch (only
photo-interpreted pixels carry a class; the rest is 0 = unlabeled).

label_type is dense_raster: we crop each 256x256 class raster into non-overlapping
64x64 UTM 10 m tiles, keep every tile containing >=1 labeled pixel, remap MARIDA ids
1-15 -> class ids 0-14, and set unlabeled pixels (0) to nodata (255). Sampling is
tiles-per-class balanced (each tile counts toward every class present in it). The full
candidate set is only 3322 tiles (well under the 25k cap and the 1000/class target for
all but Marine Water), so we keep all of them to maximize coverage of the rare debris
classes; this is the tiles-per-class-balanced outcome with no truncation needed.

Each patch is a single Sentinel-2 acquisition whose date is encoded in the scene name
(``S2_dd-mm-yy_TILE``). Sea-surface features (debris, sargassum, foam, wakes, ships) are
transient, so the label describes that one image: time_range is the 1-day window of the
acquisition date (well under 1 year; captures the specific S2 scene).

Run:
    python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.marida_marine_debris_archive
"""

import glob
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from .. import io, manifest
from ..download import download_zenodo

SLUG = "marida_marine_debris_archive"
NAME = "MARIDA (Marine Debris Archive)"
ZENODO_RECORD = "5151941"
TILE = 64
RESOLUTION = 10

# MARIDA class ids 1..15 -> output ids 0..14 (id_out = marida_id - 1). 0 = unlabeled -> 255.
MARIDA_CLASSES = [
    (
        "Marine Debris",
        "Floating marine litter / anthropogenic debris aggregations (plastics, mixed litter windrows) identified by photo-interpretation on Sentinel-2.",
    ),
    ("Dense Sargassum", "Dense floating Sargassum macroalgae rafts/mats."),
    ("Sparse Sargassum", "Sparse / diffuse floating Sargassum patches."),
    (
        "Natural Organic Material",
        "Natural floating organic matter (e.g. wood, vegetation, other biogenic material).",
    ),
    ("Ship", "Vessels on the water surface."),
    ("Clouds", "Cloud cover."),
    ("Marine Water", "Open marine water (clear background sea surface)."),
    (
        "Sediment-Laden Water",
        "Water with high suspended-sediment load (e.g. river plumes).",
    ),
    ("Foam", "Surface foam / whitewater aggregations."),
    (
        "Turbid Water",
        "Turbid water (elevated turbidity, not primarily sediment plume).",
    ),
    (
        "Shallow Water",
        "Optically shallow water where the seabed influences reflectance.",
    ),
    ("Waves", "Breaking waves / wave crests."),
    ("Cloud Shadows", "Shadows cast by clouds on the water surface."),
    ("Wakes", "Ship wakes / turbulent surface trails."),
    ("Mixed Water", "Mixed-water pixels (mixture of water types / transition zones)."),
]

SUMMARY_PATH = Path(
    "data/open_set_segmentation_data/"
    f"dataset_summaries/{SLUG}.md"
)


def _parse_date(scene: str) -> datetime:
    """Parse S2 acquisition date from a scene folder name 'S2_dd-mm-yy_TILE'."""
    parts = scene.split("_")
    d, m, y = parts[1].split("-")
    year = 2000 + int(y)
    return datetime(year, int(m), int(d), tzinfo=UTC)


def _scan_one(cl_path: str):
    """Return candidate tile records for one class raster."""
    with rasterio.open(cl_path) as ds:
        a = ds.read(1)
        h, w = a.shape
        crs = ds.crs.to_string()
        tr = ds.transform
    scene = os.path.basename(os.path.dirname(cl_path))
    date = _parse_date(scene)
    base = os.path.basename(cl_path)[: -len("_cl.tif")]  # e.g. S2_11-6-18_16PCC_0
    recs = []
    for r0 in range(0, h, TILE):
        for c0 in range(0, w, TILE):
            crop = a[r0 : r0 + TILE, c0 : c0 + TILE]
            if crop.shape != (TILE, TILE):
                continue
            labeled = crop[crop > 0]
            if labeled.size == 0:
                continue
            classes = sorted({int(x) - 1 for x in np.unique(labeled)})
            recs.append(
                {
                    "cl_path": cl_path,
                    "r0": r0,
                    "c0": c0,
                    "crs": crs,
                    "origin_x": tr.c,
                    "origin_y": tr.f,
                    "date": date.isoformat(),
                    "classes_present": classes,
                    "source_id": f"{base}_r{r0}_c{c0}",
                }
            )
    return recs


def _write_one(sample_id: str, rec: dict) -> tuple[str, list[int]]:
    """Read the 64x64 crop, remap, and write the label GeoTIFF + sidecar JSON."""
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return sample_id, rec["classes_present"]

    r0, c0 = rec["r0"], rec["c0"]
    with rasterio.open(rec["cl_path"]) as ds:
        crop = ds.read(1, window=((r0, r0 + TILE), (c0, c0 + TILE)))

    out = np.full((TILE, TILE), io.CLASS_NODATA, dtype=np.uint8)
    m = crop > 0
    out[m] = (crop[m].astype(np.int32) - 1).astype(np.uint8)

    crs = CRS.from_string(rec["crs"])
    proj = Projection(crs, RESOLUTION, -RESOLUTION)
    x_ul = rec["origin_x"] + c0 * RESOLUTION
    y_ul = rec["origin_y"] - r0 * RESOLUTION
    x_min = int(round(x_ul / RESOLUTION))
    y_min = int(round(-y_ul / RESOLUTION))
    bounds = (x_min, y_min, x_min + TILE, y_min + TILE)

    date = datetime.fromisoformat(rec["date"])
    time_range = (date, date + timedelta(days=1))

    io.write_label_geotiff(SLUG, sample_id, out, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        time_range,
        source_id=rec["source_id"],
        classes_present=rec["classes_present"],
    )
    return sample_id, rec["classes_present"]


def main() -> None:
    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    extracted = raw / "extracted"
    if not extracted.exists():
        download_zenodo(ZENODO_RECORD, raw)
        import zipfile

        with zipfile.ZipFile((raw / "MARIDA.zip").path) as z:
            z.extractall(extracted.path)

    cl_files = sorted(glob.glob(str(extracted / "patches" / "*" / "*_cl.tif")))
    print(f"scanning {len(cl_files)} class rasters ...")
    records: list[dict] = []
    with __import__("multiprocessing").Pool(64) as pool:
        for recs in pool.imap_unordered(_scan_one, cl_files):
            records.extend(recs)
    print(f"candidate 64x64 tiles with >=1 labeled pixel: {len(records)}")

    # Tiles-per-class balanced: full candidate set is far under the 25k cap, so keep all
    # tiles (dropping any would remove co-present rare debris classes). Stable ordering.
    records.sort(key=lambda r: r["source_id"])
    assert len(records) <= 25000, "exceeds 25k cap"

    from collections import Counter

    tiles_per_class: Counter = Counter()
    for r in records:
        for c in r["classes_present"]:
            tiles_per_class[c] += 1

    tasks = [{"sample_id": f"{i:06d}", "rec": rec} for i, rec in enumerate(records)]
    print(f"writing {len(tasks)} label patches ...")
    written = 0
    with __import__("multiprocessing").Pool(64) as pool:
        for _sid, _cls in star_imap_unordered(pool, _write_one, tasks):
            written += 1
            if written % 1000 == 0:
                print(f"  {written}/{len(tasks)}")
                io.check_disk()
    print(f"wrote {written} patches")

    metadata = {
        "dataset": SLUG,
        "name": NAME,
        "task_type": "classification",
        "source": "Zenodo / PLOS One",
        "license": "CC-BY-4.0",
        "provenance": {
            "url": "https://doi.org/10.5281/zenodo.5151941",
            "have_locally": False,
            "annotation_method": "manual photointerpretation of Sentinel-2 imagery",
        },
        "sensors_relevant": ["sentinel2"],
        "classes": [
            {"id": i, "name": n, "description": d}
            for i, (n, d) in enumerate(MARIDA_CLASSES)
        ],
        "nodata_value": io.CLASS_NODATA,
        "num_samples": len(records),
        "notes": (
            "256x256 Sentinel-2 class rasters (native UTM 10 m) cropped into "
            "non-overlapping 64x64 tiles; all tiles with >=1 labeled pixel kept. "
            "MARIDA ids 1-15 remapped to 0-14; unlabeled (0) -> 255 nodata. "
            "Annotations are sparse within each tile. time_range = 1-day window of the "
            "Sentinel-2 acquisition date (transient sea-surface features)."
        ),
    }
    io.write_dataset_metadata(SLUG, metadata)

    counts_by_id = {
        i: int(tiles_per_class.get(i, 0)) for i in range(len(MARIDA_CLASSES))
    }
    print("tiles-per-class (id: name -> tiles):")
    for i, (n, _d) in enumerate(MARIDA_CLASSES):
        print(f"  {i:2d} {n:24s} {counts_by_id[i]}")

    _write_summary(len(records), counts_by_id, len(cl_files))
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(records)
    )
    print(f"STATUS: completed classification num_samples={len(records)}")


def _write_summary(
    n_samples: int, counts_by_id: dict[int, int], n_patches: int
) -> None:
    lines = [
        f"# MARIDA (Marine Debris Archive) — {SLUG}",
        "",
        "- **Status**: completed",
        "- **Task type**: classification (dense_raster)",
        f"- **Samples**: {n_samples} label patches (64x64, UTM 10 m, uint8, nodata=255)",
        "- **Source**: Zenodo record 5151941 (Kikaki et al., PLOS ONE 2022); CC-BY-4.0",
        "- **URL**: https://doi.org/10.5281/zenodo.5151941",
        "- **Access**: public Zenodo download (`download_zenodo('5151941', raw_dir)`), no credentials.",
        "",
        "## What MARIDA is",
        "",
        "Manually photo-interpreted Sentinel-2 pixel annotations distinguishing marine "
        "debris from co-occurring sea-surface features. The release provides "
        f"{n_patches} patches, each a 256x256 Sentinel-2 crop already georeferenced in "
        "local UTM at 10 m/pixel, with a `*_cl.tif` class raster (float32; 0 = unlabeled, "
        "1-15 = classes) and a `*_conf.tif` confidence raster (1=High, 2=Moderate, "
        "3=Low). Annotations are sparse within each patch.",
        "",
        "## Processing",
        "",
        "- Each 256x256 `*_cl.tif` cropped into non-overlapping 64x64 UTM 10 m tiles "
        "(16 per patch); reused the source CRS/geotransform exactly (native UTM 10 m).",
        "- Kept every tile containing >=1 labeled pixel (3322 of 22096 candidate crops).",
        "- Class remap: MARIDA id 1-15 -> output id 0-14; unlabeled (0) -> 255 nodata.",
        "- **Sampling**: tiles-per-class balanced. The full candidate set (3322 tiles) is "
        "far below the 25k cap and below the 1000/class target for all classes except "
        "Marine Water (1606 tiles). Kept all tiles: dropping Marine-Water-heavy tiles "
        "would also remove co-present rare debris classes, so no truncation was applied. "
        "Marine Water is the only class above the 1000 guideline.",
        "- **Time range**: 1-day window of the Sentinel-2 acquisition date parsed from the "
        "scene name (`S2_dd-mm-yy_TILE`). Sea-surface features are transient, so each "
        "label is tied to its single acquisition (well under the 1-year limit).",
        "- All classes are natively annotated on 10 m Sentinel-2, so all 15 are viable at "
        "10 m (this dataset's raison d'être). Small-footprint classes (Ship, Wakes) are "
        "kept as annotated.",
        "",
        "## Classes (output id: name -> tiles containing class)",
        "",
    ]
    for i, (n, _d) in enumerate(MARIDA_CLASSES):
        lines.append(f"- {i}: {n} — {counts_by_id[i]}")
    lines += [
        "",
        "## Verification",
        "",
        "Output tifs are single-band uint8, UTM CRS at 10 m, 64x64, values in 0-14 plus "
        "255 nodata; each tif has a matching JSON with a 1-day time_range. Georeferencing "
        "reuses the source patches' exact UTM transform.",
        "",
        "## Reproduce",
        "",
        "```",
        f"python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.{SLUG}",
        "```",
        "",
    ]
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(lines))


if __name__ == "__main__":
    main()

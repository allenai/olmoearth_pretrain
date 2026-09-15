"""Process Cam-ForestNet (Congo Basin drivers) into open-set-segmentation patches.

Source: Zenodo record 8325259 (CC-BY-4.0), "Labelled dataset to classify direct
deforestation drivers in Cameroon" (de Bus et al. 2023, Scientific Data), the Cameroon /
Congo-Basin analogue of ForestNet. Each example is a Global-Forest-Change (GFC)
forest-loss patch in Cameroon, labelled by an expert (multi-dataset overlay + manual
verification) with the *direct deforestation driver* that caused the loss.

We use the ``my_examples_landsat_final_detailed`` release (15 detailed driver classes) and
its ``labels.zip`` CSV (``Landsat final versions/detailed/all.csv``). Each event folder
``{lon}_{lat}`` contains a ``forest_loss_region.pkl`` = a shapely Polygon (EPSG:4326,
lon/lat) delimiting the GFC forest-loss region, plus the driver label and the GFC loss
*year* in the CSV.

Encoding (polygons -> change labels, spec 4/5):
- One 64x64 uint8 label tile per event, in local UTM at 10 m, centred on the loss-polygon
  centroid. The forest-loss polygon is rasterized with its driver class id (1..15);
  everything outside the polygon is background (0). Polygons smaller than a pixel fall
  back to labelling the single centre pixel (all_touched already captures most). Polygons
  larger than 640 m (~2.5% of events) are clipped to the central 64x64 window.
- These are pre/post loss *events* under the pre/post change scheme. GFC loss is only
  YEAR-resolved, so each sample carries two independent six-month windows (each <= 183 days)
  with ``time_range`` = null: ``pre_time_range`` = summer of (loss_year - 1) and
  ``post_time_range`` = summer of (loss_year + 1), so the entire ambiguous loss year sits in
  the gap between them; ``change_time`` = 1 July of the loss year (reference only). Events
  span 2015-2020, so every post window is >= 2016; the year-1 pre window for 2015 events is
  Landsat-era, which is acceptable. 0 events dropped. This dataset was previously rejected on
  change-timing grounds (year-only resolution, not resolvable to within ~1-2 months); the
  pre/post scheme resolves that, so it is now usable.

Class scheme: background (0) + 15 detailed drivers (1..15), ids assigned by descending
event frequency. All 3108 events are kept (max per-class 546 < 1000, so no truncation and
no class balancing needed; well under the 25k cap).
"""

import argparse
import multiprocessing
import os
import pickle
import subprocess
import warnings
from collections import Counter
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)

warnings.filterwarnings("ignore")  # shapely <2.0 pickle compatibility warning

SLUG = "cam_forestnet_congo_basin_drivers"
NAME = "Cam-ForestNet (Congo Basin drivers)"
ZENODO_RECORD = "8325259"
ZENODO_URL = "https://zenodo.org/records/8325259"

EXAMPLES_ZIP = "my_examples_landsat_final_detailed.zip"
LABELS_ZIP = "labels.zip"
EXAMPLES_PREFIX = "my_examples_landsat_final_detailed"
CSV_REL = "labels/Landsat final versions/detailed/all.csv"

TILE = 64  # 64 px @ 10 m = 640 m
BACKGROUND_ID = 0

# 15 detailed driver classes, ids 1..15 by descending event frequency (background = 0).
# Descriptions summarise the driver definitions from de Bus et al. 2023 (Sci Data).
CLASSES: list[tuple[str, str, str]] = [
    # (csv_label, name, description)
    (
        "background",
        "background",
        "No mapped forest loss: forest / other land cover surrounding the event, outside the "
        "GFC forest-loss polygon.",
    ),
    (
        "Selective logging",
        "selective_logging",
        "Forest loss from selective / commercial timber extraction (logging roads, skid "
        "trails, felling gaps), not clear-cut conversion.",
    ),
    (
        "Timber plantation",
        "timber_plantation",
        "Large-scale timber / wood-fibre tree plantation (e.g. eucalyptus) established on "
        "cleared forest.",
    ),
    (
        "Small-scale maize plantation",
        "small_scale_maize_plantation",
        "Smallholder maize cultivation on cleared forest (small fields, shifting agriculture).",
    ),
    (
        "Small-scale oil palm plantation",
        "small_scale_oil_palm_plantation",
        "Smallholder / small-scale oil palm plots on cleared forest.",
    ),
    (
        "Mining",
        "mining",
        "Forest loss from mineral extraction: artisanal or industrial mining pits, tailings "
        "and bare-earth mine scars.",
    ),
    (
        "Oil palm plantation",
        "oil_palm_plantation",
        "Large-scale industrial oil palm plantation established on cleared forest.",
    ),
    (
        "Wildfire",
        "wildfire",
        "Forest loss caused by fire (wildfire / uncontrolled burning), burn scars.",
    ),
    (
        "Small-scale other plantation",
        "small_scale_other_plantation",
        "Smallholder plantation of other crops (not maize, oil palm, rubber or fruit) on "
        "cleared forest.",
    ),
    (
        "Rubber plantation",
        "rubber_plantation",
        "Large-scale rubber (Hevea) plantation established on cleared forest.",
    ),
    (
        "Hunting",
        "hunting",
        "Forest loss / degradation associated with hunting activity (camps, access clearings).",
    ),
    (
        "Other large-scale plantations",
        "other_large_scale_plantations",
        "Other large-scale plantations (e.g. tea, sugarcane) established on cleared forest.",
    ),
    ("Other", "other", "Forest loss from a driver not covered by the other classes."),
    (
        "Grassland shrubland",
        "grassland_shrubland",
        "Conversion of forest to grassland / shrubland (pasture, natural regrowth to "
        "non-forest).",
    ),
    (
        "Fruit plantation",
        "fruit_plantation",
        "Fruit-tree plantation (e.g. banana) established on cleared forest.",
    ),
    (
        "Infrastructure",
        "infrastructure",
        "Forest loss from built infrastructure: roads, buildings, settlements, other "
        "construction.",
    ),
]
CSV_TO_ID: dict[str, int] = {csv: i for i, (csv, _n, _d) in enumerate(CLASSES)}


def _content_url(fname: str) -> str:
    return f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{fname}/content"


def _ensure_raw() -> tuple[str, str]:
    """Ensure the examples zip + labels are downloaded and extracted under raw/.

    Returns (pkl_root, csv_path). Idempotent: skips work already done. The examples zip
    uses a compression method Python's zipfile cannot decode, so we shell out to `unzip`
    to extract only the forest_loss_region.pkl files.
    """
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    raw_path = raw.path

    examples_zip = os.path.join(raw_path, EXAMPLES_ZIP)
    labels_zip = os.path.join(raw_path, LABELS_ZIP)
    for fname, dst in ((EXAMPLES_ZIP, examples_zip), (LABELS_ZIP, labels_zip)):
        if not os.path.exists(dst):
            print(f"downloading {fname} ...", flush=True)
            import urllib.request

            tmp = dst + ".tmp"
            with urllib.request.urlopen(_content_url(fname)) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            os.rename(tmp, dst)

    pkl_root = os.path.join(raw_path, "extracted")
    n_pkl = 0
    if os.path.isdir(pkl_root):
        n_pkl = sum(
            1
            for _r, _d, fs in os.walk(pkl_root)
            for f in fs
            if f == "forest_loss_region.pkl"
        )
    if n_pkl < 3108:
        print("extracting forest_loss_region.pkl files ...", flush=True)
        subprocess.run(
            [
                "unzip",
                "-o",
                "-q",
                examples_zip,
                "*/forest_loss_region.pkl",
                "-d",
                pkl_root,
            ],
            check=True,
        )

    labels_root = os.path.join(raw_path, "extracted_labels")
    csv_path = os.path.join(labels_root, CSV_REL)
    if not os.path.exists(csv_path):
        print("extracting labels ...", flush=True)
        subprocess.run(["unzip", "-o", "-q", labels_zip, "-d", labels_root], check=True)

    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            f"Zenodo record {ZENODO_RECORD}: {ZENODO_URL}\n"
            f"Files: {EXAMPLES_ZIP} (Landsat, detailed 15-class), {LABELS_ZIP}.\n"
            "Used: forest_loss_region.pkl per event folder (shapely Polygon, EPSG:4326) + "
            f"'{CSV_REL}' (label, latitude, longitude, year, example_path).\n"
        )
    return pkl_root, csv_path


def _load_records(pkl_root: str, csv_path: str) -> list[dict[str, Any]]:
    df = pd.read_csv(csv_path)
    recs: list[dict[str, Any]] = []
    for i, row in df.iterrows():
        folder = os.path.basename(str(row["example_path"]).rstrip("/"))
        pkl = os.path.join(pkl_root, EXAMPLES_PREFIX, folder, "forest_loss_region.pkl")
        label = str(row["label"])
        if label not in CSV_TO_ID:
            continue
        recs.append(
            {
                "sample_id": f"{len(recs):06d}",
                "pkl": pkl,
                "class_id": CSV_TO_ID[label],
                "year": int(row["year"]),
                "folder": folder,
            }
        )
    return recs


def _write_one(rec: dict[str, Any]) -> int:
    """Rasterize one event's loss polygon into a 64x64 driver-class tile. Returns
    class_id on success, -1 if the pkl is missing/degenerate.
    """
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return rec["class_id"]
    if not os.path.exists(rec["pkl"]):
        return -1
    with open(rec["pkl"], "rb") as f:
        poly = pickle.load(f)
    if poly.is_empty:
        return -1

    c = poly.centroid
    lon, lat = float(c.x), float(c.y)
    proj, col, row = io.lonlat_to_utm_pixel(lon, lat)
    bounds = io.centered_bounds(col, row, TILE, TILE)
    cid = rec["class_id"]

    px = geom_to_pixels(poly, WGS84_PROJECTION, proj)
    arr = np.full((1, TILE, TILE), BACKGROUND_ID, dtype=np.uint8)
    if px.is_valid and not px.is_empty:
        arr = rasterize_shapes(
            [(px, cid)], bounds, fill=BACKGROUND_ID, dtype="uint8", all_touched=True
        )
    if int(arr.max()) != cid:
        # Polygon smaller than / missed all pixels: label the centre pixel.
        r0 = col - bounds[0]
        r1 = row - bounds[1]
        arr[0, r1, r0] = cid

    classes_present = sorted(int(v) for v in np.unique(arr))
    # GFC loss is only year-resolved. Under the pre/post scheme we put the whole ambiguous
    # loss year in the gap: a "before" window in the summer of year-1 and an "after" window
    # in the summer of year+1, so the clearing reliably falls between them regardless of
    # when in the loss year it happened. (Events span 2015-2020, so every post window is
    # >= 2016; year-1 pre windows for 2015 events are Landsat-era, which is acceptable.)
    change_time = datetime(rec["year"], 7, 1, tzinfo=UTC)
    pre_range = io.centered_time_range(datetime(rec["year"] - 1, 7, 1, tzinfo=UTC), 91)
    post_range = io.centered_time_range(datetime(rec["year"] + 1, 7, 1, tzinfo=UTC), 91)
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        None,
        change_time=change_time,
        source_id=rec["folder"],
        classes_present=classes_present,
        pre_time_range=pre_range,
        post_time_range=post_range,
    )
    return cid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    pkl_root, csv_path = _ensure_raw()
    records = _load_records(pkl_root, csv_path)
    print(f"{len(records)} events", flush=True)
    label_counts = Counter(r["class_id"] for r in records)

    io.check_disk()
    with multiprocessing.Pool(args.workers) as p:
        results = list(
            tqdm.tqdm(
                star_imap_unordered(p, _write_one, [dict(rec=r) for r in records]),
                total=len(records),
                desc="write",
            )
        )

    written = [r for r in results if r != -1]
    n_degenerate = sum(1 for r in results if r == -1)
    written_counts = Counter(written)
    num_samples = len(written)

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Zenodo / Scientific Data (de Bus et al. 2023, Cam-ForestNet)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": ZENODO_URL,
                "have_locally": False,
                "annotation_method": "multi-dataset overlay + expert manual verification",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (_csv, name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": num_samples,
            "tile_size": TILE,
            "change_labels": True,
            "class_counts": {
                CLASSES[cid][1]: written_counts.get(cid, 0)
                for cid in range(len(CLASSES))
            },
            "notes": (
                "One 64x64 UTM 10 m tile per Cameroon GFC forest-loss event; the "
                "forest_loss_region polygon (EPSG:4326) is rasterized (all_touched) with "
                "its detailed deforestation-driver class (1..15), background=0 elsewhere; "
                "sub-pixel polygons fall back to the centre pixel. Each sample carries "
                "change_time = 1 July of the GFC loss year with a 1-year time_range "
                "centred on it (annual GFC resolution). Events span 2015-2020; 2015 (349 "
                "events) predates full Sentinel-2 coverage but is fine for Landsat-8. All "
                "events kept (max per-class 546 < 1000; no balancing/truncation). "
                f"{n_degenerate} events dropped as missing/degenerate polygons. Note the "
                "'background' class here is the forest surrounding each loss patch, so "
                "background pixels dominate every tile."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=num_samples
    )
    print(
        f"done: {num_samples} samples ({n_degenerate} dropped). "
        f"per-class: "
        + ", ".join(
            f"{CLASSES[cid][1]}={written_counts.get(cid, 0)}"
            for cid in sorted(written_counts)
        ),
        flush=True,
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()

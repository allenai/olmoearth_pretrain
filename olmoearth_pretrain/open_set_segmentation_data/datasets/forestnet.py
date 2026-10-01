"""Process ForestNet (Irvin & Sheng et al. 2020) into open-set-segmentation patches.

Source: Stanford ML Group, "ForestNet: Classifying Drivers of Deforestation in
Indonesia using Deep Learning on Satellite Imagery" (NeurIPS 2020 CCAI workshop),
CC-BY-4.0. Downloaded as a single ~3.4 GB zip
(``http://download.cs.stanford.edu/deep/ForestNetDataset.zip``).

Each of the 2,757 examples is a Global-Forest-Change (GFC) primary-forest-loss event in
Indonesia (2001-2016), captured as a 332x332 px Landsat-8 image (visible bands
pan-sharpened to 15 m/px) centred on the loss region. Per example:
- ``forest_loss_region.pkl`` -- a shapely (Multi)Polygon delimiting the GFC forest-loss
  region, expressed in the 332x332 **image pixel grid** (not lon/lat); image-centre pixel
  (166, 166) corresponds to the CSV (latitude, longitude).
- the CSV row (train/val/test.csv) carries the fine driver ``label``, the coarse
  ``merged_label``, the image centre ``latitude``/``longitude``, and the GFC loss ``year``.

We only need the labels (CSVs + forest_loss_region.pkl); the imagery / auxiliary layers are
not extracted (pretraining supplies its own imagery).

Task recast (spec 2/5) -- presence/state classification, NOT a change label:
- ForestNet's driver date is only the GFC **annual** loss year (2001-2016), which is coarser
  than the spec's ~1-2 month change-timing requirement, so we do NOT emit dated change
  labels. Instead we treat each event as **presence/state classification of the persistent
  post-deforestation land-use driver** (oil-palm / timber / smallholder / grassland / ...),
  which stays visible for years after the clearing. change_time = null; time_range = a
  static 1-year window in the Sentinel era, ``year_range(max(loss_year + 1, 2016))`` (i.e.
  2016 or 2017), so the persistent land-use state is observable by Sentinel-2 even though
  every loss event predates 2016. This is the spec-5 "persistent post-change state ->
  presence/state classification" path, and it is also how we honour the post-2016 rule
  (the labelled *state*, not the historical loss event, is what the imagery window sees).

Encoding (polygons, spec 4): one uint8 label tile per event in local UTM at 10 m/px. The
loss polygon is scaled from the 15 m image grid to the 10 m UTM grid (x1.5), placed at the
event's UTM location, and rasterized (all_touched) with its driver class id; everything
outside the polygon is 255 = nodata/ignore (we do NOT fabricate a background class -- the
land use outside the mapped loss region is unknown; spec 5). Tiles are sized to the
footprint + a 10 px context ring, clamped to 32..64 px; polygons larger than 64 px are
clipped to the central 64x64 window; sub-pixel polygons fall back to the centre pixel.

Class scheme: the 12 fine driver classes (ids 0..11 by descending event frequency); each
class records its coarse ForestNet group (Plantation / Smallholder agriculture /
Grassland shrubland / Other) in its description. All 2,757 events kept (max per-class 599 <
1000; well under the 25k cap; no balancing/truncation).
"""

import argparse
import multiprocessing
import os
import pickle
import subprocess
import warnings
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
import shapely.affinity
import tqdm
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.rasterize import rasterize_shapes

warnings.filterwarnings("ignore")  # shapely <2.0 pickle compatibility warning

SLUG = "forestnet"
NAME = "ForestNet"
DATASET_URL = "https://stanfordmlgroup.github.io/projects/forestnet/"
ZIP_URL = "http://download.cs.stanford.edu/deep/ForestNetDataset.zip"
ZIP_NAME = "ForestNetDataset.zip"
DATA_SUBDIR = os.path.join("extracted", "deep", "downloads", "ForestNetDataset")

IMG_SIZE = 332  # ForestNet image is 332x332 px
IMG_CENTER = IMG_SIZE / 2.0  # 166.0; image centre pixel == CSV (lat, lon)
SRC_RES = 15.0  # source image resolution (m/px): visible bands pan-sharpened to 15 m
SCALE = SRC_RES / io.RESOLUTION  # 15 m -> 10 m == 1.5 UTM px per image px

MARGIN_PX = 10  # context ring around the footprint (in 10 m UTM px)
MIN_TILE = 32
MAX_TILE = io.MAX_TILE  # 64
SENTINEL_MIN_YEAR = 2016  # first full Sentinel-2 year; state window floored here

# 12 fine driver classes, ids 0..11 by descending event frequency.
# (csv_label, name, merged_group, description)
CLASSES: list[tuple[str, str, str, str]] = [
    (
        "Oil palm plantation",
        "oil_palm_plantation",
        "Plantation",
        "Large-scale / industrial oil-palm (Elaeis guineensis) plantation established on "
        "cleared primary forest. ForestNet coarse group: Plantation.",
    ),
    (
        "Small-scale agriculture",
        "small_scale_agriculture",
        "Smallholder agriculture",
        "Smallholder / small-scale cultivated cropland on cleared forest (small fields, "
        "shifting agriculture). ForestNet coarse group: Smallholder agriculture.",
    ),
    (
        "Timber plantation",
        "timber_plantation",
        "Plantation",
        "Large-scale timber / wood-fibre tree plantation (e.g. Acacia, eucalyptus) on cleared "
        "forest. ForestNet coarse group: Plantation.",
    ),
    (
        "Grassland shrubland",
        "grassland_shrubland",
        "Grassland shrubland",
        "Conversion of forest to grassland / shrubland (pasture, degraded scrub, natural "
        "non-forest regrowth). ForestNet coarse group: Grassland shrubland.",
    ),
    (
        "Small-scale mixed plantation",
        "small_scale_mixed_plantation",
        "Smallholder agriculture",
        "Smallholder mixed-crop / agroforestry plantation on cleared forest. ForestNet coarse "
        "group: Smallholder agriculture.",
    ),
    (
        "Other large-scale plantations",
        "other_large_scale_plantations",
        "Plantation",
        "Other large-scale plantations (crops other than oil palm / timber) established on "
        "cleared forest. ForestNet coarse group: Plantation.",
    ),
    (
        "Small-scale oil palm plantation",
        "small_scale_oil_palm_plantation",
        "Smallholder agriculture",
        "Smallholder / small-scale oil-palm plots on cleared forest. ForestNet coarse group: "
        "Smallholder agriculture.",
    ),
    (
        "Secondary forest",
        "secondary_forest",
        "Other",
        "Regrowth / secondary forest or forest degradation following the loss event (no clear "
        "conversion to another land use). ForestNet coarse group: Other.",
    ),
    (
        "Other",
        "other",
        "Other",
        "Forest loss from a driver not covered by the other classes. ForestNet coarse group: "
        "Other.",
    ),
    (
        "Mining",
        "mining",
        "Other",
        "Forest loss from mineral extraction (artisanal or industrial mining pits, tailings, "
        "bare-earth mine scars). ForestNet coarse group: Other.",
    ),
    (
        "Logging",
        "logging",
        "Other",
        "Forest loss from selective / commercial timber logging (logging roads, felling gaps), "
        "not clear-cut conversion. ForestNet coarse group: Other.",
    ),
    (
        "Fish pond",
        "fish_pond",
        "Other",
        "Forest / mangrove cleared for aquaculture fish ponds. ForestNet coarse group: Other.",
    ),
]
CSV_TO_ID: dict[str, int] = {csv: i for i, (csv, _n, _g, _d) in enumerate(CLASSES)}


def _ensure_raw() -> str:
    """Ensure the zip is downloaded and the CSVs + forest_loss_region.pkl are extracted.

    Returns the path to the ForestNetDataset dir. Idempotent: skips work already done.
    Only labels are extracted (CSVs + polygons); imagery / auxiliary layers are skipped.
    """
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    raw_path = raw.path

    zip_path = os.path.join(raw_path, ZIP_NAME)
    if not os.path.exists(zip_path):
        print(f"downloading {ZIP_NAME} ...", flush=True)
        import urllib.request

        tmp = zip_path + ".tmp"
        with urllib.request.urlopen(ZIP_URL) as r, open(tmp, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        os.rename(tmp, zip_path)

    data_dir = os.path.join(raw_path, DATA_SUBDIR)
    csvs_ok = all(
        os.path.exists(os.path.join(data_dir, f"{s}.csv"))
        for s in ("train", "val", "test")
    )
    if not csvs_ok:
        print("extracting CSVs ...", flush=True)
        subprocess.run(
            [
                "unzip",
                "-o",
                "-q",
                zip_path,
                "deep/downloads/ForestNetDataset/*.csv",
                "-d",
                os.path.join(raw_path, "extracted"),
            ],
            check=True,
        )

    n_pkl = 0
    ex_dir = os.path.join(data_dir, "examples")
    if os.path.isdir(ex_dir):
        n_pkl = sum(
            1
            for _r, _d, fs in os.walk(ex_dir)
            if "forest_loss_region.pkl" in fs
            for _f in [0]
        )
    if n_pkl < 2757:
        print("extracting forest_loss_region.pkl files ...", flush=True)
        subprocess.run(
            [
                "unzip",
                "-o",
                "-q",
                zip_path,
                "deep/downloads/ForestNetDataset/examples/*/forest_loss_region.pkl",
                "-d",
                os.path.join(raw_path, "extracted"),
            ],
            check=True,
        )

    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            f"ForestNet dataset: {DATASET_URL}\n"
            f"Downloaded: {ZIP_URL} ({ZIP_NAME}, ~3.4 GB, CC-BY-4.0).\n"
            "Labels used (imagery NOT extracted): train/val/test.csv (label, merged_label, "
            "latitude, longitude, year, example_path) + examples/*/forest_loss_region.pkl "
            "(shapely polygon in the 332x332 15 m image-pixel grid; centre px (166,166) == "
            "CSV lat/lon).\n"
        )
    return data_dir


def _load_records(data_dir: str) -> list[dict[str, Any]]:
    frames = []
    for split in ("train", "val", "test"):
        df = pd.read_csv(os.path.join(data_dir, f"{split}.csv"))
        df["split"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    recs: list[dict[str, Any]] = []
    for _i, row in df.iterrows():
        label = str(row["label"])
        if label not in CSV_TO_ID:
            continue
        folder = os.path.basename(str(row["example_path"]).rstrip("/"))
        pkl = os.path.join(data_dir, "examples", folder, "forest_loss_region.pkl")
        recs.append(
            {
                "sample_id": f"{len(recs):06d}",
                "pkl": pkl,
                "class_id": CSV_TO_ID[label],
                "lon": float(row["longitude"]),
                "lat": float(row["latitude"]),
                "year": int(row["year"]),
                "split": str(row["split"]),
                "folder": folder,
            }
        )
    return recs


def _write_one(rec: dict[str, Any]) -> int:
    """Rasterize one event's loss polygon into a driver-class tile. Returns class_id on
    success, -1 if the pkl is missing/degenerate.
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

    cid = rec["class_id"]
    proj, col0, row0 = io.lonlat_to_utm_pixel(rec["lon"], rec["lat"])
    # Image pixel (px, py) -> UTM pixel: scale x1.5 about the image centre, then translate
    # so the image centre (166,166) maps to the UTM centre pixel (col0, row0). Both grids
    # have row increasing southward, so no axis flip is needed.
    poly_utm = shapely.affinity.affine_transform(
        poly,
        [SCALE, 0, 0, SCALE, col0 - SCALE * IMG_CENTER, row0 - SCALE * IMG_CENTER],
    )
    cx, cy = poly_utm.centroid.x, poly_utm.centroid.y
    minx, miny, maxx, maxy = poly_utm.bounds
    size = int(round(max(maxx - minx, maxy - miny))) + 2 * MARGIN_PX
    size = max(MIN_TILE, min(MAX_TILE, size))
    bounds = io.centered_bounds(int(round(cx)), int(round(cy)), size, size)

    arr = rasterize_shapes(
        [(poly_utm, cid)], bounds, fill=io.CLASS_NODATA, dtype="uint8", all_touched=True
    )
    if not (arr == cid).any():
        # Polygon smaller than / offset from all pixels: label the centre pixel.
        arr[0, arr.shape[1] // 2, arr.shape[2] // 2] = cid

    classes_present = sorted(int(v) for v in np.unique(arr) if v != io.CLASS_NODATA)
    window_year = max(rec["year"] + 1, SENTINEL_MIN_YEAR)
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(window_year),
        change_time=None,
        source_id=f"{rec['split']}/{rec['folder']}",
        classes_present=classes_present,
    )
    return cid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    data_dir = _ensure_raw()
    records = _load_records(data_dir)
    print(f"{len(records)} events", flush=True)

    io.check_disk()
    io.locations_dir(SLUG).mkdir(parents=True, exist_ok=True)
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
            "source": "Stanford ML Group / NeurIPS 2020 CCAI (Irvin & Sheng et al. 2020)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": DATASET_URL,
                "have_locally": False,
                "annotation_method": "expert photointerpretation (Google Earth) of GFC "
                "forest-loss polygons",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (_csv, name, _g, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": num_samples,
            "region": "Indonesia",
            "change_labels": False,
            "class_counts": {
                CLASSES[cid][1]: written_counts.get(cid, 0)
                for cid in range(len(CLASSES))
            },
            "notes": (
                "One local-UTM 10 m tile per Indonesian GFC forest-loss event (ForestNet, "
                "2001-2016). The forest_loss_region polygon (332x332 15 m image-pixel grid) "
                "is scaled to the 10 m UTM grid (x1.5), centred on the event, and rasterized "
                "(all_touched) with its fine deforestation-driver class (0..11); pixels "
                "outside the loss polygon are 255 = nodata/ignore (no fabricated background "
                "class). Tiles are sized to the footprint + 10 px context, clamped 32..64 px "
                "(larger polygons clipped to 64x64). RECAST AS PRESENCE/STATE classification "
                "(not a change label): the GFC loss date is only annual (coarser than the "
                "~1-2 month change-timing rule), so change_time is null and each sample gets "
                "a static 1-year window year_range(max(loss_year+1, 2016)) = 2016 or 2017, "
                "capturing the persistent post-deforestation land-use driver in the "
                "Sentinel-2 era. Caveat: for older loss events (pre-2012) the mapped driver "
                "is assumed to persist to 2016/2017; this holds for stable land uses "
                "(plantations, agriculture) but grassland/secondary-forest states may have "
                "changed. All train/val/test events kept (max per-class 599 < 1000; no "
                f"balancing/truncation). {n_degenerate} events dropped as missing/degenerate "
                "polygons. Fine classes map to ForestNet coarse groups Plantation / "
                "Smallholder agriculture / Grassland shrubland / Other (see class "
                "descriptions)."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=num_samples
    )
    print(
        f"done: {num_samples} samples ({n_degenerate} dropped). per-class: "
        + ", ".join(
            f"{CLASSES[cid][1]}={written_counts.get(cid, 0)}"
            for cid in sorted(written_counts)
        ),
        flush=True,
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()

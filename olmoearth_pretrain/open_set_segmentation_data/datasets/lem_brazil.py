"""Process LEM+ (Brazil) into open-set-segmentation label patches (rasterized crop polygons).

Source: "LEM: A dataset for crop type mapping" / LEM+ (Mendeley Data vz6d7tw87f, v1),
Sanches et al. Monthly ground-truth crop / land-use labels for 1,854 field polygons in
tropical western Bahia, Brazil, covering the agricultural year **October 2019 - September
2020** (12 monthly columns Oct_2019 ... Sep_2020). Vector = ESRI shapefile in WGS84.
Licensed CC-BY-4.0. Total archive ~1.2 MB (labels only; pretraining supplies imagery).

Task: per-pixel **classification** (crop type). This is a **double-cropping** region, so a
single field's label changes month to month (e.g. Uncultivated soil -> Soybean -> Corn ->
Brachiaria within one year). To preserve that signal without emitting contradictory
supervision at one location, we split each polygon's 12-month label sequence into
**crop episodes** = maximal runs of consecutive months with the same label. Each episode
becomes one sample:
  - geometry = the field polygon (rasterized into a <=64x64 UTM 10 m tile centered on the
    polygon centroid; the crop class id is burned inside the polygon, 255=nodata/ignore
    outside -- we only have ground truth inside surveyed fields, so unlabeled land is
    ignore, not a background class);
  - class = the episode's crop label;
  - time_range = a window spanning the episode's months (first day of its first month to
    the first day of the month after its last), clamped to <= 360 days.
Consecutive episodes at a field have disjoint month spans, so their time windows do not
overlap and there is no contradictory multi-label supervision. This is the intended
"coherent 1-year window" (the Oct 2019 - Sep 2020 agricultural year) subdivided into the
per-crop episodes the monthly ground truth actually records; transient crops get their true
sub-year presence window, perennials/fallow that persist all year get a ~1-year window.

The label "Not identified" (annotator could not determine the crop) is treated as
ignore -- it breaks an episode run and never becomes a class. All other 15 labels are kept
as classes; ids are assigned 0..N-1 in descending global episode frequency.

Sampling: class-balanced with the 25k per-dataset cap; up to 1000 episodes per class
(effective cap min(1000, 25000 // n_classes)). Rare classes (e.g. Crotalaria=2) are kept.

Run (idempotent; skips already-written {sample_id}.tif):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.lem_brazil
"""

import argparse
import multiprocessing
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pyogrio
import shapely
import tqdm
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "lem_brazil"
NAME = "LEM+ (Brazil)"
URL = "https://data.mendeley.com/datasets/vz6d7tw87f/1"
DOWNLOAD_URL = (
    "https://data.mendeley.com/public-files/datasets/vz6d7tw87f/files/"
    "57c83c3f-b5a9-45f5-94f8-ac1df8fab923/file_downloaded"
)

# Monthly label columns in chronological order; each entry is (column, year, month).
MONTHS = [
    ("Oct_2019", 2019, 10),
    ("Nov_2019", 2019, 11),
    ("Dec_2019", 2019, 12),
    ("Jan_2020", 2020, 1),
    ("Feb_2020", 2020, 2),
    ("Mar_2020", 2020, 3),
    ("Apr_2020", 2020, 4),
    ("May_2020", 2020, 5),
    ("Jun_2020", 2020, 6),
    ("Jul_2020", 2020, 7),
    ("Aug_2020", 2020, 8),
    ("Sep_2020", 2020, 9),
]

# Labels that mean "no usable class" -> break an episode, never become a class.
IGNORE_LABELS = {"Not identified"}

# Short per-class definitions (source is a Brazilian crop/land-use field survey).
CLASS_DESCRIPTIONS = {
    "Uncultivated soil": "Bare / fallow agricultural soil with no active crop in the month.",
    "Soybean": "Soybean (Glycine max), the dominant summer commodity crop in the region.",
    "Millet": "Millet, commonly grown as a second (safrinha) or cover crop.",
    "Brachiaria": "Brachiaria forage grass (Urochloa spp.), used for pasture / cover / integrated crop-livestock.",
    "Corn": "Corn / maize (Zea mays), frequently the second crop after soybean.",
    "Sorghum": "Sorghum, a second-season cereal.",
    "Cerrado": "Native Cerrado savanna vegetation (uncultivated natural land).",
    "Cotton": "Cotton (Gossypium spp.).",
    "Pasture": "Managed pasture / grazing land.",
    "Beans": "Common beans (Phaseolus vulgaris).",
    "Conversion area": "Land being cleared / converted from native Cerrado to agricultural use.",
    "Eucalyptus": "Eucalyptus plantation (incl. some in an early development stage).",
    "Hay": "Hay / cut forage.",
    "Coffee": "Coffee (Coffea), incl. some early-stage or recently pruned plots.",
    "Crotalaria": "Crotalaria, a legume cover / green-manure crop.",
}

PER_CLASS = 1000
MAX_TILE = io.MAX_TILE  # 64
MAX_WINDOW_DAYS = 360  # spec: time_range must be <= ~1 year

_WGS84_SRC = Projection(CRS.from_epsg(4326), 1, 1)


def ensure_data() -> str:
    """Download + unzip the LEM shapefile; return the .shp path. Write SOURCE.txt."""
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    zip_path = raw / "LEM_dataset.zip"
    download.download_http(
        DOWNLOAD_URL, zip_path, headers={"User-Agent": "Mozilla/5.0"}
    )
    download.extract_zip(zip_path, raw)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "LEM+ (Brazil) crop-type field survey. Mendeley Data vz6d7tw87f v1, CC-BY-4.0.\n"
            f"{URL}\n"
            "LEM_dataset.shp: 1854 field polygons (WGS84) with monthly crop/land-use labels\n"
            "Oct_2019 ... Sep_2020. Labels only; no imagery.\n"
        )
    return str(raw / "LEM_dataset.shp")


def build_episodes(row: Any) -> list[dict[str, Any]]:
    """Split one polygon's 12 monthly labels into consecutive-run episodes.

    Returns a list of {label, start_idx, end_idx} (inclusive month indices into MONTHS).
    IGNORE_LABELS / null values break runs and are dropped.
    """
    episodes: list[dict[str, Any]] = []
    prev: str | None = None
    for i, (col, _y, _m) in enumerate(MONTHS):
        v = row[col]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            v = None
        else:
            v = str(v).strip()
            if v == "" or v in IGNORE_LABELS:
                v = None
        if v is not None and v == prev:
            episodes[-1]["end_idx"] = i
        elif v is not None:
            episodes.append({"label": v, "start_idx": i, "end_idx": i})
        prev = v
    return episodes


def episode_time_range(start_idx: int, end_idx: int) -> tuple[datetime, datetime]:
    """1-year-or-shorter UTC window spanning [start month .. end month], clamped."""
    _c, sy, sm = MONTHS[start_idx]
    start = datetime(sy, sm, 1, tzinfo=UTC)
    _c, ey, em = MONTHS[end_idx]
    # First day of the month AFTER the last episode month.
    if em == 12:
        end = datetime(ey + 1, 1, 1, tzinfo=UTC)
    else:
        end = datetime(ey, em + 1, 1, tzinfo=UTC)
    if (end - start).days > MAX_WINDOW_DAYS:
        end = start + timedelta(days=MAX_WINDOW_DAYS)
    return start, end


def _write_tile(rec: dict[str, Any]) -> tuple[str, str, int]:
    sample_id = rec["sample_id"]
    class_id = int(rec["class_id"])
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return sample_id, "skip", class_id
    try:
        geom = shapely.from_wkb(rec["geom_wkb"])  # WGS84 lon/lat
        proj = io.utm_projection_for_lonlat(rec["lon"], rec["lat"])
        pix = geom_to_pixels(geom, _WGS84_SRC, proj)
        minx, miny, maxx, maxy = pix.bounds
        cx = int(round((minx + maxx) / 2))
        cy = int(round((miny + maxy) / 2))
        w = min(MAX_TILE, max(1, int(np.ceil(maxx - minx))))
        h = min(MAX_TILE, max(1, int(np.ceil(maxy - miny))))
        bounds = io.centered_bounds(cx, cy, w, h)
        arr = rasterize_shapes(
            [(pix, class_id)],
            bounds,
            fill=io.CLASS_NODATA,
            dtype="uint8",
            all_touched=True,
        )
        if not (arr != io.CLASS_NODATA).any():
            return sample_id, "empty", class_id
        start = datetime.fromisoformat(rec["t_start"])
        end = datetime.fromisoformat(rec["t_end"])
        io.write_label_geotiff(
            SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA
        )
        io.write_sample_json(
            SLUG,
            sample_id,
            proj,
            bounds,
            (start, end),
            source_id=rec["source_id"],
            classes_present=sorted(set(np.unique(arr).tolist()) - {io.CLASS_NODATA}),
        )
        return sample_id, "ok", class_id
    except Exception as e:  # noqa: BLE001
        print(f"error on {sample_id}: {e}")
        return sample_id, "error", class_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    shp_path = ensure_data()
    gdf = pyogrio.read_dataframe(shp_path)
    gdf = gdf.to_crs(4326)
    print(f"read {len(gdf)} field polygons")

    # ---- Build all episodes; compute global class frequency ------------------------
    raw_records: list[dict[str, Any]] = []
    freq: Counter = Counter()
    for fid, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        cent = geom.centroid
        if not (np.isfinite(cent.x) and np.isfinite(cent.y)):
            continue
        wkb = shapely.to_wkb(geom)
        for ep in build_episodes(row):
            freq[ep["label"]] += 1
            raw_records.append(
                {
                    "label": ep["label"],
                    "fid": int(fid),
                    "start_idx": ep["start_idx"],
                    "end_idx": ep["end_idx"],
                    "lon": float(cent.x),
                    "lat": float(cent.y),
                    "geom_wkb": wkb,
                }
            )
    print(f"total episodes: {len(raw_records)} across {len(freq)} classes")

    # ---- Assign class ids by descending global episode frequency -------------------
    ranked = [lbl for lbl, _ in freq.most_common()]
    label_to_id = {lbl: i for i, lbl in enumerate(ranked)}
    for r in raw_records:
        r["class_id"] = label_to_id[r["label"]]
    print("class frequency:", {lbl: freq[lbl] for lbl in ranked})

    # ---- Class-balanced selection (<=1000/class, 25k cap) --------------------------
    selected = balance_by_class(
        raw_records, key="class_id", per_class=PER_CLASS, total_cap=25000
    )
    print(f"selected {len(selected)} episodes after balancing")

    # ---- Finalize records: sample ids + episode time windows -----------------------
    tile_recs: list[dict[str, Any]] = []
    for i, r in enumerate(selected):
        start, end = episode_time_range(r["start_idx"], r["end_idx"])
        s_col = MONTHS[r["start_idx"]][0]
        e_col = MONTHS[r["end_idx"]][0]
        tile_recs.append(
            {
                "sample_id": f"{i:06d}",
                "class_id": r["class_id"],
                "lon": r["lon"],
                "lat": r["lat"],
                "geom_wkb": r["geom_wkb"],
                "t_start": start.isoformat(),
                "t_end": end.isoformat(),
                "source_id": f"field{r['fid']}/{s_col}-{e_col}",
            }
        )

    # ---- Write tiles in parallel ---------------------------------------------------
    results: Counter = Counter()
    written_by_class: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for sample_id, res, class_id in tqdm.tqdm(
            star_imap_unordered(p, _write_tile, [dict(rec=r) for r in tile_recs]),
            total=len(tile_recs),
        ):
            results[res] += 1
            if res in ("ok", "skip"):
                written_by_class[class_id] += 1
    print("write results:", dict(results))
    io.check_disk()

    # ---- Metadata ------------------------------------------------------------------
    classes = [
        {
            "id": cid,
            "name": lbl,
            "description": CLASS_DESCRIPTIONS.get(lbl),
        }
        for lbl, cid in sorted(label_to_id.items(), key=lambda kv: kv[1])
    ]
    class_counts = {
        lbl: int(written_by_class.get(cid, 0))
        for lbl, cid in sorted(label_to_id.items(), key=lambda kv: kv[1])
    }
    num_written = int(results.get("ok", 0) + results.get("skip", 0))
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Mendeley Data",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": URL,
                "have_locally": False,
                "annotation_method": "manual field survey (monthly crop/land-use labels)",
                "region": "Western Bahia, Brazil",
                "agricultural_year": "2019-10 .. 2020-09",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": classes,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": num_written,
            "class_counts": class_counts,
            "notes": (
                "1854 field polygons in tropical western Bahia with monthly crop/land-use "
                "labels for the Oct 2019 - Sep 2020 agricultural year. Each field's 12 "
                "monthly labels are split into crop episodes (maximal consecutive-month "
                "runs of the same label); each episode is one sample: the field polygon "
                "rasterized into a <=64x64 UTM 10 m tile (class id inside, 255=nodata "
                "outside; no background class -- unlabeled land is ignore), with a "
                "time_range spanning the episode's months (clamped to <=360 days). "
                "'Not identified' labels are treated as ignore. Class ids 0..N-1 by "
                "descending global episode frequency. Class-balanced, <=1000/class, 25k cap."
            ),
        },
    )

    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=num_written
    )
    print(f"done: {num_written} samples across {len(classes)} classes")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()

"""Process DynamicEarthNet monthly land-cover labels into open-set-segmentation patches.

Source: DynamicEarthNet (Toker et al., CVPR 2022), TUM. 75 global AOIs of daily 4-band
PlanetFusion imagery (3 m, 2018-01-01..2019-12-31) paired with MONTHLY pixel-wise 7-class
land-cover labels. Distributed on mediaTUM (node 1650201) via https://dataserv.ub.tum.de/index.php/s/m1650201.
License CC-BY-SA-4.0.

Label-only download (task spec: only the LABELS are wanted, NOT the multi-hundred-GB Planet
imagery): the data server ships a dedicated ``labels.zip`` (1.4 GB) separate from the
``planet.*.zip`` image cubes (~500 GB total), so we pull ONLY ``labels.zip``. Inside it,
each AOI has ``Labels/Raster/<AOI>/<AOI>-YYYY-MM-01.tif``: a 7-band, 1024x1024, 3 m, uint8
one-hot land-cover mask in a local UTM CRS (each band is 0 or 255; band b active => class b).
1320 raster labels are provided = 55 AOIs x 24 months (2018-01 .. 2019-12). (The paper's 75
AOIs include a held-out test set whose labels are not in labels.zip; we use all 55 that ship
raster masks; all train/val splits are fair game per task spec 5.) A Vector/ variant of the
same labels is ignored.

Class mapping (output id = source band index; matches the manifest class order):
  0 impervious surface, 1 agriculture, 2 forest & other vegetation, 3 wetlands,
  4 bare soil, 5 water, 6 snow & ice. Pixels with no active band (rare; 6515 px total across
  all 1.38e9 label px) -> nodata 255. NOTE: the official DynamicEarthNet benchmark uses only
  6 classes and maps the snow-&-ice band (6) to ignore; for this label bank we KEEP snow & ice
  as a real class (present in 48 of 1320 AOI-months) per task-spec-5 "keep every class you
  can" (downstream assembly filters classes that end up too sparse).

VHR handling (task spec 4 / 8): the 3 m one-hot mask is collapsed to a single-band class-id
raster, then reprojected within its native UTM CRS from 3 m to 10 m with MODE resampling
(categorical majority; never bilinear), giving a ~308x308 AOI grid, which is cut into
non-overlapping <=64x64 tiles. A tile is dropped if fully nodata. All 7 classes survive
mode resampling at 10 m (land-cover zones are large relative to 10 m); no class is dropped.

Time range (task spec 5, seasonal/annual): each monthly label is the per-month LAND-COVER
STATE (not a dated change event), so change_time=null and time_range is a ~3-month window
centered on the label's month (centered_time_range(center=15th of the label month,
half_window_days=45), i.e. ~90 days bracketing that calendar month).

Sampling: tiles-per-class balanced (spec 5) via sampling.select_tiles_per_class -
<=1000 tiles/class, rarest-class-first, total capped at 25k. A tile counts toward every class
present in it. Scan records (reprojected tiles) are cached to raw/{slug}/scan_cache.pkl.

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.dynamicearthnet
"""

import argparse
import math
import multiprocessing
import pickle
import re
import zipfile
from datetime import UTC, datetime
from io import BytesIO
from typing import Any

import numpy as np
import rasterio
import tqdm
from affine import Affine
from rasterio.warp import Resampling, reproject
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    select_tiles_per_class,
)

SLUG = "dynamicearthnet"
NAME = "DynamicEarthNet"
ZIP_NAME = "labels.zip"
RSYNC_URL = "rsync://m1650201@dataserv.ub.tum.de/m1650201/labels.zip"
RSYNC_PASSWORD = "m1650201"
TARGET_RES = 10.0
TILE = io.MAX_TILE  # 64
PER_CLASS = 1000

# Output id -> (name, description). Output id == source band index.
CLASSES = [
    (
        "impervious surface",
        "Human-made sealed/paved surfaces: buildings, roads, parking lots, and other "
        "artificial impervious ground. DynamicEarthNet class 0.",
    ),
    (
        "agriculture",
        "Cultivated land: cropland, farmland, paddy, plantations, managed pasture. "
        "DynamicEarthNet class 1.",
    ),
    (
        "forest & other vegetation",
        "Natural vegetation: forest, woodland, shrubland, grassland and other (semi-)natural "
        "green cover. DynamicEarthNet class 2.",
    ),
    (
        "wetlands",
        "Vegetated wetlands, marsh, and periodically inundated land. DynamicEarthNet class 3.",
    ),
    (
        "bare soil",
        "Exposed bare ground with little/no vegetation: soil, sand, rock, dry riverbeds. "
        "DynamicEarthNet class 4.",
    ),
    (
        "water",
        "Open water: rivers, lakes, reservoirs, ponds, sea. DynamicEarthNet class 5.",
    ),
    (
        "snow & ice",
        "Persistent or seasonal snow and ice cover. Rare in this dataset (present in only 48 "
        "of 1320 AOI-months) and ignored by the official 6-class benchmark; kept here as a "
        "real class. DynamicEarthNet class 6.",
    ),
]
NUM_CLASSES = len(CLASSES)  # 7


def _zip_path() -> Any:
    return io.raw_dir(SLUG) / ZIP_NAME


_ZIP: zipfile.ZipFile | None = None


def _worker_init() -> None:
    global _ZIP
    _ZIP = zipfile.ZipFile(str(_zip_path()), "r")


# Date embedded in the label filename, with either '-' or '_' separators, e.g.
# ...-SR-2018-03-01.tif or ...-SR-2019_11_01.tif
_DATE_RE = re.compile(r"(\d{4})[-_](\d{2})[-_]\d{2}\.tif$")


def _list_label_members() -> list[str]:
    # Most AOIs: labels/<AOI>/Labels/Raster/...; one AOI (1417_3281_13_11N) omits the
    # "Labels" level (labels/<AOI>/Raster/...). Match "/Raster/" to cover both.
    with zipfile.ZipFile(str(_zip_path()), "r") as z:
        return sorted(n for n in z.namelist() if "/Raster/" in n and n.endswith(".tif"))


def _one_hot_to_classid(arr: np.ndarray) -> np.ndarray:
    """(7,H,W) one-hot (0/255) -> (H,W) uint8 class id; no active band -> 255 nodata."""
    active = arr == 255  # (7,H,W) bool
    count = active.sum(axis=0)
    classid = active.argmax(axis=0).astype(np.uint8)  # 0..6 (0 where all-zero)
    classid[count == 0] = io.CLASS_NODATA
    return classid


def _reproject_to_10m(classid: np.ndarray, src_crs: Any, src_t: Affine) -> tuple:
    """Reproject a 3 m class-id raster to 10 m (mode) in the SAME UTM CRS.

    Returns (dst_uint8, (col0, row0)) where the dst grid is snapped to the 10 m UTM grid and
    pixel bounds are integer multiples under Projection(crs, 10, -10). nodata=255 preserved.
    """
    H, W = classid.shape
    minx = src_t.c
    maxx = src_t.c + src_t.a * W
    maxy = src_t.f
    miny = src_t.f + src_t.e * H  # e is negative
    col0 = math.floor(minx / TARGET_RES)
    col1 = math.ceil(maxx / TARGET_RES)
    row0 = math.floor(maxy / -TARGET_RES)
    row1 = math.ceil(miny / -TARGET_RES)
    dw, dh = col1 - col0, row1 - row0
    dst_t = Affine(TARGET_RES, 0, col0 * TARGET_RES, 0, -TARGET_RES, row0 * -TARGET_RES)
    dst = np.full((dh, dw), io.CLASS_NODATA, dtype=np.uint8)
    reproject(
        classid,
        dst,
        src_transform=src_t,
        src_crs=src_crs,
        dst_transform=dst_t,
        dst_crs=src_crs,
        src_nodata=io.CLASS_NODATA,
        dst_nodata=io.CLASS_NODATA,
        resampling=Resampling.mode,
    )
    return dst, (col0, row0)


def _scan_member(member: str) -> list[dict[str, Any]]:
    """Read one AOI-month label, reproject to 10 m, cut into <=64x64 tiles -> tile records."""
    try:
        data = _ZIP.read(member)  # type: ignore[union-attr]
        with rasterio.open(BytesIO(data)) as ds:
            arr = ds.read()
            src_crs = ds.crs
            src_t = ds.transform
    except Exception as e:  # noqa: BLE001
        print(f"WARN read failed {member}: {e}")
        return []
    if src_crs is None:
        print(f"WARN no CRS {member}")
        return []
    classid = _one_hot_to_classid(arr)
    dst, (col0, row0) = _reproject_to_10m(classid, src_crs, src_t)
    crs_str = src_crs.to_string()
    # member: labels/<AOI>/[Labels/]Raster/<sub>/<sub>-YYYY[-_]MM[-_]01.tif
    parts = member.split("/")
    aoi = parts[1]
    m = _DATE_RE.search(parts[-1])
    if m is None:
        print(f"WARN no date in {member}")
        return []
    year = int(m.group(1))
    ym = f"{m.group(1)}-{m.group(2)}"
    dh, dw = dst.shape
    recs: list[dict[str, Any]] = []
    for tr in range(0, dh, TILE):
        for tc in range(0, dw, TILE):
            sub = dst[tr : tr + TILE, tc : tc + TILE]
            present = sorted(int(v) for v in np.unique(sub) if v != io.CLASS_NODATA)
            if not present:
                continue
            x_min = col0 + tc
            y_min = row0 + tr
            bounds = (x_min, y_min, x_min + sub.shape[1], y_min + sub.shape[0])
            recs.append(
                {
                    "array": np.ascontiguousarray(sub),
                    "crs": crs_str,
                    "bounds": bounds,
                    "classes_present": present,
                    "year": year,
                    "source_id": f"{aoi}/{ym}/{tr}_{tc}",
                }
            )
    return recs


def _scan_all(workers: int) -> list[dict[str, Any]]:
    cache = io.raw_dir(SLUG) / "scan_cache.pkl"
    if cache.exists():
        print(f"loading cached scan from {cache}")
        with cache.open("rb") as f:
            return pickle.load(f)
    members = _list_label_members()
    print(f"scanning {len(members)} AOI-month labels (3 m -> 10 m mode, tile <=64)")
    recs: list[dict[str, Any]] = []
    with multiprocessing.Pool(workers, initializer=_worker_init) as p:
        for r in tqdm.tqdm(
            star_imap_unordered(p, _scan_member, [dict(member=m) for m in members]),
            total=len(members),
        ):
            recs.extend(r)
    print(f"scanned {len(recs)} candidate tiles")
    io.raw_dir(SLUG).mkdir(parents=True, exist_ok=True)
    tmp = cache.parent / "scan_cache.pkl.tmp"
    with tmp.open("wb") as f:
        pickle.dump(recs, f)
    tmp.rename(cache)
    return recs


def _write_one(rec: dict[str, Any]) -> None:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return
    proj = Projection(rasterio.crs.CRS.from_string(rec["crs"]), TARGET_RES, -TARGET_RES)
    bounds = tuple(rec["bounds"])
    io.write_label_geotiff(
        SLUG, sample_id, rec["array"], proj, bounds, nodata=io.CLASS_NODATA
    )
    # Each label is one specific calendar month's land-cover state, so use a ~3-month
    # window centered on the label month (not the full year the label year falls in).
    # Parse YYYY-MM from source_id ("{aoi}/{YYYY-MM}/{tr}_{tc}") so it works with any
    # cached scan record.
    year_str, month_str = rec["source_id"].split("/")[1].split("-")[:2]
    center = datetime(int(year_str), int(month_str), 15, tzinfo=UTC)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.centered_time_range(center, half_window_days=45),
        source_id=rec["source_id"],
        classes_present=rec["classes_present"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    if not _zip_path().exists():
        raise RuntimeError(
            f"{_zip_path()} missing. Download with:\n"
            f"  RSYNC_PASSWORD={RSYNC_PASSWORD} rsync -av {RSYNC_URL} {raw}/"
        )

    records = _scan_all(args.workers)
    selected = select_tiles_per_class(
        records, classes_key="classes_present", per_class=PER_CLASS
    )
    selected.sort(key=lambda r: r["source_id"])
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(f"selected {len(selected)} tiles (of {len(records)} scanned)")

    with multiprocessing.Pool(args.workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            pass

    tile_counts = {i: 0 for i in range(NUM_CLASSES)}
    for r in selected:
        for c in r["classes_present"]:
            tile_counts[c] += 1

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "mediaTUM node 1650201 (DynamicEarthNet, Toker et al. CVPR 2022)",
            "license": "CC-BY-SA-4.0",
            "provenance": {
                "url": "https://mediatum.ub.tum.de/1650201",
                "have_locally": False,
                "annotation_method": "manual pixel-wise land-cover annotation (monthly)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_tile_counts": {
                CLASSES[i][0]: tile_counts[i] for i in range(NUM_CLASSES)
            },
            "notes": (
                "Monthly 7-class land-cover masks over 55 global AOIs (2018-01..2019-12; "
                "1320 AOI-months = 55x24). Only labels.zip (1.4 GB) pulled from the mediaTUM "
                "rsync server; the ~500 GB Planet imagery was NOT downloaded (pretraining "
                "supplies its own imagery). Source 7-band one-hot (0/255) 3 m masks collapsed "
                "to single-band class ids (output id = band index), reprojected 3 m -> 10 m in "
                "their native UTM CRS with MODE resampling, and cut into <=64x64 tiles (~308x308 "
                "grid per AOI -> up to 25 tiles). Pixels with no active band -> nodata 255. "
                "All 7 classes kept, incl. snow & ice (band 6; present in 48/1320 AOI-months) "
                "which the official 6-class benchmark instead ignores. Each monthly label is the "
                "per-month land-cover STATE (not a dated change), so change_time=null and "
                "time_range is a ~3-month window centered on the label month. Tiles-per-class "
                "balanced to <=1000/class, rarest-first, <=25k total."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("class tile counts:")
    for i in range(NUM_CLASSES):
        print(f"  {i:>2} {CLASSES[i][0]:28} {tile_counts[i]}")
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()

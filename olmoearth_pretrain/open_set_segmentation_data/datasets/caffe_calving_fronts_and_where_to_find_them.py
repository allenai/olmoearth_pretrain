"""Process CaFFe (Calving Fronts and where to Find thEm) into open-set-segmentation tiles.

Source: CaFFe benchmark (Gourmelon et al. 2022, ESSD 14, 4287-4313), PANGAEA record
940950 (https://doi.pangaea.de/10.1594/PANGAEA.940950). 681 preprocessed, geocoded and
orthorectified SAR amplitude images of 7 marine-terminating glaciers (5 on the Antarctic
Peninsula, Jakobshavn/Greenland, Columbia/Alaska), each with two manually-annotated
expert labels: a multi-class ZONE segmentation and a binary CALVING FRONT line.

  - zone PNG grayscale values: 0 = N/A (SAR shadow/layover / no info), 64 = rock,
    127 = glacier, 254 = ocean + ice melange  (confirmed from the CaFFe repo
    data_postprocessing.py: model class 1->64, 2->127, 3->254).
  - front PNG: 0 background, 255 = calving-front line.

Georeferencing: the PANGAEA release ships plain grayscale PNGs with NO embedded geo tags
and pixel-coordinate bounding boxes only. The torchgeo/caffe HuggingFace mirror adds a
``meta_data.csv`` giving, for every image, its PROJECTED bounding box + CRS
(EPSG:3031 Antarctic polar stereographic for the 5 Peninsula glaciers, EPSG:32606 for
Columbia, EPSG:32622 for Jakobshavn). bbox_width / png_width matches the stated native
resolution exactly, so a north-up affine (origin = top-left, res = bbox/px) georeferences
every pixel. That table is what makes this dataset usable (otherwise: no recoverable
geocoordinates). We download meta_data.csv from the HF mirror and the label PNGs from
PANGAEA (data_raw.zip), join by image base name, and reproject each label into a local
UTM 10 m grid.

Unified class scheme (dense_raster zones + rasterized front line, per spec 5 "combine
multi-target sources into ONE class map"):
  0 = ocean_and_ice_melange, 1 = glacier, 2 = rock, 3 = calving_front, 255 = nodata.
The front line (dilated to ~3 px / ~30 m at 10 m so it survives resampling) is overlaid
on top of the zone segmentation.

Time filtering: the source spans 1995-2020; per spec we KEEP ONLY 2016+ images (Sentinel
era) so the labels can be co-located with S2/S1/Landsat. 52 of 681 images are >= 2016
(Columbia, Jorum, Mapple; Sentinel-1 + a few TanDEM-X). Each tile gets a 1-year window
centered on the acquisition date. Caveat: calving fronts shift seasonally, so the yearly
window is an approximation for the front-line class (noted in the summary).

Run (idempotent; skips already-written tiles):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.caffe_calving_fronts_and_where_to_find_them
"""

import argparse
import csv
import io as _io
import multiprocessing
import re
import zipfile
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import tqdm
from affine import Affine
from PIL import Image
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.warp import Resampling, reproject, transform_bounds
from rslearn.utils.geometry import Projection
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from rslearn.utils.mp import star_imap_unordered
from scipy.ndimage import binary_dilation

from olmoearth_pretrain.open_set_segmentation_data import download, io, sampling

SLUG = "caffe_calving_fronts_and_where_to_find_them"
NAME = "CaFFe (Calving Fronts and where to Find thEm)"

PANGAEA_ZIP = "https://download.pangaea.de/dataset/940950/files/data_raw.zip"
HF_META = "https://huggingface.co/datasets/torchgeo/caffe/resolve/main/meta_data.csv"

YEAR_MIN = 2016  # Sentinel era; source spans 1995-2020, keep only >= 2016.
TILE = io.MAX_TILE  # 64 -> 640 m tiles at 10 m.
MIN_VALID_PIXELS = 64  # drop tiles with < this many observed (non-nodata) pixels.
FRONT_DILATE_ITERS = 1  # 3x3 dilation at 10 m -> front line ~3 px (~30 m) wide.
PER_CLASS = 1000

# Unified output classes.
CID_OCEAN, CID_GLACIER, CID_ROCK, CID_FRONT = 0, 1, 2, 3
CLASSES = [
    {
        "id": CID_OCEAN,
        "name": "ocean_and_ice_melange",
        "description": "Open ocean and ice melange (fjord water plus the mixture of "
        "sea ice and calved icebergs in front of the glacier). CaFFe zone value 254.",
    },
    {
        "id": CID_GLACIER,
        "name": "glacier",
        "description": "Glacier ice / the glacier body. CaFFe zone value 127.",
    },
    {
        "id": CID_ROCK,
        "name": "rock",
        "description": "Bare rock outcrops / land surrounding the glacier. CaFFe zone "
        "value 64.",
    },
    {
        "id": CID_FRONT,
        "name": "calving_front",
        "description": "The marine-terminating glacier calving front (ice-ocean "
        "boundary) on the acquisition date. Manually digitized line, dilated to ~30 m "
        "so it is visible at 10 m/pixel. Overlaid on top of the zone segmentation.",
    },
]

# CaFFe zone grayscale value -> unified class id (0/N-A handled as nodata separately).
ZONE_TO_CID = {254: CID_OCEAN, 127: CID_GLACIER, 64: CID_ROCK}


# --------------------------------------------------------------------------------------
# Metadata table + PNG access.
# --------------------------------------------------------------------------------------
def _base_name(image_name: str) -> str:
    """CSV image_name 'COL_..._geo.tif' -> PANGAEA base 'COL_...'."""
    return re.sub(r"_geo\.tif$", "", image_name)


def _parse_bbox(s: str) -> tuple[float, float, float, float]:
    m = re.search(
        r"left=([-\d.]+), bottom=([-\d.]+), right=([-\d.]+), top=([-\d.]+)", s
    )
    assert m, f"unparseable bbox: {s!r}"
    left, bottom, right, top = (float(x) for x in m.groups())
    return left, bottom, right, top


def load_meta(csv_path: str) -> dict[str, dict[str, Any]]:
    """Parse meta_data.csv -> {base_name: {crs, bbox, date, glacier, sensor, year}}."""
    out: dict[str, dict[str, Any]] = {}
    with open(csv_path, encoding="latin-1") as f:
        rd = csv.reader(f, delimiter=";")
        header = [h.strip().strip('"').strip() for h in next(rd)]
        for row in rd:
            if not row or not row[0].strip():
                continue
            r = {
                header[i]: row[i].strip().strip('"').strip() for i in range(len(header))
            }
            base = _base_name(r["image_name"])
            day, month, year = (int(x) for x in r["date"].split("."))
            out[base] = {
                "crs": r["Coordinate system"],
                "bbox": _parse_bbox(r["Bounding box coordinates"]),
                "date": datetime(year, month, day, tzinfo=UTC),
                "year": year,
                "glacier": r["glacier_name"],
                "sensor": r["sensor"],
                "resolution": r["resolution (m)"],
            }
    return out


def _read_png(zf: zipfile.ZipFile, index: dict[str, str], key: str) -> np.ndarray:
    return np.array(Image.open(_io.BytesIO(zf.read(index[key]))))


# --------------------------------------------------------------------------------------
# Per-image reprojection + tiling (worker).
# --------------------------------------------------------------------------------------
def process_image(
    base: str, meta: dict[str, Any], zip_path: str
) -> list[dict[str, Any]]:
    """Reproject one image's zone+front labels to local UTM 10 m and split into tiles.

    Returns candidate tile records (each carries its uint8 array, out CRS, absolute
    pixel bounds, classes_present, date, source base name).
    """
    zf = zipfile.ZipFile(zip_path)
    # Locate this image's zone/front PNGs inside the archive.
    names = zf.namelist()
    zkey = next(
        n for n in names if "/zones/" in n and n.split("/")[-1] == base + "_zones.png"
    )
    fkey = next(
        n for n in names if "/fronts/" in n and n.split("/")[-1] == base + "_front.png"
    )
    zone_png = np.array(Image.open(_io.BytesIO(zf.read(zkey))))
    front_png = np.array(Image.open(_io.BytesIO(zf.read(fkey)))) == 255

    H, W = zone_png.shape
    left, bottom, right, top = meta["bbox"]
    src_crs = CRS.from_string(meta["crs"])
    xres = (right - left) / W
    yres = (top - bottom) / H
    src_tf = Affine(xres, 0, left, 0, -yres, top)

    # Source labels remapped to unified class ids (nodata 255 for N/A zone value 0).
    src_cls = np.full((H, W), io.CLASS_NODATA, dtype=np.uint8)
    for val, cid in ZONE_TO_CID.items():
        src_cls[zone_png == val] = cid
    src_front = front_png.astype(np.uint8)

    # Output local-UTM projection from image center.
    cx, cy = (left + right) / 2, (bottom + top) / 2
    tr = Transformer.from_crs(src_crs.to_epsg(), 4326, always_xy=True)
    clon, clat = tr.transform(cx, cy)
    out_proj = get_utm_ups_projection(clon, clat, io.RESOLUTION, -io.RESOLUTION)
    out_crs = out_proj.crs

    # Footprint of the image in output-CRS metres -> rslearn 10 m pixel grid.
    xmin, ymin, xmax, ymax = transform_bounds(
        src_crs, out_crs, left, bottom, right, top, densify_pts=21
    )
    col_min = int(np.floor(xmin / io.RESOLUTION))
    col_max = int(np.ceil(xmax / io.RESOLUTION))
    # rslearn pixel row = -y / res (y_res is negative); north (max y) -> smallest row.
    row_min = int(np.floor(-ymax / io.RESOLUTION))
    row_max = int(np.ceil(-ymin / io.RESOLUTION))
    Wo, Ho = col_max - col_min, row_max - row_min
    if Wo <= 0 or Ho <= 0:
        return []
    dst_tf = Affine(
        io.RESOLUTION,
        0,
        io.RESOLUTION * col_min,
        0,
        -io.RESOLUTION,
        -io.RESOLUTION * row_min,
    )

    dz = np.full((Ho, Wo), io.CLASS_NODATA, dtype=np.uint8)
    df = np.zeros((Ho, Wo), dtype=np.uint8)
    reproject(
        src_cls,
        dz,
        src_transform=src_tf,
        src_crs=src_crs,
        dst_transform=dst_tf,
        dst_crs=out_crs,
        resampling=Resampling.nearest,
        src_nodata=io.CLASS_NODATA,
        dst_nodata=io.CLASS_NODATA,
    )
    reproject(
        src_front,
        df,
        src_transform=src_tf,
        src_crs=src_crs,
        dst_transform=dst_tf,
        dst_crs=out_crs,
        resampling=Resampling.nearest,
        src_nodata=0,
        dst_nodata=0,
    )
    # Overlay the dilated front line as its own class (only where observed).
    if df.any():
        df = binary_dilation(df.astype(bool), iterations=FRONT_DILATE_ITERS)
        dz[df & (dz != io.CLASS_NODATA)] = CID_FRONT

    date_iso = meta["date"].isoformat()
    records: list[dict[str, Any]] = []
    for r0 in range(0, Ho, TILE):
        th = min(TILE, Ho - r0)
        for c0 in range(0, Wo, TILE):
            tw = min(TILE, Wo - c0)
            tile = dz[r0 : r0 + th, c0 : c0 + tw]
            present = [int(v) for v in np.unique(tile) if v != io.CLASS_NODATA]
            if not present:
                continue
            if int((tile != io.CLASS_NODATA).sum()) < MIN_VALID_PIXELS:
                continue
            abs_c = col_min + c0
            abs_r = row_min + r0
            records.append(
                {
                    "base": base,
                    "crs": out_crs.to_string(),
                    "bounds": (abs_c, abs_r, abs_c + tw, abs_r + th),
                    "classes_present": present,
                    "array": tile.copy(),
                    "date": date_iso,
                }
            )
    return records


# --------------------------------------------------------------------------------------
# Writer (worker).
# --------------------------------------------------------------------------------------
def _write_tile(rec: dict[str, Any]) -> str:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return "skip"
    proj = Projection(CRS.from_string(rec["crs"]), io.RESOLUTION, -io.RESOLUTION)
    bounds = tuple(rec["bounds"])
    io.write_label_geotiff(
        SLUG, sample_id, rec["array"], proj, bounds, nodata=io.CLASS_NODATA
    )
    date = datetime.fromisoformat(rec["date"])
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        (date - timedelta(days=180), date + timedelta(days=180)),
        source_id=rec["base"],
        classes_present=rec["classes_present"],
    )
    return "write"


# --------------------------------------------------------------------------------------
# Main.
# --------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)

    zip_path = (raw / "data_raw.zip").path
    if not (raw / "data_raw.zip").exists():
        print("downloading data_raw.zip from PANGAEA ...")
        download.download_http(PANGAEA_ZIP, raw / "data_raw.zip")
    csv_path = (raw / "meta_data.csv").path
    if not (raw / "meta_data.csv").exists():
        print("downloading meta_data.csv from torchgeo/caffe HF mirror ...")
        download.download_http(HF_META, raw / "meta_data.csv")
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "CaFFe (Gourmelon et al. 2022, ESSD 14:4287-4313), PANGAEA 940950.\n"
            f"  labels/images: {PANGAEA_ZIP}\n"
            f"  georef metadata table (projected bbox + CRS per image): {HF_META}\n"
            "PNGs joined to meta_data.csv by image base name; north-up affine from the\n"
            "projected bbox georeferences every pixel. Zones: 0=N/A,64=rock,127=glacier,\n"
            "254=ocean+melange. Fronts: 255=calving-front line.\n"
        )

    meta = load_meta(csv_path)
    subset = {b: m for b, m in meta.items() if m["year"] >= YEAR_MIN}
    print(f"{len(meta)} images total; {len(subset)} with year >= {YEAR_MIN}")

    io.check_disk()

    # Reproject + tile every kept image (52 images -> pool).
    tasks = [dict(base=b, meta=m, zip_path=zip_path) for b, m in sorted(subset.items())]
    all_records: list[dict[str, Any]] = []
    with multiprocessing.Pool(min(args.workers, len(tasks))) as p:
        for recs in tqdm.tqdm(
            star_imap_unordered(p, process_image, tasks), total=len(tasks)
        ):
            all_records.extend(recs)
    # Deterministic candidate ordering (pool returns are unordered) so the seeded
    # balancing below -- and hence the whole selection and sample-id assignment -- is
    # reproducible across runs (idempotent).
    all_records.sort(key=lambda r: (r["base"], r["bounds"][1], r["bounds"][0]))
    print(f"candidate tiles: {len(all_records)}")
    cand_counts = Counter()
    for r in all_records:
        for c in r["classes_present"]:
            cand_counts[c] += 1
    print("candidate tiles per class:", dict(sorted(cand_counts.items())))

    # Tiles-per-class balanced selection (prioritize rare classes: front, rock).
    selected = sampling.balance_tiles_by_class(
        all_records, "classes_present", per_class=PER_CLASS
    )
    # Deterministic ordering for stable/idempotent sample ids.
    selected.sort(key=lambda r: (r["base"], r["bounds"][1], r["bounds"][0]))
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(f"selected tiles: {len(selected)}")

    io.check_disk()

    results: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_tile, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            results[res] += 1
    print("write results:", dict(results))

    sel_counts = Counter()
    for r in selected:
        for c in r["classes_present"]:
            sel_counts[c] += 1
    glacier_hist = Counter(subset[r["base"]]["glacier"] for r in selected)
    year_hist = Counter(subset[r["base"]]["year"] for r in selected)

    num_samples = sum(1 for _ in io.locations_dir(SLUG).glob("*.tif"))
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "PANGAEA / ESSD (torchgeo HF mirror for geo metadata)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://doi.pangaea.de/10.1594/PANGAEA.940950",
                "have_locally": False,
                "annotation_method": "manual (expert) zone + calving-front labels",
                "citation": "Gourmelon et al. 2022, ESSD 14, 4287-4313 (CaFFe).",
                "georef_metadata": HF_META,
            },
            "sensors_relevant": ["sentinel1", "sentinel2", "landsat"],
            "classes": CLASSES,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": num_samples,
            "class_tile_counts": {
                CLASSES[c]["name"]: sel_counts.get(c, 0) for c in range(len(CLASSES))
            },
            "glacier_tile_counts": dict(glacier_hist),
            "year_tile_counts": dict(sorted(year_hist.items())),
            "notes": (
                "Multi-mission SAR calving-front benchmark. Zone segmentation "
                "(dense_raster) + binary calving-front line combined into ONE class map: "
                "0=ocean+ice_melange, 1=glacier, 2=rock, 3=calving_front, 255=nodata "
                "(CaFFe N/A zone + unobserved). Labels are grayscale PNGs georeferenced "
                "via meta_data.csv (projected bbox + CRS per image; EPSG:3031 Antarctic "
                "polar-stereo for Peninsula glaciers, EPSG:32606/32622 UTM for Columbia/"
                "Jakobshavn), reprojected to local UTM 10 m (nearest) and tiled into "
                "<=64x64 patches. Calving-front line dilated to ~3 px (~30 m) and overlaid "
                "on the zones. KEPT ONLY year >= 2016 (Sentinel era): 52 of 681 images "
                "(Columbia, Jorum, Mapple; mostly Sentinel-1 20 m + a few TanDEM-X 7 m); "
                "629 pre-2016 images dropped per spec. Tiles-per-class balanced at "
                f"{PER_CLASS}/class, prioritizing rare classes (calving_front, rock). "
                "1-year time window centered on the acquisition date. CAVEAT: calving "
                "fronts and near-front zones shift seasonally, so the yearly window is an "
                "approximation for the front-line class; the glacier/rock/ocean zones are "
                "more temporally stable. Jakobshavn (EPSG:32622) has no >=2016 images so "
                "does not appear in the kept subset."
            ),
        },
    )
    print(f"done: {num_samples} samples")
    print(
        "selected tiles per class:",
        {CLASSES[c]["name"]: sel_counts.get(c, 0) for c in range(len(CLASSES))},
    )
    print("glaciers:", dict(glacier_hist))


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()

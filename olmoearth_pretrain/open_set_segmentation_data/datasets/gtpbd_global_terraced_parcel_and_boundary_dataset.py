"""Process GTPBD (Global Terraced Parcel and Boundary Dataset) into open-set-segmentation patches.

Source: GTPBD, "A Fine-Grained Global Terraced Parcel and Boundary Dataset" (Zhang et al.,
NeurIPS 2025; arXiv 2507.14697). Distributed on Hugging Face (``wxqzzw/GTD``, public,
non-gated) as a single ``GTPBD_enhenced_png.zip`` (16 GB, CC-BY-NC-4.0). The archive ships
47,537 manually annotated 512x512 tiles cropped from high-resolution optical scenes (GF-2 +
Google Earth, 0.1-1 m GSD) over 7 Chinese agricultural zones + transcontinental regions,
with three co-registered binary label rasters per tile under ``mask_labels/``,
``boundary_labels/``, ``parcel_labels/`` (train/val/test x region). Each label PNG has a
GDAL ``.png.aux.xml`` PAM sidecar carrying its CRS + GeoTransform.

License: CC-BY-NC-4.0 (non-commercial). Recorded in metadata; acceptable for this research
pretraining use. Attribution: Zhang et al., GTPBD, NeurIPS 2025.

=== GEOREFERENCING (task spec 8 gate) — PARTIAL; only a verifiable subset is used ===
The per-tile ``.aux.xml`` GeoTransforms fall into two regimes, distinguished by pixel size:

  * Case-B (KEPT, ~6150 mask tiles): pixel size is a genuine small WGS84 degree value
    (~4.5e-6 - 7.2e-6 deg/px, i.e. ~0.44-0.8 m GSD). Verified internally consistent:
    within a scene, neighbouring tiles' origins differ by exactly (tile-pixel-offset x
    px_deg), so these tiles carry real, correct WGS84 georeferencing and can be placed on
    the S2 grid exactly. All Case-B tiles fall in Southwest + Central China (lon ~104-112,
    lat ~26.7-29.5).

  * Case-A (REJECTED, ~9994 mask tiles, incl. ALL "Rest of the world"/global tiles):
    pixel size is stored as 0.3 in a WGS84 (degrees) CRS. 0.3 deg = ~33 km/px, impossible
    for a 512-px VHR tile — this is a units bug: the ~0.3 m GSD was written into a degree
    CRS, and each sub-tile origin was computed as parent_origin + pixel_offset * 0.3 (deg),
    producing off-the-earth origins (lon 194, lat -277, ...). The parent-image origin is
    recoverable (subtract offset*0.3), but the TRUE per-image GSD (0.1-1 m per the paper,
    varying scene to scene) is NOT reliably recoverable: single-block parents give no scale
    information, and multi-block parents' block-offset naming has ambiguous pixel units. An
    assumed GSD would misplace/mis-scale the label on the S2 grid (errors up to ~km for
    corner tiles). Rather than emit misregistered labels, the Case-A subset is dropped.

Consequence/caveat: the processed subset is geographically narrower than the full GTPBD
(Southwest + Central China only), losing the global "Rest of the world" tiles.

=== CLASS SCHEME (task spec 5 multi-target -> one unified class map) ===
GTPBD provides three co-registered label types per tile: mask (terrace area, binary),
boundary (parcel ridge/edge, binary, thin), parcel (parcel interior, binary). At 10 m the
terrace PARCEL AREA (mask) is clearly resolvable (manifest note: terraced hillslopes visible
at 10-30 m), but the boundary/edge ridges are sub-metre (~1-3 native px, <0.3 px at 10 m) and
do not survive mode resampling; the parcel labels are instance-oriented and not a fixed
per-pixel class set. So the unified scheme uses the MASK layer as a per-pixel binary
segmentation:
    0 = background (non-terraced land within the scene)
    1 = terraced parcel
Boundary and parcel layers are NOT encoded (unresolvable / not per-pixel-class at 10 m);
noted in the summary.

VHR handling (task spec 4 VHR-native): each 512x512 binary mask (WGS84, ~0.5-0.8 m) is
reprojected to a local UTM grid at 10 m with MODE resampling (categorical majority; never
bilinear), yielding one ~23-36 px tile per source tile (all <= 64). Augmented tiles (``_flip``
/ ``_rot90`` in the filename) are geometric copies of originals and are excluded.

Time range: no per-tile acquisition date; terraces are static agricultural features (imagery
spans 2016-2025). Treated as static (task spec 5) with a representative 1-year Sentinel-era
window (2020).

Labels are read directly out of the local HF zip (only ``mask_labels/*.png`` + their
``.aux.xml`` are decoded). Scanned tile records are cached to ``raw/{slug}/scan_cache.pkl``.

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.gtpbd_global_terraced_parcel_and_boundary_dataset
"""

import argparse
import itertools
import math
import multiprocessing
import pickle
import re
import zipfile
from io import BytesIO
from typing import Any

import numpy as np
import rasterio
import tqdm
from affine import Affine
from PIL import Image
from pyproj import Transformer
from rasterio.warp import Resampling, reproject
from rslearn.utils.geometry import Projection
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    MAX_SAMPLES_PER_DATASET,
    select_tiles_per_class,
)

SLUG = "gtpbd_global_terraced_parcel_and_boundary_dataset"
NAME = "GTPBD (Global Terraced Parcel and Boundary Dataset)"
ZIP_NAME = "GTPBD_enhenced_png.zip"
TARGET_RES = 10.0
PER_CLASS = 1000
REPRESENTATIVE_YEAR = 2020
# Below this pixel-size (deg) a GeoTransform is genuine WGS84 degrees (Case-B, correct);
# at/above it the pixel size is the meters-as-degrees units bug (Case-A, rejected).
PX_DEG_MAX = 0.001

CLASSES = [
    (
        "background",
        "Non-terraced land within the terrace scene: forest, buildings, water, flat fields, "
        "bare ground. GTPBD mask_labels value 0.",
    ),
    (
        "terraced parcel",
        "Manually delineated agricultural terrace parcel on a hillslope (stepped/contoured "
        "cultivated field). GTPBD mask_labels value 1.",
    ),
]
NUM_CLASSES = len(CLASSES)


def _zip_path() -> Any:
    return io.raw_dir(SLUG) / ZIP_NAME


_ZIP: zipfile.ZipFile | None = None


def _worker_init() -> None:
    global _ZIP
    _ZIP = zipfile.ZipFile(str(_zip_path()), "r")


def _parse_geotransform(xml_bytes: bytes) -> tuple[float, float, float, float] | None:
    """Return (origin_lon, px, origin_lat, py) from a GDAL PAM .aux.xml, or None."""
    txt = xml_bytes.decode("utf-8", "replace")
    m = re.search(r"<GeoTransform>(.*?)</GeoTransform>", txt, re.S)
    if not m:
        return None
    v = [float(x) for x in m.group(1).split(",")]
    # GDAL GeoTransform: [origin_x, px_w, 0, origin_y, 0, px_h]
    return v[0], v[1], v[3], v[5]


def _reproject_mask(
    arr: np.ndarray, lon0: float, px: float, lat0: float, py: float, W: int, H: int
) -> tuple | None:
    """Reproject a binary WGS84 mask to local UTM 10 m (mode).

    Returns (out_uint8, utm_crs_str, (col0,row0,col1,row1)) or None if degenerate/too large.
    """
    src_crs = "EPSG:4326"
    src_t = Affine(px, 0, lon0, 0, py, lat0)
    cx = lon0 + px * W / 2.0
    cy = lat0 + py * H / 2.0
    lon, lat = cx, cy
    if not (
        np.isfinite(lon)
        and np.isfinite(lat)
        and -180 <= lon <= 180
        and -90 <= lat <= 90
    ):
        return None
    utm = get_utm_ups_projection(lon, lat, TARGET_RES, -TARGET_RES).crs
    to_utm = Transformer.from_crs(src_crs, utm, always_xy=True)
    xs = [lon0, lon0 + px * W]
    ys = [lat0, lat0 + py * H]
    pts = [to_utm.transform(X, Y) for X, Y in itertools.product(xs, ys)]
    if not all(np.isfinite(p[0]) and np.isfinite(p[1]) for p in pts):
        return None
    cols = [p[0] / TARGET_RES for p in pts]
    rows = [p[1] / -TARGET_RES for p in pts]
    col0, col1 = math.floor(min(cols)), math.ceil(max(cols))
    row0, row1 = math.floor(min(rows)), math.ceil(max(rows))
    dw, dh = col1 - col0, row1 - row0
    if dw <= 0 or dh <= 0 or dw > io.MAX_TILE or dh > io.MAX_TILE:
        return None
    dst_t = Affine(TARGET_RES, 0, col0 * TARGET_RES, 0, -TARGET_RES, row0 * -TARGET_RES)
    dst = np.zeros((dh, dw), dtype=np.uint8)
    reproject(
        arr.astype(np.uint8),
        dst,
        src_transform=src_t,
        src_crs=src_crs,
        dst_transform=dst_t,
        dst_crs=utm,
        resampling=Resampling.mode,
    )
    return dst, utm.to_string(), (col0, row0, col1, row1)


def _scan_member(member: str) -> dict[str, Any] | None:
    """Read one mask_labels/*.png + its .aux.xml; keep only Case-B (correct degrees)."""
    aux = member + ".aux.xml"
    try:
        xml = _ZIP.read(aux)  # type: ignore[union-attr]
    except KeyError:
        return None
    gt = _parse_geotransform(xml)
    if gt is None:
        return None
    lon0, px, lat0, py = gt
    if abs(px) >= PX_DEG_MAX:
        return None  # Case-A units-bug tile: unrecoverable GSD -> drop
    try:
        arr = np.array(Image.open(BytesIO(_ZIP.read(member))))  # type: ignore[union-attr]
    except Exception as e:  # noqa: BLE001
        print(f"WARN read failed {member}: {e}")
        return None
    if arr.ndim != 2:
        arr = arr[..., 0]
    arr = (arr > 0).astype(np.uint8)  # binarize: 1 = terrace, 0 = background
    H, W = arr.shape
    res = _reproject_mask(arr, lon0, px, lat0, py, W, H)
    if res is None:
        return None
    out, crs_str, bounds = res
    present = sorted(int(v) for v in np.unique(out))
    if 1 not in present:
        return None  # keep only tiles that actually contain terrace
    parts = member.split("/")
    region = parts[-2]
    split = parts[-3]
    tile = parts[-1][: -len(".png")]
    return {
        "array": out,
        "crs": crs_str,
        "bounds": bounds,
        "classes_present": present,
        "source_id": f"{split}/{region}/{tile}",
    }


def _list_mask_members() -> list[str]:
    with zipfile.ZipFile(str(_zip_path()), "r") as z:
        return sorted(
            n
            for n in z.namelist()
            if "/mask_labels/" in n
            and n.endswith(".png")
            and "_flip" not in n
            and "_rot" not in n
        )


def _scan_all(workers: int) -> list[dict[str, Any]]:
    cache = io.raw_dir(SLUG) / "scan_cache.pkl"
    if cache.exists():
        print(f"loading cached scan from {cache}")
        with cache.open("rb") as f:
            return pickle.load(f)
    members = _list_mask_members()
    print(
        f"scanning {len(members)} original mask tiles (Case-B filter + reproject to 10 m UTM)"
    )
    recs: list[dict[str, Any]] = []
    with multiprocessing.Pool(workers, initializer=_worker_init) as p:
        for r in tqdm.tqdm(
            star_imap_unordered(p, _scan_member, [dict(member=m) for m in members]),
            total=len(members),
        ):
            if r is not None:
                recs.append(r)
    print(f"kept {len(recs)} georeferenced terrace tiles (of {len(members)} originals)")
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
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(REPRESENTATIVE_YEAR),
        source_id=rec["source_id"],
        classes_present=rec["classes_present"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    if not _zip_path().exists():
        raise SystemExit(
            f"missing {_zip_path()}; download GTPBD_enhenced_png.zip from HF wxqzzw/GTD first"
        )

    records = _scan_all(args.workers)
    selected = select_tiles_per_class(
        records,
        classes_key="classes_present",
        per_class=PER_CLASS,
        total_cap=MAX_SAMPLES_PER_DATASET,
    )
    selected.sort(key=lambda r: r["source_id"])
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(f"selected {len(selected)} tiles (of {len(records)} scanned)")

    io.check_disk()
    with multiprocessing.Pool(args.workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            pass

    tile_counts = {i: 0 for i in range(NUM_CLASSES)}
    for r in selected:
        for c in r["classes_present"]:
            if c in tile_counts:
                tile_counts[c] += 1

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Hugging Face wxqzzw/GTD (GTPBD, Zhang et al., NeurIPS 2025, arXiv 2507.14697)",
            "license": "CC-BY-NC-4.0",
            "provenance": {
                "url": "https://huggingface.co/datasets/wxqzzw/GTD",
                "have_locally": False,
                "annotation_method": "manual (50+ annotators) delineation of terrace parcels on 0.1-1 m GF-2 / Google Earth imagery",
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
                "Binary terrace segmentation (background / terraced parcel) from GTPBD "
                "mask_labels, reprojected from WGS84 to local UTM at 10 m with MODE "
                "resampling (one ~23-36 px tile per 512x512 source tile). ONLY the subset of "
                "tiles with correct WGS84 degree-based geotransforms (~0.44-0.8 m GSD, "
                "Southwest + Central China) is used: the remaining ~9994 tiles (incl. all "
                "'Rest of the world'/global tiles) store the GSD (~0.3 m) as 0.3 DEGREES in a "
                "WGS84 CRS (units bug) with an unrecoverable true per-image GSD (0.1-1 m), so "
                "they cannot be placed accurately on the S2 grid and were dropped. Boundary "
                "and parcel label layers not encoded (sub-metre ridges unresolvable at 10 m; "
                "parcel labels are instance-oriented). Augmented (_flip/_rot90) tiles "
                "excluded. No per-tile date; terraces treated as static -> representative "
                "1-year window (2020). All source splits used. Tiles-per-class balanced to "
                "<=1000/class, <=25k total."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("class tile counts:")
    for i in range(NUM_CLASSES):
        print(f"  {i} {CLASSES[i][0]:20} {tile_counts[i]}")
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()

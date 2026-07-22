"""Process OSCD (Onera Satellite Change Detection) into open-set-segmentation labels.

Source: Onera Satellite Change Detection dataset (Caye Daudt, Le Saux, Boulch, Gousseau;
IGARSS 2018), https://rcdaudt.github.io/oscd/ . 24 registered bitemporal Sentinel-2 image
pairs over cities worldwide (14 "train" + 10 "test" split; we use ALL 24 per spec 5) with
**pixel-level binary urban-change** ground truth (manual photointerpretation). Images are
2015-2018 Sentinel-2 acquisitions (entirely in the Sentinel era).

Access (no credentials): images from the IMT/rcdaudt mirror
(https://partage.imt.fr/index.php/s/gKRaWgRnLMfwMGo), train labels from
https://partage.mines-telecom.fr/index.php/s/2D6n03k58ygBSpu , test labels from the
HuggingFace mirror hkristen/oscd (the rcdaudt test-labels mirror is dead / 404).

GEOREFERENCING (spec 8.2 check): the widely-used `imgs_*_rect` band TIFs are stripped of
their CRS (torchgeo treats OSCD as a NonGeoDataset), but the ORIGINAL `imgs_1/<S2>_Bxx.tif`
band crops **retain georeferencing** (EPSG:4326, geotransform matching each city's true
lon/lat). Each city's change map (`cm/<city>-cm.tif`) is on the SAME pixel grid as the
`imgs_1` 10 m bands (B02/B03/B04/B08) -- verified: cm dims == imgs_1 B04 dims for all 24
cities. We therefore read the CRS+transform from `imgs_1/*_B04.tif`, attach it to the change
map, and reproject the label to local UTM at 10 m (nearest, categorical), then tile.

Class scheme (dense per-pixel CLASSIFICATION, matching the manifest's 2 classes):
    id 0 = no-change   (cm.tif == 1)
    id 1 = change      (cm.tif == 2 ; urban change / new construction)
    255  = nodata/ignore  (geometric fill outside the rotated source footprint after
                           reprojection to UTM)

Processing (label_type = dense_raster): reproject each city's binary change map from its
EPSG:4326 imgs_1 grid to local UTM 10 m (nearest); cut into non-overlapping full 64x64
tiles (partial edge tiles dropped); drop tiles that are > 50% nodata. Sampling is
**tiles-per-class balanced** (spec 5): a tile counts toward `change` if it has >= MIN_CHANGE
change px and toward `no-change` if it has >= MIN_NOCHANGE no-change px; the rarer class
(`change`) is filled first, up to PER_CLASS tiles/class.

Time range (CHANGE label, spec 5, pre/post scheme): OSCD gives bitemporal pairs with two
acquisition dates date_1 (earlier) and date_2 (later), ~1-2.7 years apart, with urban change
occurring between them. Each sample carries two independent six-month windows (each <= 183
days) and `time_range` = null: `pre_time_range` = a ~6-month window centered on date_1 and
`post_time_range` = a ~6-month window centered on date_2; `change_time` = the midpoint (a
reference only). The two windows are naturally far apart -- exactly what the pre/post scheme
is for -- so the coarse multi-year change interval falls between them. This dataset was
previously rejected on change-timing grounds (the event not resolvable to within ~1-2
months); the pre/post scheme resolves that, so it is now usable.

Run:  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.oscd_onera_satellite_change_detection
"""

import argparse
import glob
import math
import multiprocessing
import warnings
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

import numpy as np
import tqdm
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from rasterio.warp import transform as warp_transform
from rslearn.utils.geometry import Projection
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, manifest, sampling

warnings.filterwarnings("ignore")  # rasterio NotGeoreferenced warnings on cm.tif

SLUG = "oscd_onera_satellite_change_detection"
NAME = "OSCD (Onera Satellite Change Detection)"

RAW = io.raw_dir(SLUG)
IMG_ROOT = RAW / "Onera Satellite Change Detection dataset - Images"
TRAIN_ROOT = RAW / "Onera Satellite Change Detection dataset - Train Labels"
TEST_ROOT = RAW / "Onera Satellite Change Detection dataset - Test Labels"

TILE = 64
PER_CLASS = 1000
MIN_CHANGE = 4  # >= this many change px for a tile to count as change (~400 m^2)
MIN_NOCHANGE = 64  # >= this many no-change px for a tile to count as no-change
MAX_NODATA_FRAC = 0.5

NO_CHANGE, CHANGE = 0, 1
CLASSES = [
    (
        "no-change",
        "No urban change between the two Sentinel-2 acquisitions at this pixel "
        "(OSCD change map value 1); the background class.",
    ),
    (
        "change",
        "Urban change (new construction / urban growth) mapped by manual "
        "photointerpretation between the two Sentinel-2 acquisitions (OSCD change map "
        "value 2).",
    ),
]


def _cm_path(city: str) -> str | None:
    for base in (TRAIN_ROOT, TEST_ROOT):
        p = base / city / "cm" / f"{city}-cm.tif"
        if p.exists():
            return p.path
    return None


def _split_of(city: str) -> str:
    return "train" if (TRAIN_ROOT / city / "cm" / f"{city}-cm.tif").exists() else "test"


def _cities() -> list[str]:
    """Cities that have both imagery (imgs_1) and a change-map label."""
    out = []
    for d in sorted(IMG_ROOT.iterdir()):
        if not d.is_dir():
            continue
        city = d.name
        if _cm_path(city) and glob.glob(f"{(d / 'imgs_1').path}/*B04.tif"):
            out.append(city)
    return out


def _change_time(
    city: str,
) -> tuple[datetime, tuple[datetime, datetime], tuple[datetime, datetime]]:
    """(change_time, pre_range, post_range) from dates.txt.

    OSCD pairs are two acquisitions ``date_1`` (earlier) and ``date_2`` (later), often
    1-2.7 years apart, with the urban change occurring between them. We give each its own
    ~6-month window (season-blind, centered on the acquisition): ``pre_range`` around
    ``date_1``, ``post_range`` around ``date_2``. ``change_time`` = the midpoint, kept for
    reference only.
    """
    txt = (IMG_ROOT / city / "dates.txt").read_text()
    vals = {}
    for line in txt.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            vals[k.strip()] = v.strip()
    d1 = datetime.strptime(vals["date_1"], "%Y%m%d").replace(tzinfo=UTC)
    d2 = datetime.strptime(vals["date_2"], "%Y%m%d").replace(tzinfo=UTC)
    mid = d1 + (d2 - d1) / 2
    pre_range = io.centered_time_range(d1, 91)
    post_range = io.centered_time_range(d2, 91)
    return mid, pre_range, post_range


def _reproject_city(city: str) -> tuple[np.ndarray, Projection, int, int]:
    """Reproject a city's binary change map to local UTM 10 m.

    Returns (dst_label[H,W] uint8 in {0,1,255}, utm_projection, col_off, row_off) where
    col_off/row_off are the integer 10 m pixel offsets of the dst raster's top-left in the
    UTM projection (so tile (ti,tj) has pixel bounds col_off+tj*64 .., row_off+ti*64 ..).
    """
    import rasterio

    cm = _cm_path(city)
    band = glob.glob(f"{(IMG_ROOT / city / 'imgs_1').path}/*B04.tif")[0]
    with rasterio.open(cm) as ds:
        lab = ds.read(1)
    with rasterio.open(band) as ds:
        src_crs, src_tr = ds.crs, ds.transform
        H, W = ds.height, ds.width
    # remap: cm 1 -> no-change (0), cm 2 -> change (1)
    src = np.where(lab == 2, CHANGE, NO_CHANGE).astype(np.uint8)

    cx = src_tr.c + src_tr.a * W / 2
    cy = src_tr.f + src_tr.e * H / 2
    utm = get_utm_ups_projection(cx, cy, io.RESOLUTION, -io.RESOLUTION)
    dst_crs = utm.crs

    # UTM bbox of the source footprint (4 corners), snapped to the 10 m grid.
    xs = [src_tr.c, src_tr.c + src_tr.a * W]
    ys = [src_tr.f, src_tr.f + src_tr.e * H]
    ux, uy = warp_transform(
        src_crs, dst_crs, [xs[0], xs[1], xs[0], xs[1]], [ys[0], ys[0], ys[1], ys[1]]
    )
    r = io.RESOLUTION
    x0 = math.floor(min(ux) / r) * r
    x1 = math.ceil(max(ux) / r) * r
    y0 = math.floor(min(uy) / r) * r
    y1 = math.ceil(max(uy) / r) * r
    dst_w = int((x1 - x0) / r)
    dst_h = int((y1 - y0) / r)
    dst_tr = from_origin(x0, y1, r, r)
    dst = np.full((dst_h, dst_w), io.CLASS_NODATA, dtype=np.uint8)
    reproject(
        source=src,
        destination=dst,
        src_transform=src_tr,
        src_crs=src_crs,
        dst_transform=dst_tr,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
        src_nodata=None,
        dst_nodata=io.CLASS_NODATA,
    )
    col_off = int(round(x0 / r))
    row_off = int(round(-y1 / r))
    return dst, utm, col_off, row_off


def _tile_classes(sub: np.ndarray) -> list[int] | None:
    """Class ids present in a 64x64 tile, or None to skip (too much nodata / empty)."""
    nod = int((sub == io.CLASS_NODATA).sum())
    if nod > MAX_NODATA_FRAC * TILE * TILE:
        return None
    classes = []
    if int((sub == CHANGE).sum()) >= MIN_CHANGE:
        classes.append(CHANGE)
    if int((sub == NO_CHANGE).sum()) >= MIN_NOCHANGE:
        classes.append(NO_CHANGE)
    return classes or None


def _scan_city(city: str) -> list[dict[str, Any]]:
    dst, _proj, _c, _r = _reproject_city(city)
    dst_h, dst_w = dst.shape
    recs = []
    for ti in range(dst_h // TILE):
        for tj in range(dst_w // TILE):
            sub = dst[ti * TILE : (ti + 1) * TILE, tj * TILE : (tj + 1) * TILE]
            classes = _tile_classes(sub)
            if classes is None:
                continue
            recs.append({"city": city, "ti": ti, "tj": tj, "classes_present": classes})
    return recs


def _write_city(city: str, tiles: list[dict[str, Any]]) -> None:
    dst, proj, col_off, row_off = _reproject_city(city)
    change_time, pre_range, post_range = _change_time(city)
    split = _split_of(city)
    for t in tiles:
        sid = t["sample_id"]
        if (io.locations_dir(SLUG) / f"{sid}.tif").exists():
            continue
        ti, tj = t["ti"], t["tj"]
        sub = dst[ti * TILE : (ti + 1) * TILE, tj * TILE : (tj + 1) * TILE]
        bounds = (
            col_off + tj * TILE,
            row_off + ti * TILE,
            col_off + (tj + 1) * TILE,
            row_off + (ti + 1) * TILE,
        )
        io.write_label_geotiff(SLUG, sid, sub, proj, bounds, nodata=io.CLASS_NODATA)
        present = sorted(int(v) for v in np.unique(sub) if v != io.CLASS_NODATA)
        io.write_sample_json(
            SLUG,
            sid,
            proj,
            bounds,
            None,
            change_time=change_time,
            source_id=f"{split}/{city}_r{ti}_c{tj}",
            classes_present=present,
            pre_time_range=pre_range,
            post_time_range=post_range,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    assert IMG_ROOT.exists(), f"missing {IMG_ROOT}; download+extract raw first"
    cities = _cities()
    print(f"{len(cities)} OSCD cities with imagery + labels")

    print("Scanning cities into 64x64 UTM tiles...")
    with multiprocessing.Pool(args.workers) as p:
        all_recs: list[dict[str, Any]] = []
        for recs in tqdm.tqdm(
            star_imap_unordered(p, _scan_city, [dict(city=c) for c in cities]),
            total=len(cities),
        ):
            all_recs.extend(recs)
    print(f"  {len(all_recs)} candidate tiles")

    selected = sampling.balance_tiles_by_class(
        all_recs, classes_key="classes_present", per_class=PER_CLASS
    )
    selected.sort(
        key=lambda r: (r["city"], r["ti"], r["tj"])
    )  # stable ids (idempotent)
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(
        f"  selected {len(selected)} tiles (tiles-per-class balanced, <= {PER_CLASS}/class)"
    )

    by_city: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in selected:
        by_city[r["city"]].append(r)

    io.check_disk()
    print(f"Writing tiles for {len(by_city)} cities...")
    with multiprocessing.Pool(args.workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(
                p,
                _write_city,
                [dict(city=c, tiles=ts) for c, ts in by_city.items()],
            ),
            total=len(by_city),
        ):
            pass

    tile_class_counts = {name: 0 for name, _ in CLASSES}
    for r in selected:
        for c in r["classes_present"]:
            tile_class_counts[CLASSES[c][0]] += 1
    print("tiles containing each class:", tile_class_counts)

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "OSCD (Onera Satellite Change Detection), IGARSS 2018",
            "license": "Change maps: CC-BY-NC-SA 4.0 (research/non-commercial). "
            "Imagery: modified Copernicus Sentinel data 2015-2018.",
            "provenance": {
                "url": "https://rcdaudt.github.io/oscd/",
                "have_locally": False,
                "annotation_method": "manual photointerpretation of Sentinel-2 image pairs",
                "citation": "Caye Daudt, Le Saux, Boulch, Gousseau, "
                "'Urban Change Detection for Multispectral Earth Observation Using "
                "Convolutional Neural Networks', IGARSS 2018",
                "access": "images: IMT mirror; train labels: mines-telecom mirror; "
                "test labels: HuggingFace hkristen/oscd (rcdaudt test mirror is 404)",
            },
            "sensors_relevant": ["sentinel2"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "tile_class_counts": tile_class_counts,
            "notes": (
                "24 registered bitemporal Sentinel-2 pairs (all train+test cities used) "
                "with binary urban-change masks. Georeferencing recovered from the ORIGINAL "
                "imgs_1/<S2>_B04.tif crops (EPSG:4326); the widely-used imgs_*_rect band "
                "TIFs are CRS-stripped. Each change map shares the imgs_1 10 m grid; "
                "reprojected to local UTM 10 m (nearest) and cut into 64x64 tiles. Classes: "
                "0 no-change, 1 change, 255 nodata (geometric fill after reprojection). "
                "Tiles-per-class balanced (<=1000/class), change filled first. CHANGE label: "
                "change_time = midpoint of the pair's two acquisition dates, time_range = "
                "1-year window centered on it. OSCD pairs span ~1-2.7 years, so change is a "
                "diffuse multi-year urban-growth signal; the 1-year window is a representative "
                "anchor within the interval."
            ),
        },
    )

    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()

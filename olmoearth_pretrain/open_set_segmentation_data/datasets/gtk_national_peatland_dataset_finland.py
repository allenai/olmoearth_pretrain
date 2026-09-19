"""Process the GTK National Peatland Dataset (Finland) into open-set-segmentation patches.

Source: Geological Survey of Finland (GTK), "Peatland site types of Finland 1.0/2023"
(Finnish: "Suotyypit ja turvekankaat 1.0/2023"), published 2023-11-03, GTK Open Licence.
A national 10 m raster covering all Finnish peatlands (~9 Mha) where each pixel is a
machine-learning-modelled peatland *site type* (40 types), its drainage state
(undrained/ojittamaton vs drained/ojitettu), plus peatland land-use classes (cultivated
and abandoned organic-soil fields) and a companion peat-extraction-area product. Produced
in the MaaTi project from Sentinel-1/-2, MML lidar DEM derivatives and NFI field reference
data. Catalogued at https://hakku.gtk.fi (location id 229). Native CRS EPSG:3067
(ETRS-TM35FIN), 10 m.

ACCESS (no credential, no transaction). The Hakku file download for this product is
delivered only through an order/checkout flow that submits a customer identity, so we do
NOT use it. Instead we read the *identical* raster through GTK's open, anonymous OGC/ArcGIS
interface (the sanctioned machine-access channel; the product's own metadata advertises
open WMS/WFS access): the ArcGIS MapServer
``Rajapinnat/GTK_Maapera_WMS`` exposes the peatland raster as three colour-mapped raster
layers -- 90 = undrained (2xxx codes), 89 = drained/turvekangas (3xxx codes), 96 = peat
extraction areas (1101-1104). We ``export`` PNG tiles at native 10 m (nearest-neighbour;
verified to render a clean discrete colour set with no interpolation) and decode the
rendered colour back to the raw site-type code. The MapServer's rendered colormap does not
match its REST legend swatches, so the colour->code map was recovered empirically once via
the MapServer ``identify`` operation (cached in raw/{slug}/color_decoder.json). Layers 89
and 90 share the rendered colormap (colour = site type; the *layer* gives drainage), so we
only need three special colours plus "any other opaque colour = a peatland site type".

CLASS MAPPING (4 classes; spec's manifest scheme). Non-peat / not-modelled -> 255 ignore.
  0 undrained mire            <- layer 90 opaque, any site-type colour except the specials
  1 forestry-drained peatland <- layer 89 opaque, any turvekangas colour except the specials
                                 ("turvekangas" = a drained peatland forest-vegetation type)
  2 agricultural organic soil <- colours (101,101,101) Turvepelto=cultivated organic-soil
                                 field, and (213,213,213) Kytoheitto=abandoned peat field,
                                 in either layer 89 or 90
  3 peat production area       <- layer 96 opaque (any of the 4 extraction-cover colours)
  255 ignore                  <- (0,0,0) Negatiivinen/mineraalimaa (modelled mineral soil,
                                 i.e. non-peat) and all not-modelled / transparent pixels

SAMPLING (spec sections 4 & 5): national derived-product map -> bounded-tile sampling. A
grid of 20 km blocks over the Finnish peatland extent is exported once (cached as combined
class-id GeoTIFFs in EPSG:3067 under raw/{slug}/blocks/), blocks with negligible peat are
skipped, and 64x64 (640 m) windows with enough peat coverage are scanned. Windows are
selected tiles-per-class balanced (rarest class first, <= 1000/class, 25k cap) by the
classes present, then each is reprojected from EPSG:3067 10 m to a local UTM projection at
10 m with NEAREST resampling (categorical). Output tiles keep the true class of every pixel
(full multi-class segmentation), not just a dominant class.

TIME: quasi-static land classification (product v1.0/2023, trained on 2016-2023 EO). Static
1-year window on 2023, change_time=null (spec section 5). No pre-2016 labels.

task_type=classification, label_type=dense_raster.
Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.gtk_national_peatland_dataset_finland
Idempotent: cached block GeoTIFFs and existing locations/{id}.tif are skipped on re-run.
"""

import argparse
import io as _io
import multiprocessing
import urllib.parse
import urllib.request
from collections import Counter
from typing import Any

import numpy as np
import rasterio
import tqdm
from PIL import Image
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import from_bounds
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import get_transform_from_projection_and_bounds

from olmoearth_pretrain.open_set_segmentation_data import io
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    select_tiles_per_class,
)

SLUG = "gtk_national_peatland_dataset_finland"
NAME = "GTK National Peatland Dataset (Finland)"
URL = "https://hakku.gtk.fi/en/locations/search?location_id=229"
BASE = (
    "https://gtkdata.gtk.fi/arcgis/rest/services/Rajapinnat/GTK_Maapera_WMS/MapServer"
)
LAYER_UNDRAINED = 90
LAYER_DRAINED = 89
LAYER_PEATPROD = 96

YEAR = 2023
SRC_CRS = "EPSG:3067"
TILE = 64  # output UTM tile: 64 px @ 10 m = 640 m
BLK = 2000  # block export side in px @ 10 m = 20 km (<= MapServer max 2048)
GRID_STEP = 80000  # block-centre spacing (m) over the peat extent
PER_CLASS = 1000
MIN_CLASS_PIX = 25  # a class counts toward a window if it has >= this many px
MIN_PEAT_FRAC = 0.20  # a window must be >= 20% peat (classes 0-3) to be a candidate
BLOCK_MIN_PEAT_FRAC = 0.01  # skip a whole block if < 1% of it is peat
WIN_STEP = 64  # non-overlapping 64-px scan windows
PAD_M = 400.0  # native-metre pad so a reprojected UTM tile is fully covered
SEED = 42

# Peatland extent (EPSG:3067) from the product's storageLocations bounding box.
EXT_XMIN, EXT_YMIN, EXT_XMAX, EXT_YMAX = 20000, 6570000, 788000, 7818000

# Rendered-colour specials (shared by layers 89 & 90; see module docstring / color_decoder.json).
AGRI_COLORS = {(101, 101, 101), (213, 213, 213)}  # Turvepelto + Kytoheitto -> class 2
MINERAL_COLOR = (0, 0, 0)  # Negatiivinen (mineraalimaa) -> ignore

CLASSES = [
    (
        0,
        "undrained mire",
        "Pristine / undrained peatland (ojittamaton). Any of the ~36 undrained mire site "
        "types (spruce/pine mires, fens, bogs; codes 2011-2103) in the GTK MaaTi taxonomy.",
    ),
    (
        1,
        "forestry-drained peatland",
        "Forestry-drained peatland, i.e. a peatland-forest ('turvekangas') vegetation type "
        "resulting from ditching (ojitettu; codes 3011-3103). Drained peatland under forestry.",
    ),
    (
        2,
        "agricultural organic soil",
        "Cultivated organic-soil field on peat (Turvepelto, code 2120/3120) or abandoned peat "
        "field (Kytoheitto, code 2130/3130) -- agricultural land use on peat soils.",
    ),
    (
        3,
        "peat production area",
        "Active or abandoned peat-extraction area (turvetuotantoalue; peat-covered, vegetated, "
        "tree-covered or water-covered; codes 1101-1104).",
    ),
]

CLASS_NAME = {c: n for c, n, _ in CLASSES}


def blocks_dir():
    return io.raw_dir(SLUG) / "blocks"


def block_path(name: str):
    return blocks_dir() / f"{name}.tif"


def grid_centers() -> list[tuple[str, int, int]]:
    """Grid of block centres (name, cx, cy) over the peatland extent in EPSG:3067."""
    out = []
    half = BLK * 10 // 2
    x = EXT_XMIN + half
    while x < EXT_XMAX:
        y = EXT_YMIN + half
        while y < EXT_YMAX:
            out.append((f"x{x}_y{y}", x, y))
            y += GRID_STEP
        x += GRID_STEP
    return out


def _fetch(url: str, tries: int = 4) -> bytes:
    import time

    last = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001 - retry transient server errors
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"fetch failed {url[:90]}: {last}")


def _export(cx: int, cy: int, layer: int) -> np.ndarray:
    """Export one BLKxBLK PNG (RGBA) for a layer, block centred on (cx, cy) in EPSG:3067."""
    x0, y0, x1, y1 = cx - BLK * 5, cy - BLK * 5, cx + BLK * 5, cy + BLK * 5
    q = {
        "bbox": f"{x0},{y0},{x1},{y1}",
        "bboxSR": "3067",
        "imageSR": "3067",
        "size": f"{BLK},{BLK}",
        "format": "png32",
        "transparent": "true",
        "dpi": "96",
        "layers": f"show:{layer}",
        "f": "image",
    }
    url = BASE + "/export?" + urllib.parse.urlencode(q)
    return np.array(Image.open(_io.BytesIO(_fetch(url))).convert("RGBA"))


def _combine(undr: np.ndarray, drn: np.ndarray, peat: np.ndarray) -> np.ndarray:
    """Combine the three rendered layers into a single uint8 class-id raster.

    Priority: peat-production (3) > undrained (0/2) > drained (1/2). 255 = ignore.
    """
    h, w, _ = undr.shape
    out = np.full((h, w), io.CLASS_NODATA, np.uint8)

    def rgb(a):
        a = a.astype(np.int32)
        return a[:, :, 0] * 1_000_000 + a[:, :, 1] * 1000 + a[:, :, 2]

    agri_keys = {r * 1_000_000 + g * 1000 + b for (r, g, b) in AGRI_COLORS}
    min_key = MINERAL_COLOR[0] * 1_000_000 + MINERAL_COLOR[1] * 1000 + MINERAL_COLOR[2]

    # drained (layer 89): fill first so undrained/peatprod can override on overlaps
    d_op = drn[:, :, 3] > 200
    d_rgb = rgb(drn)
    d_agri = d_op & np.isin(d_rgb, list(agri_keys))
    d_min = d_op & (d_rgb == min_key)
    d_peat = d_op & ~d_agri & ~d_min
    out[d_peat] = 1
    out[d_agri] = 2
    # mineral stays ignore

    # undrained (layer 90) overrides
    u_op = undr[:, :, 3] > 200
    u_rgb = rgb(undr)
    u_agri = u_op & np.isin(u_rgb, list(agri_keys))
    u_min = u_op & (u_rgb == min_key)
    u_peat = u_op & ~u_agri & ~u_min
    out[u_peat] = 0
    out[u_agri] = 2
    out[u_min] = io.CLASS_NODATA

    # peat production (layer 96) overrides everything
    p_op = peat[:, :, 3] > 200
    out[p_op] = 3
    return out


def build_block(name: str, cx: int, cy: int) -> bool:
    """Export + combine one block, cache as a class-id GeoTIFF (EPSG:3067). Returns True if
    the block contains enough peat to keep.
    """
    dst = block_path(name)
    if dst.exists():
        with rasterio.open(dst.path) as ds:
            a = ds.read(1)
        return float((a != io.CLASS_NODATA).mean()) >= BLOCK_MIN_PEAT_FRAC
    undr = _export(cx, cy, LAYER_UNDRAINED)
    drn = _export(cx, cy, LAYER_DRAINED)
    peat = _export(cx, cy, LAYER_PEATPROD)
    cls = _combine(undr, drn, peat)
    keep = float((cls != io.CLASS_NODATA).mean()) >= BLOCK_MIN_PEAT_FRAC
    if not keep:
        return False
    x0, y1 = cx - BLK * 5, cy + BLK * 5
    transform = Affine(10, 0, x0, 0, -10, y1)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / (dst.name + ".tmp")
    with rasterio.open(
        tmp.path,
        "w",
        driver="GTiff",
        height=BLK,
        width=BLK,
        count=1,
        dtype="uint8",
        crs=SRC_CRS,
        transform=transform,
        nodata=io.CLASS_NODATA,
        compress="deflate",
    ) as ds:
        ds.write(cls, 1)
    tmp.rename(dst)
    return True


def _build_block_task(name: str, cx: int, cy: int) -> tuple[str, bool]:
    return name, build_block(name, cx, cy)


def scan_block(name: str) -> list[dict[str, Any]]:
    """Find candidate 64x64 windows in a cached block; record centre lon/lat + classes."""
    with rasterio.open(block_path(name).path) as ds:
        arr = ds.read(1)
        st = ds.transform
    from pyproj import Transformer

    tf = Transformer.from_crs(SRC_CRS, "EPSG:4326", always_xy=True)
    recs = []
    denom = float(TILE * TILE)
    for r0 in range(0, arr.shape[0] - TILE + 1, WIN_STEP):
        for c0 in range(0, arr.shape[1] - TILE + 1, WIN_STEP):
            win = arr[r0 : r0 + TILE, c0 : c0 + TILE]
            peat_frac = float((win != io.CLASS_NODATA).mean())
            if peat_frac < MIN_PEAT_FRAC:
                continue
            cnt = Counter(int(v) for v in win.ravel() if v != io.CLASS_NODATA)
            classes = sorted(c for c, n in cnt.items() if n >= MIN_CLASS_PIX)
            if not classes:
                continue
            cx = st.c + (c0 + TILE / 2.0) * st.a
            cy = st.f + (r0 + TILE / 2.0) * st.e
            lon, lat = tf.transform(cx, cy)
            recs.append(
                {
                    "block": name,
                    "lon": float(lon),
                    "lat": float(lat),
                    "classes_present": classes,
                    "source_id": f"{name}_r{r0}_c{c0}",
                }
            )
    return recs


def write_one(rec: dict[str, Any]) -> None:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return
    lon, lat = rec["lon"], rec["lat"]
    dst_proj, col, row = io.lonlat_to_utm_pixel(lon, lat)
    bounds = io.centered_bounds(col, row, TILE, TILE)
    dst_transform = get_transform_from_projection_and_bounds(dst_proj, bounds)

    xs = [bounds[0] * io.RESOLUTION, bounds[2] * io.RESOLUTION]
    ys = [bounds[1] * -io.RESOLUTION, bounds[3] * -io.RESOLUTION]
    left, right, bottom, top = min(xs), max(xs), min(ys), max(ys)
    l2, b2, r2, t2 = transform_bounds(dst_proj.crs, SRC_CRS, left, bottom, right, top)

    with rasterio.open(block_path(rec["block"]).path) as ds:
        w = from_bounds(l2 - PAD_M, b2 - PAD_M, r2 + PAD_M, t2 + PAD_M, ds.transform)
        src = ds.read(1, window=w, boundless=True, fill_value=io.CLASS_NODATA)
        win_transform = ds.window_transform(w)
        src_crs = ds.crs

    dst = np.full((TILE, TILE), io.CLASS_NODATA, np.uint8)
    reproject(
        source=src,
        destination=dst,
        src_transform=win_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_proj.crs,
        resampling=Resampling.nearest,
        src_nodata=io.CLASS_NODATA,
        dst_nodata=io.CLASS_NODATA,
    )
    io.write_label_geotiff(
        SLUG, sample_id, dst, dst_proj, bounds, nodata=io.CLASS_NODATA
    )
    present = sorted(int(v) for v in np.unique(dst) if v != io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        dst_proj,
        bounds,
        io.year_range(YEAR),
        change_time=None,
        source_id=rec["source_id"],
        classes_present=present,
    )


def _write_source_txt(n_blocks: int) -> None:
    d = io.raw_dir(SLUG)
    d.mkdir(parents=True, exist_ok=True)
    (d / "SOURCE.txt").write_text(
        "GTK 'Peatland site types of Finland 1.0/2023' (Suotyypit ja turvekankaat 1.0/2023).\n"
        "Geological Survey of Finland (GTK). GTK Open Licence. Native EPSG:3067, 10 m.\n"
        "Catalogue: https://hakku.gtk.fi (location id 229). productOid 1.2.246.563.1.127231.\n"
        "The Hakku file download requires an order/checkout that submits a customer identity,\n"
        "so it is NOT used. The identical raster is read anonymously through GTK's open ArcGIS\n"
        "MapServer Rajapinnat/GTK_Maapera_WMS (layers 90=undrained, 89=drained, 96=peat\n"
        "production) via export (PNG @10 m, nearest) + a one-time identify colour->code\n"
        "calibration (color_decoder.json). Blocks cached in blocks/ as class-id GeoTIFFs.\n"
        f"{n_blocks} peat-containing 20 km blocks were used for bounded-tile sampling.\n"
        "Code table: peatland_site_types_fertilitylevels.pdf.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--fetch-workers", type=int, default=12)
    args = parser.parse_args()

    io.check_disk()
    centers = grid_centers()
    print(f"grid: {len(centers)} candidate 20 km blocks over the peat extent")

    print("Exporting + combining blocks from the GTK ArcGIS MapServer...")
    kept = []
    with multiprocessing.Pool(args.fetch_workers) as p:
        for name, keep in tqdm.tqdm(
            star_imap_unordered(
                p,
                _build_block_task,
                [dict(name=n, cx=cx, cy=cy) for (n, cx, cy) in centers],
            ),
            total=len(centers),
        ):
            if keep:
                kept.append(name)
    print(f"{len(kept)} blocks contain peat")
    io.check_disk()
    _write_source_txt(len(kept))

    print("Scanning blocks for candidate 64x64 windows...")
    all_recs: list[dict[str, Any]] = []
    with multiprocessing.Pool(min(len(kept), 24) or 1) as p:
        for recs in tqdm.tqdm(
            star_imap_unordered(p, scan_block, [dict(name=n) for n in kept]),
            total=len(kept),
        ):
            all_recs.extend(recs)
    print(f"{len(all_recs)} candidate windows")
    cand = Counter()
    for r in all_recs:
        for c in r["classes_present"]:
            cand[c] += 1
    print(
        "candidate windows per class:",
        {CLASS_NAME[k]: cand.get(k, 0) for k, _, _ in CLASSES},
    )

    selected = select_tiles_per_class(all_recs, "classes_present", per_class=PER_CLASS)
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(f"selected {len(selected)} windows (tiles-per-class balanced, 25k cap)")

    io.check_disk()
    with multiprocessing.Pool(args.workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            pass

    # class counts over selected tiles (a tile counts toward every class present)
    sel_counts = Counter()
    for r in selected:
        for c in r["classes_present"]:
            sel_counts[c] += 1
    class_counts = {CLASS_NAME[k]: sel_counts.get(k, 0) for k, _, _ in CLASSES}
    print("selected tiles per class:", class_counts)

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Geological Survey of Finland (GTK)",
            "license": "GTK Open Licence (open, free reuse with attribution)",
            "provenance": {
                "url": URL,
                "have_locally": False,
                "annotation_method": (
                    "machine-learning modelling (MaaTi project) from Sentinel-1/-2, MML "
                    "lidar DEM derivatives and National Forest Inventory data, trained on "
                    "field-observed peatland site-type and land-use reference data"
                ),
                "accessed_via": (
                    "GTK open ArcGIS MapServer Rajapinnat/GTK_Maapera_WMS export (PNG @10 m) "
                    "+ identify colour->code calibration; layers 90/89/96. Hakku order "
                    "download not used (avoids submitting a customer identity)."
                ),
                "product": "Suotyypit ja turvekankaat 1.0/2023 (Peatland site types of Finland)",
                "product_oid": "1.2.246.563.1.127231",
                "native_crs": SRC_CRS,
                "native_resolution_m": 10,
                "year": YEAR,
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [{"id": i, "name": n, "description": d} for i, n, d in CLASSES],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": class_counts,
            "notes": (
                "Bounded-tile sampling of the GTK national 10 m peatland raster (spec 4 & 5). "
                "The MapServer renders the raster with a colormap that differs from its REST "
                "legend; the colour->code map was recovered via one-time identify calls "
                "(raw/color_decoder.json). Layers 89 (drained/turvekangas) and 90 (undrained) "
                "share the rendered colormap, so drainage is taken from the layer and the "
                "colour from the site type; 3 special colours give agricultural organic soil "
                "(Turvepelto (101,101,101) + Kytoheitto (213,213,213)) and mineral soil "
                "((0,0,0) -> ignore); layer 96 gives peat-production areas. 20 km blocks were "
                "exported at native 10 m in EPSG:3067 (nearest; verified clean discrete "
                "colours), combined to a class-id raster, scanned for 64x64 windows with "
                ">=20% peat coverage, selected tiles-per-class balanced (<=1000/class), and "
                "reprojected to local UTM at 10 m with NEAREST resampling. Output tiles keep "
                "the true class of every pixel. Static 2023 label: change_time=null, 1-year "
                "window on 2023. Non-peat/mineral/not-modelled pixels are 255 (ignore)."
            ),
        },
    )
    print(f"num_samples={len(selected)} task_type=classification")
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()

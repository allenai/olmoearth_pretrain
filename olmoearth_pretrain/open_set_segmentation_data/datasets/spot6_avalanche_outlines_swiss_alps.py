"""Process SPOT6-mapped snow-avalanche outlines (Swiss Alps, 24 January 2018) into
avalanche-presence segmentation label tiles.

Source: EnviDat / WSL-SLF "SPOT6 Avalanche outlines 24 January 2018" (Hafner &
Buehler 2019, doi:10.16904/envidat.77). 18,737 avalanche outlines were *manually*
mapped (photointerpretation) from a single SPOT6 satellite acquisition on
**24 January 2018**, documenting an extreme avalanche period (danger level 5) over
the Swiss Alps (Buehler et al. 2019, The Cryosphere 13, 3225). License: Open Database
License (ODbL) with Database Contents License (DbCL) -- free to use with attribution
(see summary / metadata provenance).

Each shapefile feature is one avalanche-outline POLYGON (source CRS EPSG:2056,
CH1903+/LV95) with per-avalanche attributes: TYP (SLAB / LOOSE_SNOW / FULL_DEPTH /
UNKNOWN), AVAL_SHAPE (outline quality: 1=exact, 2=estimated, 3=created), sze (size
class), aspect, start_zone/dpo_alt (altitudes). These attributes are per-avalanche and
NOT reliably observable per-pixel from 10-30 m S2/S1/Landsat imagery, so they are kept
as provenance metadata only, not as label classes.

Task: **single-class avalanche-presence segmentation** (label_type: polygons).
This is a positive-only foreground dataset (avalanche outlines were mapped where
avalanches occurred; the absence of a polygon is not a verified "no avalanche"), so
per spec §5 we do NOT fabricate negatives:

    0   = avalanche  (inside a mapped avalanche outline)
    255 = nodata/ignore (everything else)

The assembly step supplies negatives from other datasets.

Change semantics: the avalanches all released during the 22-24 January 2018 storm
cycle and were mapped from the 24 Jan 2018 SPOT6 image -- the event date is known to
within days. Avalanche debris is snow, visible for weeks after the event but gone by
the following summer, so a static full-year presence label would be misleading (summer
2018 imagery shows no debris). We therefore treat each sample as a dated CHANGE label
(spec §5): ``change_time`` = 2018-01-24 for every sample, and ``time_range`` is a
360-day window CENTERED on it (+/-180 d). Pretraining only uses a sample when the
sampled input window spans ``change_time``, so it always sees the late-Jan-2018 debris
period (undisturbed snowpack before -> avalanche-debris texture after).

Tiling (mirrors cal_fire_frap): each outline is reprojected to a local UTM projection
at 10 m/pixel. An avalanche whose footprint fits in a 64x64 tile (640 m) yields one
centered tile; larger avalanches are gridded into non-overlapping 64x64 windows and up
to MAX_TILES_PER_AVAL intersecting windows are sampled. Inside outline -> 0, everything
else -> 255 (nodata). Selection is round-robin across avalanches (every avalanche
contributes >=1 tile before large ones add more), capped at 25,000 tiles.

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.spot6_avalanche_outlines_swiss_alps
Idempotent: existing locations/{id}.tif are skipped.
"""

import argparse
import math
import multiprocessing
import random
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import shapely
import shapely.geometry
import shapely.ops
import shapely.wkb
import tqdm
from pyproj import Transformer
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, sampling

SLUG = "spot6_avalanche_outlines_swiss_alps"
NAME = "SPOT6 Avalanche Outlines (Swiss Alps)"

URL = "https://doi.org/10.16904/envidat.77"
DOWNLOAD_URL = (
    "https://www.envidat.ch/dataset/fa4adf13-d0e5-4479-9b46-cbb07233999f/resource/"
    "309d5260-f13e-4fd1-881e-8968c829941b/download/aval_outlines2018.zip"
)
SHP = io.raw_dir(SLUG) / "extracted" / "outlines2018.shp"
SRC_EPSG = "EPSG:2056"  # CH1903+ / LV95 (Swiss)

TILE = 64
MAX_TILES_PER_AVAL = 20
MAX_SAMPLES = sampling.MAX_SAMPLES_PER_DATASET  # 25000

# All avalanches mapped from the 24 Jan 2018 SPOT6 image; the storm cycle was 22-24 Jan.
CHANGE_TIME = datetime(2018, 1, 24, tzinfo=UTC)
HALF_WINDOW = timedelta(
    days=180
)  # +/-180 d => 360-day (<=1 year) window centered on event

AVAL, NODATA = 0, io.CLASS_NODATA
CLASSES = [
    (
        "avalanche",
        "Interior of a manually mapped snow-avalanche outline (SLF/WSL "
        "photointerpretation of a 24 Jan 2018 SPOT6 image documenting an extreme "
        "avalanche period, danger level 5, over the Swiss Alps). Covers the full "
        "avalanche extent (release + track + deposit) as delineated. Positive-only: "
        "everything outside a mapped outline is nodata (255), not a verified negative.",
    ),
]

# Per-avalanche attribute domains (recorded as provenance metadata only; NOT per-pixel
# classes -- avalanche type / quality / size are not observable per-pixel at 10-30 m).
TYP_CODES = {
    "SLAB": "Slab avalanche (distinct fracture line; slab releases over a large area).",
    "LOOSE_SNOW": "Loose-snow avalanche (point release fanning out downslope).",
    "FULL_DEPTH": "Full-depth / gliding avalanche (whole snowpack slides on the ground).",
    "UNKNOWN": "Type not identifiable (only deposit visible or ambiguous).",
}
AVAL_SHAPE_CODES = {
    1: "exact (outline clearly and entirely visible)",
    2: "estimated (mostly visible; gaps connected considering terrain)",
    3: "created (only release or deposit visible; rest inferred, or cut off at image edge)",
}


# --------------------------------------------------------------------------- download


def download() -> None:
    from olmoearth_pretrain.open_set_segmentation_data import download as dl

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "SPOT6 Avalanche outlines 24 January 2018 (Hafner & Buehler 2019).\n"
            "EnviDat / WSL-SLF, doi:10.16904/envidat.77.\n"
            f"Landing page: {URL}\n"
            f"Download: {DOWNLOAD_URL}\n"
            "License: ODbL with Database Contents License (DbCL). Attribution required.\n"
            "Contents: outlines2018.shp (18,737 avalanche outline polygons, EPSG:2056) "
            "+ ExampleKey_AvalMapping.pdf (attribute key).\n"
        )
    zip_path = raw / "aval_outlines2018.zip"
    if not SHP.exists():
        io.check_disk()
        dl.download_http(
            DOWNLOAD_URL, zip_path, headers={"User-Agent": "Mozilla/5.0"}, timeout=600
        )
        dl.extract_zip(zip_path, raw / "extracted")
    print(f"  shapefile: {SHP} (exists={SHP.exists()})")


# --------------------------------------------------------------------------- tiling


def _aval_candidates(aval: dict[str, Any]) -> list[dict[str, Any]]:
    """Return candidate tile records for one avalanche (bounds + clipped pixel geom WKB).

    ``aval['wkb']`` is the outline geometry already reprojected to WGS84 lon/lat.
    """
    from shapely.geometry import box
    from shapely.prepared import prep

    from olmoearth_pretrain.open_set_segmentation_data.rasterize import geom_to_pixels

    geom = shapely.wkb.loads(aval["wkb"])
    if geom.is_empty:
        return []
    c = geom.centroid
    proj = io.utm_projection_for_lonlat(float(c.x), float(c.y))
    px = geom_to_pixels(geom, WGS84_PROJECTION, proj)
    if px.is_empty or px.area <= 0:
        return []
    minx, miny, maxx, maxy = px.bounds
    w, h = maxx - minx, maxy - miny
    crs = proj.crs.to_string()

    base = {"crs": crs, "source_id": aval["source_id"]}
    out: list[dict[str, Any]] = []

    if w <= TILE and h <= TILE:
        col = round((minx + maxx) / 2.0)
        row = round((miny + maxy) / 2.0)
        b = io.centered_bounds(col, row, TILE, TILE)
        clip = px.intersection(box(*b))
        if not clip.is_empty and clip.area > 0:
            out.append({**base, "bounds": b, "clip_wkb": shapely.wkb.dumps(clip)})
        return out

    # Large avalanche: grid the bbox into non-overlapping 64x64 windows; keep intersecting.
    x0, y0 = math.floor(minx), math.floor(miny)
    cells = []
    x = x0
    while x < maxx:
        y = y0
        while y < maxy:
            cells.append((x, y, x + TILE, y + TILE))
            y += TILE
        x += TILE
    rng = random.Random(aval["idx"])
    rng.shuffle(cells)
    prepared = prep(px)
    for b in cells:
        if len(out) >= MAX_TILES_PER_AVAL:
            break
        bx = box(*b)
        if not prepared.intersects(bx):
            continue
        clip = px.intersection(bx)
        if clip.is_empty or clip.area <= 0:
            continue
        out.append({**base, "bounds": b, "clip_wkb": shapely.wkb.dumps(clip)})
    return out


def _write_one(rec: dict[str, Any]) -> str | None:
    from rasterio.crs import CRS

    from olmoearth_pretrain.open_set_segmentation_data.rasterize import rasterize_shapes

    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return None

    proj = Projection(CRS.from_string(rec["crs"]), io.RESOLUTION, -io.RESOLUTION)
    bounds = tuple(rec["bounds"])
    clip = shapely.wkb.loads(rec["clip_wkb"])
    # Positive-only: avalanche interior -> 0, everything else -> 255 (nodata).
    label = rasterize_shapes(
        [(clip, AVAL)], bounds, fill=NODATA, dtype="uint8", all_touched=True
    )[0]

    time_range = (CHANGE_TIME - HALF_WINDOW, CHANGE_TIME + HALF_WINDOW)
    present = sorted(int(v) for v in np.unique(label) if int(v) != NODATA)
    io.write_label_geotiff(SLUG, sample_id, label, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        time_range,
        change_time=CHANGE_TIME,
        source_id=rec["source_id"],
        classes_present=present,
    )
    return "ok" if present else "empty"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    io.check_disk()
    download()
    io.check_disk()

    # ---- load outlines, reproject to WGS84
    import fiona

    transformer = Transformer.from_crs(SRC_EPSG, "EPSG:4326", always_xy=True)

    def to_wgs84(geom):
        return shapely.ops.transform(lambda xs, ys: transformer.transform(xs, ys), geom)

    avals: list[dict[str, Any]] = []
    typ_counts: Counter = Counter()
    qual_counts: Counter = Counter()
    with fiona.open(str(SHP)) as src:
        for i, feat in enumerate(src):
            if feat["geometry"] is None:
                continue
            geom = shapely.geometry.shape(feat["geometry"])
            if geom.is_empty:
                continue
            geom = to_wgs84(geom)
            if geom.is_empty:
                continue
            props = feat["properties"]
            oid = props.get("OBJECTID")
            typ = props.get("typ") or "UNKNOWN"
            qual = props.get("aval_shape")
            sze = props.get("sze")
            typ_counts[typ] += 1
            qual_counts[qual] += 1
            avals.append(
                {
                    "idx": i,
                    "wkb": shapely.wkb.dumps(geom),
                    "source_id": f"OBJECTID={oid}:typ={typ}:qual={qual}:sze={sze}",
                }
            )
    print(f"{len(avals)} avalanche outlines loaded")
    print("typ:", dict(typ_counts), "quality:", dict(qual_counts))

    # ---- Phase B: per-avalanche candidate tiles (parallel)
    io.check_disk()
    per_aval: list[list[dict[str, Any]]] = []
    with multiprocessing.Pool(args.workers) as p:
        for cands in tqdm.tqdm(
            star_imap_unordered(p, _aval_candidates, [dict(aval=a) for a in avals]),
            total=len(avals),
            desc="candidates",
        ):
            if cands:
                per_aval.append(cands)
    total_cand = sum(len(c) for c in per_aval)
    print(f"{total_cand} candidate tiles across {len(per_aval)} avalanches")

    # ---- Phase C: round-robin selection across avalanches, capped at MAX_SAMPLES
    rng = random.Random(42)
    for lst in per_aval:
        rng.shuffle(lst)
    rng.shuffle(per_aval)
    selected: list[dict[str, Any]] = []
    i = 0
    active = [lst for lst in per_aval if lst]
    while active and len(selected) < MAX_SAMPLES:
        lst = active[i % len(active)]
        selected.append(lst.pop())
        i += 1
        if i % len(active) == 0:
            active = [lst for lst in active if lst]
    for j, r in enumerate(selected):
        r["sample_id"] = f"{j:06d}"
    print(f"selected {len(selected)} tiles (cap {MAX_SAMPLES})")

    # ---- Phase D: write tiles (parallel)
    io.check_disk()
    counts: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
            desc="write tiles",
        ):
            if res is not None:
                counts[res] += 1

    n_written = len(list(io.locations_dir(SLUG).glob("*.tif")))
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "EnviDat (WSL/SLF)",
            "license": "ODbL with Database Contents License (DbCL)",
            "provenance": {
                "url": URL,
                "download_url": DOWNLOAD_URL,
                "have_locally": False,
                "annotation_method": "manual photointerpretation of SPOT6 imagery",
                "acquisition_date": "2018-01-24",
                "citation": (
                    "Hafner, E. & Buehler, Y. (2019). SPOT6 Avalanche outlines "
                    "24 January 2018. EnviDat. doi:10.16904/envidat.77."
                ),
                "attribution": (
                    "Data: WSL Institute for Snow and Avalanche Research SLF / EnviDat "
                    "(Hafner & Buehler 2019), ODbL/DbCL."
                ),
                "typ_codes": TYP_CODES,
                "aval_shape_codes": AVAL_SHAPE_CODES,
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": n_written,
            "is_change_dataset": True,
            "change_time": CHANGE_TIME.isoformat(),
            "notes": (
                "Single-class avalanche-presence segmentation from 18,737 manually "
                "mapped SPOT6 avalanche outlines (Swiss Alps, 24 Jan 2018). 64x64 uint8 "
                "tiles, local UTM at 10 m; class 0 = avalanche interior, 255 = nodata "
                "(positive-only foreground -- no fabricated negatives, per spec §5; "
                "assembly adds negatives from other datasets). Dated CHANGE label: "
                "change_time = 2018-01-24 (storm cycle 22-24 Jan; event known to within "
                "days), time_range = +/-180 d (360-day window) centered on it. Avalanche "
                "debris is snow (visible weeks, gone by summer), so a static year-long "
                "presence label would be misleading -- change framing pairs imagery with "
                "the debris period. Small avalanches -> 1 centered tile; large ones "
                f"gridded into non-overlapping 64x64 windows, up to {MAX_TILES_PER_AVAL} "
                "sampled per avalanche. Round-robin across avalanches (every avalanche "
                f">=1 tile) capped at {MAX_SAMPLES}. TYP/AVAL_SHAPE/sze are per-avalanche "
                "attributes not observable per-pixel, kept as provenance metadata only. "
                "all_touched=True rasterization so thin avalanche tracks stay visible at "
                "10 m."
            ),
        },
    )
    print("write results:", dict(counts))
    print("total tif on disk:", n_written)
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()

"""Process USGS HVO Kilauea episode 61g lava-flow shapefiles into lava-flow segmentation
label tiles.

Source: USGS Hawaiian Volcano Observatory data release "GIS shapefiles for Kilauea's
episode 61g lava flow, Puu Oo eruption: May 2016 to May 2017" (Orr et al., 2017,
https://doi.org/10.5066/F7DN43XR), ScienceBase item 597230e4e4b0ec1a4885edc1. Public
domain. Downloaded (no credentials) from the ScienceBase file endpoint as a single ~9.7 MB
zip of 28 shapefiles for 14 mapping dates (one per calendar month, May 2016 -> May 2017;
two dates in June 2016; May 3 2017 substitutes for a missing April 2017 map).

Each mapping date has two shapefiles, both in EPSG:32605 (WGS84 / UTM zone 5N, metres):
  * ``Ep61g_YYYYMMDD_flow.shp``     -- ONE polygon: the FULL (cumulative) extent of the
                                       episode-61g lava flow as of that date.
  * ``Ep61g_YYYYMMDD_contacts.shp`` -- polylines: mapped lava-flow contacts (flow margins),
                                       attribute LineType='contact', accuracy 10-25 m.

Task: **lava-flow presence/state segmentation** (label_type: polygons + lines) with a
unified 3-class scheme (spec §5 multi-modality -> one class map combining polygons + lines):

    0 = background     (outside the flow / kipuka islands of older ground within it)
    1 = lava_flow      (interior of the mapped episode-61g flow extent)
    2 = flow_contact   (mapped flow-margin contact polylines, buffered to ~30 m width so
                        they are resolvable at 10-30 m; burned ON TOP of the flow interior)

Why presence/state (not a per-increment change mask): a solidified basaltic lava flow is a
PERSISTENT surface -- fresh dark basalt stays highly discernible in Sentinel-2 / Landsat
SWIR for years after emplacement. Each date's polygon is the cumulative flow field, i.e. a
snapshot of the persistent fresh-flow surface present on that date. We keep all 14 dated
snapshots as separate temporal samples: the flow grows from ~4 ha (2016-05-24) to ~947 ha
(2017-05-31), so at a fixed location the label legitimately transitions background->lava
over the sequence, giving genuine multi-temporal signal.

Change timing (spec §5): the mapping dates are precise (single calendar dates, <= ~1 month
apart), well inside the "known to within ~1-2 months" requirement, so we DO set a
per-sample ``change_time`` = the mapping date and emit two adjacent six-month windows split
exactly at it (via ``io.pre_post_time_ranges``): ``pre_time_range`` = the ~6 months
(<=183 days) immediately before the date and ``post_time_range`` = the ~6 months (<=183 days)
immediately after, with ``time_range`` = null. This lets pretraining pair the "before" image
stack with the "after" stack and probe on their difference; because the fresh basalt
persists, the cumulative mask also remains a valid presence mask after the date. (Caveat:
the mask is the extent AS OF the date, so imagery in the post window may show the flow having
grown a little beyond the mask near the active toe -- a minor, conservative under-count noted
in the summary.)

Tiling: geometries are reprojected from EPSG:32605 metres into 10 m/pixel pixel space in
the SAME UTM zone (no CRS change; just a resolution scaling), then the flow's pixel bbox is
gridded into non-overlapping 64x64 windows. A window is kept if it intersects the flow
polygon or a contact. Per window: flow interior -> 1, contacts (buffered) -> 2 on top,
elsewhere -> 0 (255 nodata unused; the flow perimeter is authoritative so out-of-flow
pixels are genuine background, not ignore).

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.usgs_kilauea_lava_flow_shapefiles
Idempotent: existing locations/{id}.tif are skipped.
"""

import argparse
import glob
import math
import multiprocessing
import os
import zipfile
from collections import Counter
from datetime import UTC, datetime
from typing import Any

import numpy as np
import shapely
import shapely.wkb
import tqdm
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, sampling

SLUG = "usgs_kilauea_lava_flow_shapefiles"
NAME = "USGS Kilauea Lava Flow Shapefiles"

SB_ITEM = "597230e4e4b0ec1a4885edc1"
URL = "https://www.sciencebase.gov/catalog/item/" + SB_ITEM
DOI = "https://doi.org/10.5066/F7DN43XR"
ZIP_NAME = "shapefiles.zip"
SHAPE_SUBDIR = "PuuOo_Ep61g_20160524-20170531_Shapefiles"

SRC_EPSG = 32605  # WGS84 / UTM zone 5N (metres) -- native shapefile CRS
TILE = 64
CONTACT_BUFFER_PX = 1.5  # half-width in 10 m pixels => ~30 m wide contact ribbon
MAX_SAMPLES = sampling.MAX_SAMPLES_PER_DATASET  # 25000 (not reached; ~538 tiles total)

BG, LAVA, CONTACT = 0, 1, 2
CLASSES = [
    (
        "background",
        "Ground outside the mapped episode-61g flow (older lava/vegetation) or kipuka "
        "islands of older ground enclosed by the flow. The HVO flow perimeter is "
        "authoritative, so out-of-flow pixels are genuine non-lava context (no synthetic "
        "negatives added).",
    ),
    (
        "lava_flow",
        "Interior of the mapped cumulative extent of Kilauea's Puu Oo episode-61g "
        "basaltic lava flow as of the mapping date. Fresh basalt, highly discernible in "
        "Sentinel-2 / Landsat SWIR. Mapped by HVO via field GPS and orthophoto / "
        "satellite image digitizing.",
    ),
    (
        "flow_contact",
        "Mapped lava-flow contact (flow margin) polylines (attribute LineType='contact', "
        "positional accuracy 10-25 m), buffered to a ~30 m ribbon so they are resolvable "
        "at 10-30 m and burned on top of the flow interior.",
    ),
]


# --------------------------------------------------------------------------- download


def ensure_raw() -> str:
    """Ensure the shapefile zip is downloaded and extracted; return the shapefile dir.

    Public-domain USGS data with no credentials. Idempotent.
    """
    import urllib.request

    rd = io.raw_dir(SLUG)
    rd.mkdir(parents=True, exist_ok=True)
    zip_path = rd / ZIP_NAME
    shp_dir = rd / SHAPE_SUBDIR

    with (rd / "SOURCE.txt").open("w") as f:
        f.write(
            "USGS HVO data release (public domain): 'GIS shapefiles for Kilauea's "
            "episode 61g lava flow, Puu Oo eruption: May 2016 to May 2017'.\n"
            f"Landing page: {URL}\nDOI: {DOI}\n"
            f"Downloaded file: {ZIP_NAME} (ScienceBase item {SB_ITEM}), extracted to "
            f"{SHAPE_SUBDIR}/ (28 shapefiles, 14 mapping dates).\n"
        )

    if not zip_path.exists():
        io.check_disk()
        # ScienceBase resolves ?f=__disk__<hash> to the specific attached file; the plain
        # item file endpoint also serves the primary zip.
        dl = (
            "https://www.sciencebase.gov/catalog/file/get/"
            f"{SB_ITEM}?f=__disk__40%2F76%2F5e%2F40765e106400494dce021f0abe9ca2c92a46b216"
        )
        req = urllib.request.Request(dl, headers={"User-Agent": "Mozilla/5.0"})
        tmp = rd / (ZIP_NAME + ".tmp")
        with urllib.request.urlopen(req, timeout=300) as r, tmp.open("wb") as out:
            out.write(r.read())
        tmp.rename(zip_path)
        print(f"  downloaded {ZIP_NAME}")

    if not shp_dir.exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(rd)
        print(f"  extracted to {shp_dir}")

    return str(shp_dir)


# --------------------------------------------------------------------------- geometry


def _to_pixels(geom: Any) -> Any:
    """Reproject an EPSG:32605-metre geometry into 10 m/pixel pixel space (same UTM zone)."""
    src = Projection(CRS.from_epsg(SRC_EPSG), 1, 1)
    dst = Projection(CRS.from_epsg(SRC_EPSG), io.RESOLUTION, -io.RESOLUTION)
    return STGeometry(src, geom, None).to_projection(dst).shp


def _date_from_name(path: str) -> datetime:
    """Parse Ep61g_YYYYMMDD_flow.shp -> midday-UTC datetime for that mapping date."""
    stamp = os.path.basename(path).split("_")[1]
    return datetime(int(stamp[0:4]), int(stamp[4:6]), int(stamp[6:8]), 12, tzinfo=UTC)


def build_candidates(shp_dir: str) -> list[dict[str, Any]]:
    """Grid every mapping date's flow into 64x64 tiles; return per-tile records."""
    import geopandas as gpd
    from shapely.geometry import box
    from shapely.prepared import prep

    records: list[dict[str, Any]] = []
    flow_files = sorted(glob.glob(os.path.join(shp_dir, "*_flow.shp")))
    assert flow_files, f"no flow shapefiles in {shp_dir}"

    for fpath in flow_files:
        change_time = _date_from_name(fpath)
        stamp = os.path.basename(fpath).split("_")[1]
        flow = gpd.read_file(fpath).to_crs(SRC_EPSG).union_all()
        cpath = fpath.replace("_flow.shp", "_contacts.shp")
        contact_px = None
        if os.path.exists(cpath):
            cg = gpd.read_file(cpath).to_crs(SRC_EPSG)
            if len(cg):
                contact_line = cg.union_all()
                contact_px = _to_pixels(contact_line).buffer(CONTACT_BUFFER_PX)

        flow_px = _to_pixels(flow)
        if flow_px.is_empty or flow_px.area <= 0:
            continue
        minx, miny, maxx, maxy = flow_px.bounds
        x0, y0 = math.floor(minx), math.floor(miny)
        prepared = prep(flow_px)
        prepared_c = prep(contact_px) if contact_px is not None else None

        x = x0
        while x < maxx:
            y = y0
            while y < maxy:
                b = (x, y, x + TILE, y + TILE)
                bx = box(*b)
                hit_flow = prepared.intersects(bx)
                hit_c = prepared_c.intersects(bx) if prepared_c is not None else False
                if hit_flow or hit_c:
                    clip_f = flow_px.intersection(bx) if hit_flow else None
                    clip_c = contact_px.intersection(bx) if hit_c else None
                    records.append(
                        {
                            "bounds": b,
                            "change_ms": int(change_time.timestamp() * 1000),
                            "source_id": f"Ep61g_{stamp}",
                            "flow_wkb": (
                                shapely.wkb.dumps(clip_f)
                                if clip_f is not None and not clip_f.is_empty
                                else None
                            ),
                            "contact_wkb": (
                                shapely.wkb.dumps(clip_c)
                                if clip_c is not None and not clip_c.is_empty
                                else None
                            ),
                        }
                    )
                y += TILE
            x += TILE
    return records


# --------------------------------------------------------------------------- write


def _write_one(rec: dict[str, Any]) -> list[int] | None:
    from olmoearth_pretrain.open_set_segmentation_data.rasterize import rasterize_shapes

    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return None

    proj = Projection(CRS.from_epsg(SRC_EPSG), io.RESOLUTION, -io.RESOLUTION)
    bounds = tuple(rec["bounds"])
    shapes: list[tuple[Any, int]] = []
    if rec["flow_wkb"] is not None:
        shapes.append((shapely.wkb.loads(rec["flow_wkb"]), LAVA))
    if (
        rec["contact_wkb"] is not None
    ):  # burned after flow -> contact wins on the margin
        shapes.append((shapely.wkb.loads(rec["contact_wkb"]), CONTACT))
    if not shapes:
        return None

    label = rasterize_shapes(shapes, bounds, fill=BG, dtype="uint8", all_touched=False)[
        0
    ]
    present = sorted(int(v) for v in np.unique(label))

    change_time = datetime.fromtimestamp(rec["change_ms"] / 1000.0, tz=UTC)
    pre_range, post_range = io.pre_post_time_ranges(change_time)
    time_range = (pre_range[0], post_range[1])  # outer bounding span

    io.write_label_geotiff(SLUG, sample_id, label, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        time_range,
        change_time=change_time,
        source_id=rec["source_id"],
        classes_present=present,
        pre_time_range=pre_range,
        post_time_range=post_range,
    )
    return present


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    io.check_disk()
    shp_dir = ensure_raw()
    io.check_disk()

    records = build_candidates(shp_dir)
    if len(records) > MAX_SAMPLES:  # not expected (~538); guard the hard cap anyway
        import random

        random.Random(42).shuffle(records)
        records = records[:MAX_SAMPLES]
    for j, r in enumerate(records):
        r["sample_id"] = f"{j:06d}"
    print(f"{len(records)} candidate tiles across 14 mapping dates")

    io.check_disk()
    class_tiles: Counter = Counter()  # #tiles containing each class
    dates: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for present in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in records]),
            total=len(records),
            desc="write tiles",
        ):
            if present is not None:
                for c in present:
                    class_tiles[c] += 1
    for r in records:
        dates[r["source_id"]] += 1

    n_written = len(list(io.locations_dir(SLUG).glob("*.tif")))
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "USGS HVO",
            "license": "public domain",
            "provenance": {
                "url": URL,
                "doi": DOI,
                "sciencebase_item": SB_ITEM,
                "have_locally": False,
                "annotation_method": "manual (field GPS + aerial/satellite image digitizing)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": n_written,
            "tiles_per_class": {
                CLASSES[c][0]: class_tiles.get(c, 0) for c in (BG, LAVA, CONTACT)
            },
            "samples_per_mapping_date": dict(sorted(dates.items())),
            "is_change_dataset": True,
            "notes": (
                "Episode-61g (Puu Oo) lava-flow presence/state segmentation from USGS HVO "
                "shapefiles (Orr et al. 2017). 64x64 uint8 tiles, local UTM 5N "
                "(EPSG:32605) at 10 m. Classes: 0 background, 1 lava_flow (cumulative flow "
                "extent polygon), 2 flow_contact (contact polylines buffered to ~30 m, "
                "burned over the flow); 255 nodata unused. Each of 14 monthly mapping "
                "dates (May 2016 - May 2017) is kept as a separate temporal snapshot of "
                "the persistent fresh-basalt flow field; the flow grows ~4 -> ~947 ha over "
                "the sequence. change_time = mapping date (dates precise to <= ~1 month, "
                "satisfying the >=1-2-month timing rule); time_range = +/-180 d centered "
                "on it. Masks are the extent AS OF each date, so imagery late in a window "
                "may show the active toe grown slightly beyond the mask (minor conservative "
                "under-count). Contacts are all LineType='contact' (flow margins; this "
                "release contains no separate fissure lines)."
            ),
        },
    )
    print("tiles per class (id->#tiles):", dict(class_tiles))
    print("samples per mapping date:", dict(sorted(dates.items())))
    print("total tif on disk:", n_written)
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()

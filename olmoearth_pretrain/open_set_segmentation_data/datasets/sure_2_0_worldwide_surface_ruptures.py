"""SURE 2.0 (Worldwide Surface Ruptures) -> coseismic surface-rupture change segmentation.

Source: Nurminen, F. et al. "SURE 2.0 - New release of the worldwide database of surface
ruptures for fault displacement hazard analyses." Sci Data 9, 729 (2022),
https://doi.org/10.1038/s41597-022-01835-z. Data on Zenodo (record 7020265,
https://doi.org/10.5281/zenodo.7020265, CC-BY-4.0). Public, no credentials.

The database holds mapped coseismic surface-rupture traces (line shapefiles, one per event)
and slip observation points for **50 crustal earthquakes worldwide, spanning 1872-2019**.
Each earthquake ships as ``YYYYMMDD_EventName_SURE2.0_ruptures.shp`` (WGS84 LineStrings), so
every rupture trace carries a **day-precise event date** via its filename / IdE and the
SURE2.0_Earthquakes.xlsx table (Year/Month/Day, Mw, focal mechanism).

TRIAGE (spec sections 2/5/8). Surface ruptures are produced by a dated earthquake, so a
rupture trace is a genuine, date-resolvable CHANGE signal (before->after fault break /
scarp / deformation zone visible in imagery) -- but ONLY for earthquakes in the Sentinel
era. Of the 50 events, 42 are pre-2016 (32 in the 1900s, incl. Landsat-era-only ones); their
surface expression is decades-eroded AND they fail the pre-2016 change rule, so they are
DROPPED. The remaining 8 events are 2016+. Of those we additionally drop 2019 Le Teil
(Mw 4.9, ~cm offsets, rupture detected mainly by InSAR/field -> sub-pixel / not observable
at 10 m). The 7 KEPT events are all Mw >= 6.0 -- significant earthquakes whose rupture zones
(surface breaks, scarps, wide deformation belts) are plausibly observable at 10 m when the
line is buffered to a zone:

    20160415 Kumamoto     Mw 7.0 strike-slip  (Japan)          1145 traces
    20160520 Petermann    Mw 6.1 reverse      (Australia)       229 traces
    20160824 Amatrice     Mw 6.0 normal       (Italy)           120 traces
    20161030 Norcia       Mw 6.5 normal       (Italy)           732 traces
    20161201 Parina       Mw 6.2 normal       (Peru)             21 traces
    20190704 Ridgecrest1  Mw 6.4 strike-slip  (USA)            7074 traces
    20190705 Ridgecrest2  Mw 7.1 strike-slip  (USA)           10875 traces

All event dates are known to the day (<< the ~1-2 month change-timing requirement), so we
set per-sample ``change_time`` = the earthquake date and keep it as the reference for
building the windows. Instead of a single centered window, we emit two adjacent six-month
windows split at ``change_time``: ``pre_time_range`` (the <=183 days immediately before) and
``post_time_range`` (the <=183 days immediately after), with ``time_range`` set to null
(built via ``io.pre_post_time_ranges(change_time, ...)``), so pretraining pairs a "before"
image stack with an "after" stack and probes on their difference.

TASK: change (event) classification, ``label_type`` lines. Per spec section 4 (lines) we
rasterize the rupture traces to a mask with a small dilation so they are resolvable at 10 m:
we BUFFER each line to a ~30 m half-width (3 px @ 10 m -> ~60 m wide rupture zone), then tile
each event's rupture footprint into non-overlapping 64x64 UTM 10 m tiles. Unified 2-class
scheme:

    0 = background        (no mapped rupture; genuine non-rupture context within the tile)
    1 = surface_rupture   (buffered coseismic rupture zone: principal + all distributed
                           ranks; the SURE Comp_rank field is recorded but all ranks are
                           merged into one rupture class)

Tiling is in each event's local UTM zone (from the event centroid), at 10 m/pixel. Lines are
converted to pixel space (E/10, -N/10) and buffered; the buffered footprint's pixel bbox is
gridded into 64x64 tiles and a tile is kept if it intersects a buffered rupture.

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.sure_2_0_worldwide_surface_ruptures
Idempotent: existing locations/{id}.tif are skipped.
"""

import argparse
import glob
import math
import multiprocessing
import os
import warnings
import zipfile
from collections import Counter
from datetime import UTC, datetime
from typing import Any

import numpy as np
import shapely
import shapely.affinity
import shapely.wkb
import tqdm
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, sampling

warnings.filterwarnings("ignore")

SLUG = "sure_2_0_worldwide_surface_ruptures"
NAME = "SURE 2.0 (Worldwide Surface Ruptures)"

ZENODO_RECORD = "7020265"
URL = "https://doi.org/10.5281/zenodo.7020265"
PAPER = "https://doi.org/10.1038/s41597-022-01835-z"
RUPTURES_ZIP = "SURE2.0_Ruptures.zip"
EARTHQUAKES_XLSX = "SURE2.0_Earthquakes.xlsx"
RUPTURES_SUBDIR = "SURE2.0_Ruptures"

TILE = 64
BUF_PX = 3  # ~30 m half-width => ~60 m wide rupture zone at 10 m/pixel
MIN_MW = 5.5  # keep only significant events (drops 2019 Le Teil Mw 4.9)
MIN_YEAR = 2016  # Sentinel era (change-signal + pre-2016 rule)
MAX_SAMPLES = sampling.MAX_SAMPLES_PER_DATASET  # 25000 (not reached; ~1300 tiles)

BG, RUPTURE = 0, 1
CLASSES = [
    (
        "background",
        "No mapped coseismic surface rupture. Genuine non-rupture context within the tile "
        "(the earthquake's rupture footprint is authoritative for the mapped traces, so "
        "off-trace pixels are treated as background, not ignore). Distributed rupturing may "
        "be locally under-mapped for small events.",
    ),
    (
        "surface_rupture",
        "Coseismic surface-rupture zone from a significant (Mw >= 6) 2016+ earthquake: the "
        "mapped rupture trace (principal fault rank 1 plus all distributed ranks 1.5/2/3/"
        "21/22, merged) buffered to a ~30 m half-width (~60 m wide) zone so the surface "
        "break / fault scarp / deformation belt is resolvable at 10 m. Field-mapped (SURE "
        "2.0, manual field mapping + georeferenced maps + satellite/LiDAR).",
    ),
]


# --------------------------------------------------------------------------- download


def ensure_raw() -> tuple[str, str]:
    """Download the SURE 2.0 rupture shapefiles + earthquake table; return (shp_dir, xlsx)."""
    from olmoearth_pretrain.open_set_segmentation_data import download

    rd = io.raw_dir(SLUG)
    rd.mkdir(parents=True, exist_ok=True)
    with (rd / "SOURCE.txt").open("w") as f:
        f.write(
            "SURE 2.0 (Worldwide Surface Ruptures), Nurminen et al., Sci Data 9, 729 (2022).\n"
            f"Paper: {PAPER}\nData (CC-BY-4.0): {URL} (Zenodo record {ZENODO_RECORD}).\n"
            f"Downloaded: {RUPTURES_ZIP} (50 per-event rupture line shapefiles) and "
            f"{EARTHQUAKES_XLSX} (event dates/Mw/mechanism). No credentials.\n"
        )

    zip_path = rd / RUPTURES_ZIP
    xlsx_path = rd / EARTHQUAKES_XLSX
    if not zip_path.exists() or not xlsx_path.exists():
        io.check_disk()
        download.download_zenodo(
            ZENODO_RECORD, rd, filenames=[RUPTURES_ZIP, EARTHQUAKES_XLSX]
        )

    shp_dir = rd / RUPTURES_SUBDIR
    if not shp_dir.exists():
        with zipfile.ZipFile(str(zip_path)) as z:
            z.extractall(str(rd))
    # The zip may extract into a nested dir named SURE2.0_Ruptures.
    if not any(glob.glob(os.path.join(str(shp_dir), "*.shp"))):
        cands = glob.glob(os.path.join(str(rd), "**", "*_ruptures.shp"), recursive=True)
        assert cands, f"no rupture shapefiles found under {rd}"
        shp_dir = os.path.dirname(cands[0])
    return str(shp_dir), str(xlsx_path)


# --------------------------------------------------------------------------- geometry


def _load_event_dates(xlsx_path: str) -> dict[int, dict[str, Any]]:
    """Map IdE (YYYYMMDD int) -> {date, mw, mech, region} from the earthquake table."""
    import pandas as pd

    df = pd.read_excel(xlsx_path, header=0)
    out: dict[int, dict[str, Any]] = {}
    for _, r in df.iterrows():
        try:
            ide = int(r["IdE"])
            y, m, d = int(r["Year"]), int(r["Month"]), int(r["Day"])
        except (ValueError, TypeError):
            continue
        out[ide] = {
            "date": datetime(y, m, d, 12, tzinfo=UTC),
            "mw": float(r["Mw"]) if not (r["Mw"] != r["Mw"]) else None,
            "mech": str(r.get("Focal mechanism", "")),
            "region": str(r.get("Region", "")),
            "name": str(r.get("Name (hyperlink to USGS)", r.get("Name", ""))),
        }
    return out


def _to_pixels(geom: Any) -> Any:
    """UTM-metre geometry -> 10 m pixel space: (E, N) -> (E/10, -N/10)."""
    return shapely.affinity.scale(geom, xfact=0.1, yfact=-0.1, origin=(0, 0))


def build_candidates(shp_dir: str, xlsx_path: str) -> list[dict[str, Any]]:
    """Grid each kept post-2016 event's buffered rupture footprint into 64x64 tiles."""
    import geopandas as gpd
    from affine import Affine
    from rasterio.features import rasterize as rio_rasterize
    from shapely.geometry import box
    from shapely.strtree import STRtree

    dates = _load_event_dates(xlsx_path)
    records: list[dict[str, Any]] = []
    shps = sorted(glob.glob(os.path.join(shp_dir, "*_ruptures.shp")))
    kept_events: list[dict[str, Any]] = []

    for spath in shps:
        ide = int(os.path.basename(spath).split("_")[0])
        year = ide // 10000
        info = dates.get(ide)
        if info is None:
            continue
        mw = info["mw"]
        if year < MIN_YEAR:
            continue
        if mw is not None and mw < MIN_MW:
            continue

        g = gpd.read_file(spath).to_crs(4326)
        if len(g) == 0:
            continue
        c = g.geometry.union_all().centroid
        proj = get_utm_ups_projection(c.x, c.y, io.RESOLUTION, -io.RESOLUTION)
        epsg = proj.crs.to_epsg()
        g_utm = g.to_crs(epsg)

        # Buffered rupture zones in 10 m pixel space.
        buffered = [_to_pixels(geom).buffer(BUF_PX) for geom in g_utm.geometry.values]
        buffered = [b for b in buffered if not b.is_empty]
        if not buffered:
            continue
        tree = STRtree(buffered)

        minx = min(b.bounds[0] for b in buffered)
        miny = min(b.bounds[1] for b in buffered)
        maxx = max(b.bounds[2] for b in buffered)
        maxy = max(b.bounds[3] for b in buffered)
        x0 = math.floor(minx / TILE) * TILE
        y0 = math.floor(miny / TILE) * TILE
        W = int(math.ceil((maxx - x0) / TILE))
        H = int(math.ceil((maxy - y0) / TILE))

        # Fast pass: which TILE-sized cells the buffered ruptures touch.
        cover = rio_rasterize(
            ((b, 1) for b in buffered),
            out_shape=(H, W),
            transform=Affine(TILE, 0, x0, 0, TILE, y0),
            fill=0,
            dtype="uint8",
            all_touched=True,
        )
        ys, xs = np.nonzero(cover)
        change_time = info["date"]
        ev_name = os.path.basename(spath).split("_SURE")[0]
        n_ev = 0
        for tx, ty in zip(xs.tolist(), ys.tolist()):
            bx0 = x0 + tx * TILE
            by0 = y0 + ty * TILE
            bounds = (bx0, by0, bx0 + TILE, by0 + TILE)
            tbox = box(*bounds)
            idxs = tree.query(tbox)
            parts = [buffered[i].intersection(tbox) for i in idxs]
            parts = [p for p in parts if not p.is_empty]
            if not parts:
                continue
            clip = shapely.unary_union(parts)
            if clip.is_empty:
                continue
            records.append(
                {
                    "epsg": epsg,
                    "bounds": bounds,
                    "change_ms": int(change_time.timestamp() * 1000),
                    "source_id": f"{ev_name}",
                    "rupture_wkb": shapely.wkb.dumps(clip),
                }
            )
            n_ev += 1
        kept_events.append(
            {
                "event": ev_name,
                "ide": ide,
                "mw": mw,
                "mech": info["mech"],
                "tiles": n_ev,
            }
        )
        print(f"  {ev_name:26s} Mw={mw} traces={len(g):6d} tiles={n_ev}")

    records.sort(key=lambda r: (r["source_id"], r["bounds"]))
    build_candidates.kept_events = kept_events  # type: ignore[attr-defined]
    return records


# --------------------------------------------------------------------------- write


def _write_one(rec: dict[str, Any]) -> tuple[str, list[int]] | None:
    from olmoearth_pretrain.open_set_segmentation_data.rasterize import rasterize_shapes

    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return rec["source_id"], []

    proj = Projection(CRS.from_epsg(rec["epsg"]), io.RESOLUTION, -io.RESOLUTION)
    bounds = tuple(rec["bounds"])
    clip = shapely.wkb.loads(rec["rupture_wkb"])
    label = rasterize_shapes(
        [(clip, RUPTURE)], bounds, fill=BG, dtype="uint8", all_touched=True
    )[0]
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
    return rec["source_id"], present


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    io.check_disk()
    shp_dir, xlsx_path = ensure_raw()
    io.check_disk()

    records = build_candidates(shp_dir, xlsx_path)
    if len(records) > MAX_SAMPLES:  # not expected (~1300); guard the hard cap anyway
        import random

        random.Random(42).shuffle(records)
        records = records[:MAX_SAMPLES]
        records.sort(key=lambda r: (r["source_id"], r["bounds"]))
    for j, r in enumerate(records):
        r["sample_id"] = f"{j:06d}"
    print(f"{len(records)} candidate rupture tiles across kept events")

    io.check_disk()
    class_tiles: Counter = Counter()
    per_event: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in records]),
            total=len(records),
            desc="write tiles",
        ):
            if res is None:
                continue
            src, present = res
            per_event[src] += 1
            for c in present:
                class_tiles[c] += 1

    n_written = len(list(io.locations_dir(SLUG).glob("*.tif")))
    kept_events = getattr(build_candidates, "kept_events", [])
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "INQUA / Sci Data (Nurminen et al. 2022)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": URL,
                "paper": PAPER,
                "zenodo_record": ZENODO_RECORD,
                "have_locally": False,
                "annotation_method": "manual field mapping (SURE 2.0 unified rupture traces)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": n_written,
            "is_change_dataset": True,
            "buffer_m": BUF_PX * io.RESOLUTION,
            "min_magnitude": MIN_MW,
            "min_year": MIN_YEAR,
            "tiles_per_class": {
                CLASSES[c][0]: class_tiles.get(c, 0) for c in (BG, RUPTURE)
            },
            "kept_events": kept_events,
            "tiles_per_event": dict(sorted(per_event.items())),
            "notes": (
                "Coseismic surface-rupture change segmentation from SURE 2.0. Of 50 events "
                "(1872-2019), 42 pre-2016 events are DROPPED (pre-2016 change rule + "
                "decades-eroded surface expression) and 2019 Le Teil (Mw 4.9, ~cm offsets, "
                "sub-pixel at 10 m) is dropped for observability. 7 kept events are all "
                "Mw >= 6.0 and 2016+ (Kumamoto, Petermann, Amatrice, Norcia, Parina, "
                "Ridgecrest 1 & 2). Rupture LineStrings (WGS84) buffered to a ~30 m "
                "half-width (~60 m wide) rupture zone and rasterized (1=surface_rupture, "
                "0=background) into 64x64 uint8 tiles in each event's local UTM at 10 m. "
                "All fault ranks (principal + distributed) merged into one rupture class. "
                "Event dates are day-precise, so change_time = the earthquake date and "
                "time_range = +/-180 d centered on it (genuine before->after rupture "
                "signal). Non-rupture pixels are background (0), not ignore; 255 nodata "
                "unused. Caveat: distributed rupturing may be locally under-mapped and the "
                "smallest kept events (Mw ~6) have narrow zones near the 10 m limit."
            ),
        },
    )
    print("tiles per class (id->#tiles):", dict(class_tiles))
    print("tiles per event:", dict(sorted(per_event.items())))
    print("total tif on disk:", n_written)
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()

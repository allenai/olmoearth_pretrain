"""Kuro Siwo (global multi-temporal SAR flood dataset) -> open-set-segmentation masks.

Source: Kuro Siwo (Bountos et al., NeurIPS 2024; Orion-AI-Lab). A global, manually
annotated (by SAR experts) multi-temporal Sentinel-1 flood-mapping benchmark spanning
Copernicus EMS Rapid Mapping flood activations (EMSR events). Home:
https://github.com/Orion-AI-Lab/KuroSiwo (CC-BY).

Label-only extraction (spec S3/S8): the full Kuro Siwo GRD/SLC products ship the SAR
imagery bundled with the masks in large Dropbox/HF archives, but pretraining supplies its
own S1/S2 imagery -- we need ONLY the labels. Kuro Siwo publishes its annotation polygons
separately in the small companion repo
https://github.com/Orion-AI-Lab/KuroSiwo-annotations (git-cloned, ~a few hundred MB, no
SAR), and the per-event acquisition/reference dates live in the main repo's
``catalogue/catalogue.yaml``. We use those two sources only.

Per Copernicus EMS activation (EMSR<act_id>_<region>), one or more AOIs, each a mapped
revision folder with three shapefiles (all EPSG:3857):
  * ``aoi/aoi.shp``     -- the mapped AOI extent (1 polygon): defines the observed region.
  * ``event/event.shp`` -- the observed flood-water extent polygons for the event.
  * ``hydro/hydroA.shp``-- reference permanent-water bodies (rivers, lakes, reservoirs).

3-class dense per-pixel CLASSIFICATION (Kuro Siwo's native MLU scheme):
    id 0 = no_water         (inside AOI, neither flood nor permanent water)
    id 1 = permanent_water  (hydroA reference water; wins over flood on overlap)
    id 2 = flood            (event flood-water extent)
    255  = nodata/ignore    (outside the mapped AOI)
Permanent water is painted last so it wins flood/no-water overlaps (matching sen1floods11's
"permanent wins" convention: a flooded river channel is still permanent water).

Processing (label_type = dense_raster): work in each AOI's local-UTM pixel space at 10 m.
Each AOI is rasterized ONCE across its whole pixel grid (no_water=0/permanent=1/flood=2
inside the AOI, 255 outside; even the largest AOI, Pakistan ~25k x 41k px, is ~1 GB
uint8), then the 64x64 tiles that contain a flood or permanent-water class (the signal;
no-water co-occurs inside those tiles as the surrounding land) are found and sliced with
pure vectorized numpy. Rasterizing once (rather than re-rasterizing the AOI's tens of
thousands of flood/hydro polygons per window) is what keeps huge AOIs fast. Tiles are then
tiles-per-class balanced (spec S5, <=1000/class, rare flood prioritized).

Change label (spec S5): a flood is a transient event, so change_time is set to the event's
reference date (catalogue.yaml ``ref_date``, resolved to the day -- well within the
~1-2 month timing requirement) and kept as the reference used to build two adjacent windows
via ``io.pre_post_time_ranges``: ``pre_time_range`` (the ~6 months, <=183 days, immediately
before change_time) and ``post_time_range`` (the ~6 months, <=183 days, immediately after);
``time_range`` is null. Pretraining pairs a "before" image stack with an "after" stack and
probes on their difference. Events dated before 2016 (outside the Sentinel era) are dropped
(EMSR118/130/147).

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.kuro_siwo
"""

import argparse
import glob
import math
import multiprocessing
import re
import subprocess
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import shapely
import tqdm
import yaml
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import (
    io,
    manifest,
    rasterize,
    sampling,
)

SLUG = "kuro_siwo"
NAME = "Kuro Siwo"

ANNOT_REPO = "https://github.com/Orion-AI-Lab/KuroSiwo-annotations.git"
CATALOGUE_URL = "https://raw.githubusercontent.com/Orion-AI-Lab/KuroSiwo/main/catalogue/catalogue.yaml"

TILE = 64
PER_CLASS = 1000
MIN_CLASS_PX = 32  # a tile counts toward a class only with >= this many px of it
MAX_NODATA_FRAC = 0.5  # skip tiles that are more than half outside the AOI (nodata)
# Per-AOI candidate-tile caps (seeded random subsample) so a few huge AOIs don't dominate
# and memory/scan-time stay bounded; the dataset only needs ~1000 flood tiles overall.
FLOOD_CAP_PER_AOI = 400
PERM_CAP_PER_AOI = 200
MIN_YEAR = 2016  # Sentinel era; drop earlier events (spec S2 post-2016 rule)

NO_WATER, PERM, FLOOD = 0, 1, 2
CLASSES = [
    (
        "no_water",
        "Inside the mapped AOI but neither flood-water nor permanent water at the event "
        "acquisition, i.e. dry land / non-water observed by the SAR expert annotation.",
    ),
    (
        "permanent_water",
        "Reference permanent open water (rivers, lakes, reservoirs) from the Copernicus EMS "
        "hydrography layer (hydroA), co-registered to the event. Painted over flood on "
        "overlap (a flooded permanent channel stays permanent water).",
    ),
    (
        "flood",
        "Observed flood-water extent at the event's Sentinel-1 acquisition (Copernicus EMS "
        "'event' delineation), excluding pixels reclassified as permanent water.",
    ),
]


def raw_root():
    return io.raw_dir(SLUG)


def _annot_dir():
    return raw_root() / "KuroSiwo-annotations"


def _catalogue_path():
    return raw_root() / "catalogue.yaml"


def download_raw() -> None:
    """Clone the annotation-polygon repo and fetch catalogue.yaml (idempotent, label-only)."""
    raw_root().mkdir(parents=True, exist_ok=True)
    io.check_disk()
    annot = _annot_dir()
    if not (annot / ".git").exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", ANNOT_REPO, str(annot)], check=True
        )
    cat = _catalogue_path()
    if not cat.exists():
        subprocess.run(["curl", "-sL", CATALOGUE_URL, "-o", str(cat)], check=True)
    with (raw_root() / "SOURCE.txt").open("w") as f:
        f.write(
            "Kuro Siwo flood dataset (Orion-AI-Lab, NeurIPS 2024), CC-BY.\n"
            "Label-only extraction: annotation polygons git-cloned from "
            f"{ANNOT_REPO} (event=flood, hydroA=permanent water, aoi=extent; EPSG:3857 "
            "shapefiles), per-event reference dates from the main repo's "
            "catalogue/catalogue.yaml. No SAR imagery is downloaded (pretraining supplies "
            "its own). Full GRD/SLC products live on the project's Dropbox / Hugging Face.\n"
        )


def _load_event_dates() -> dict[str, datetime]:
    """act_id (str) -> flood reference datetime (UTC), parsed from catalogue.yaml."""
    yaml.add_constructor(
        "!join",
        lambda loader, node: "".join(str(i) for i in loader.construct_sequence(node)),
        Loader=yaml.SafeLoader,
    )
    with _catalogue_path().open() as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    dates: dict[str, datetime] = {}
    for flood in cfg["Floods"]:
        d = datetime.strptime(str(flood["ref_date"]), "%Y%m%dT%H%M%S").replace(
            tzinfo=UTC
        )
        dates[str(flood["act_id"])] = d
    return dates


def _list_aoi_jobs() -> list[dict[str, Any]]:
    """Enumerate processable AOI revision dirs (post-2016), with their event date."""
    dates = _load_event_dates()
    jobs: list[dict[str, Any]] = []
    pattern = str(
        _annot_dir() / "polygons" / "*" / "aoi" / "*" / "*" / "aoi" / "aoi.shp"
    )
    for aoi_shp in sorted(glob.glob(pattern)):
        parts = aoi_shp.split("/")
        event_dir, aoi_id, rev = parts[-6], parts[-4], parts[-3]
        m = re.match(r"EMSR(\d+)_", event_dir)
        if not m:
            continue
        act_id = m.group(1)
        change_time = dates.get(act_id)
        if change_time is None or change_time.year < MIN_YEAR:
            continue
        base = aoi_shp[: -len("/aoi/aoi.shp")]
        jobs.append(
            {
                "base": base,
                "event_dir": event_dir,
                "aoi_id": aoi_id,
                "rev": rev,
                "act_id": act_id,
                "change_time": change_time,
            }
        )
    return jobs


_PX_SCALE = np.array([1.0 / io.RESOLUTION, -1.0 / io.RESOLUTION])


def _read_px_geoms(shp_path: str, utm_epsg: int) -> np.ndarray:
    """Read a shapefile, reproject to UTM, return valid geoms in 10 m pixel space.

    Fully vectorized (shapely 2.x C ops over the whole GeometryArray) so AOIs with tens
    of thousands of polygons (e.g. Pakistan ~48k) process in seconds, not minutes.
    """
    import os

    import geopandas as gpd

    if not os.path.exists(shp_path):
        return np.empty(0, dtype=object)
    try:
        gdf = gpd.read_file(shp_path)
    except Exception:
        return np.empty(0, dtype=object)
    if gdf.empty:
        return np.empty(0, dtype=object)
    geoms = gdf.to_crs(utm_epsg).geometry.values
    # Skip make_valid: GEOS make_valid is pathologically slow on some EMS polygons, and
    # rasterio.features.rasterize scan-fills invalid (self-intersecting) polygons fine.
    geoms = geoms[~shapely.is_missing(geoms)]
    geoms = geoms[~shapely.is_empty(geoms)]
    if len(geoms) == 0:
        return geoms
    # crs-metres -> north-up 10 m pixel coords (x/10, -y/10), vectorized over all geoms.
    return shapely.transform(geoms, lambda c: c * _PX_SCALE)


def _scan_aoi(
    base: str,
    event_dir: str,
    aoi_id: str,
    rev: str,
    act_id: str,
    change_time: datetime,
) -> list[dict[str, Any]]:
    """Rasterize one AOI once at 10 m, then slice water-bearing 64x64 tiles into records.

    Rather than re-rasterizing per tile (which re-processes the AOI's giant flood/hydro
    multipolygons once per window and blows up on huge AOIs like Pakistan ~48k polygons),
    each label layer is rasterized ONCE across the whole AOI pixel grid (memory is cheap:
    even the largest AOI is ~1 GB uint8), then candidate tiles are found and sliced with
    pure vectorized numpy. Semantics are identical: 0=no_water / 1=permanent / 2=flood
    inside the AOI, 255 outside; permanent painted last so it wins flood/no-water overlaps.
    """
    import random

    import geopandas as gpd

    # UTM zone from the AOI centroid (WGS84).
    aoi_gdf = gpd.read_file(f"{base}/aoi/aoi.shp")
    if aoi_gdf.empty:
        return []
    c = aoi_gdf.to_crs(4326).geometry.union_all().centroid
    proj = io.utm_projection_for_lonlat(float(c.x), float(c.y))
    utm_epsg = proj.crs.to_epsg()

    aoi_px = _read_px_geoms(f"{base}/aoi/aoi.shp", utm_epsg)
    flood_px = _read_px_geoms(f"{base}/event/event.shp", utm_epsg)
    perm_px = _read_px_geoms(f"{base}/hydro/hydroA.shp", utm_epsg)
    if len(aoi_px) == 0 or (len(flood_px) == 0 and len(perm_px) == 0):
        return []

    xs0, ys0, xs1, ys1 = [], [], [], []
    for g in aoi_px:
        bx0, by0, bx1, by1 = g.bounds
        xs0.append(bx0)
        ys0.append(by0)
        xs1.append(bx1)
        ys1.append(by1)
    x0 = int(math.floor(min(xs0)))
    y0 = int(math.floor(min(ys0)))
    ntx = max(1, int(math.ceil((max(xs1) - x0) / TILE)))
    nty = max(1, int(math.ceil((max(ys1) - y0) / TILE)))
    W, H = ntx * TILE, nty * TILE  # pixel grid, exact multiples of TILE
    whole = (x0, y0, x0 + W, y0 + H)

    # Rasterize each layer ONCE over the whole grid (0 inside AOI, 255 outside).
    full = rasterize.rasterize_shapes(
        [(g, 0) for g in aoi_px], whole, fill=io.CLASS_NODATA, dtype="uint8"
    )[0]
    inside = full == 0
    if len(flood_px) > 0:
        fm = rasterize.rasterize_shapes(
            [(g, 1) for g in flood_px], whole, fill=0, dtype="uint8"
        )[0]
        full[inside & (fm == 1)] = FLOOD
        del fm
    if len(perm_px) > 0:
        pm = rasterize.rasterize_shapes(
            [(g, 1) for g in perm_px], whole, fill=0, dtype="uint8"
        )[0]
        full[inside & (pm == 1)] = PERM  # permanent wins over flood
        del pm

    # Per-tile class pixel counts, vectorized over the (nty, TILE, ntx, TILE) block view.
    blk = full.reshape(nty, TILE, ntx, TILE)
    cnt_now = (blk == NO_WATER).sum(axis=(1, 3))
    cnt_perm = (blk == PERM).sum(axis=(1, 3))
    cnt_flood = (blk == FLOOD).sum(axis=(1, 3))
    cnt_inside = cnt_now + cnt_perm + cnt_flood

    has_flood = cnt_flood >= MIN_CLASS_PX
    has_perm = cnt_perm >= MIN_CLASS_PX
    enough_inside = cnt_inside >= MAX_NODATA_FRAC * (TILE * TILE)
    # A usable tile carries a water class (flood or permanent) and is mostly inside the AOI.
    keep = (has_flood | has_perm) & enough_inside

    flood_ij = [(int(i), int(j)) for i, j in zip(*np.nonzero(keep & has_flood))]
    perm_only_ij = [
        (int(i), int(j)) for i, j in zip(*np.nonzero(keep & has_perm & ~has_flood))
    ]

    rng = random.Random(hash((event_dir, aoi_id, rev)) & 0xFFFFFFFF)
    if len(flood_ij) > FLOOD_CAP_PER_AOI:
        rng.shuffle(flood_ij)
        flood_ij = flood_ij[:FLOOD_CAP_PER_AOI]
    if len(perm_only_ij) > PERM_CAP_PER_AOI:
        rng.shuffle(perm_only_ij)
        perm_only_ij = perm_only_ij[:PERM_CAP_PER_AOI]
    cand = sorted(set(flood_ij) | set(perm_only_ij))

    recs: list[dict[str, Any]] = []
    for ti, tj in cand:
        out = np.ascontiguousarray(
            full[ti * TILE : (ti + 1) * TILE, tj * TILE : (tj + 1) * TILE]
        )
        present = [
            cls
            for cls, cnt in (
                (NO_WATER, cnt_now[ti, tj]),
                (PERM, cnt_perm[ti, tj]),
                (FLOOD, cnt_flood[ti, tj]),
            )
            if cnt >= MIN_CLASS_PX
        ]
        if not present or present == [NO_WATER]:
            continue
        bx0 = x0 + tj * TILE
        by0 = y0 + ti * TILE
        recs.append(
            {
                "array": out.astype(np.uint8),
                "crs": proj.crs.to_string(),
                "bounds": [bx0, by0, bx0 + TILE, by0 + TILE],
                "classes_present": present,
                "change_time": change_time.isoformat(),
                "source_id": f"{event_dir}/aoi{aoi_id}/{rev}/r{ti}_c{tj}",
            }
        )
    return recs


def _write_one(rec: dict[str, Any]) -> int:
    from rasterio.crs import CRS
    from rslearn.utils.geometry import Projection

    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return 0
    proj = Projection(CRS.from_string(rec["crs"]), io.RESOLUTION, -io.RESOLUTION)
    bounds = tuple(rec["bounds"])
    ct = datetime.fromisoformat(rec["change_time"])
    pre_range, post_range = io.pre_post_time_ranges(ct)
    tr = (pre_range[0], post_range[1])  # outer bounding span
    io.write_label_geotiff(
        SLUG, sample_id, rec["array"], proj, bounds, nodata=io.CLASS_NODATA
    )
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        tr,
        change_time=ct,
        source_id=rec["source_id"],
        classes_present=rec["classes_present"],
        pre_time_range=pre_range,
        post_time_range=post_range,
    )
    return 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument(
        "--probe", action="store_true", help="scan/report only, no writes"
    )
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    print("Downloading Kuro Siwo annotation polygons (label-only)...")
    download_raw()
    jobs = _list_aoi_jobs()
    print(f"  {len(jobs)} post-{MIN_YEAR} AOIs to scan")
    io.check_disk()

    print("Scanning AOIs into 64x64 tiles...")
    all_recs: list[dict[str, Any]] = []
    with multiprocessing.Pool(args.workers) as p:
        for recs in tqdm.tqdm(star_imap_unordered(p, _scan_aoi, jobs), total=len(jobs)):
            all_recs.extend(recs)
    print(f"  {len(all_recs)} candidate tiles")

    selected = sampling.select_tiles_per_class(
        all_recs, classes_key="classes_present", per_class=PER_CLASS
    )
    selected.sort(key=lambda r: r["source_id"])
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(
        f"  selected {len(selected)} tiles (tiles-per-class balanced, <= {PER_CLASS}/class)"
    )

    tile_counts = {name: 0 for name, _ in CLASSES}
    for r in selected:
        for c in r["classes_present"]:
            tile_counts[CLASSES[c][0]] += 1
    print("tiles containing each class:", tile_counts)

    if args.probe:
        print("probe only; exiting before writes")
        return

    io.check_disk()
    print(f"Writing {len(selected)} tiles...")
    written = 0
    with multiprocessing.Pool(args.workers) as p:
        for n in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            written += n
    print(f"wrote {written} new tiles ({len(selected)} total selected)")

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Kuro Siwo (Orion-AI-Lab) / NeurIPS 2024",
            "license": "CC-BY",
            "provenance": {
                "url": "https://github.com/Orion-AI-Lab/KuroSiwo",
                "annotations_repo": ANNOT_REPO,
                "have_locally": False,
                "annotation_method": "manual (SAR experts); Copernicus EMS delineation polygons",
                "citation": "Bountos et al. 2024, NeurIPS (Kuro Siwo)",
            },
            "sensors_relevant": ["sentinel1", "sentinel2"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "tile_class_counts": tile_counts,
            "notes": (
                "Label-only extraction from the Kuro Siwo annotation polygons "
                "(event=flood, hydroA=permanent water, aoi=extent; EPSG:3857); no SAR "
                "imagery downloaded. Per-event reference dates from catalogue.yaml. Each AOI "
                "reprojected to local UTM at 10 m; only 64x64 tiles intersecting a flood or "
                "permanent-water polygon are rasterized (memory-safe on huge AOIs) with "
                "no_water(0)/permanent(1)/flood(2), 255 outside AOI; permanent painted last "
                "(wins overlaps). Tiles-per-class balanced (<=1000/class); flood prioritized. "
                "Flood is an event label: change_time set to the event reference date, "
                "time_range a 360-day window centered on it. Events before 2016 "
                "(EMSR118/130/147) dropped. Per-AOI candidate caps "
                f"(flood {FLOOD_CAP_PER_AOI}, permanent-only {PERM_CAP_PER_AOI}) keep "
                "geographic diversity."
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

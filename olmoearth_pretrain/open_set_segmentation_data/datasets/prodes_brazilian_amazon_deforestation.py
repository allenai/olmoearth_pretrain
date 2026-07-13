"""Process INPE PRODES yearly clear-cut deforestation for the Brazilian Amazon.

Source: INPE TerraBrasilis PRODES, served as WFS from the TerraBrasilis GeoServer
(https://terrabrasilis.dpi.inpe.br/geoserver/ows). PRODES is the official annual,
wall-to-wall monitoring of clear-cut (corte raso) deforestation in the Brazilian Legal
Amazon, produced by expert visual photointerpretation of Landsat-class imagery
(Landsat-8/OLI, CBERS-4, Sentinel-2, etc.). Each yearly increment is a set of polygons of
newly clear-cut forest, tagged with:

* ``year``       -- the PRODES reference year the increment belongs to (Aug..Jul cycle).
* ``image_date`` -- the actual satellite image DATE on which the clear-cut was confirmed
                    (day-precise; this is our change_time).
* ``main_class`` -- always ``DESMATAMENTO`` (deforestation) for this increment layer.
* ``sub_class``  -- corte raso variant (exposed soil / residual vegetation) for recent
                    years; a ``d{year}`` placeholder for older years.
* ``state``      -- Legal Amazon state (AC/AM/AP/MA/MT/PA/RO/RR/TO).

Layer used: ``prodes-legal-amz:yearly_deforestation`` (the classic Brazilian Legal Amazon
PRODES yearly increment).

This is a dated CHANGE dataset (forest -> clear-cut). Per spec 5 we use the change_time
scheme: each sample's ``change_time`` is the polygon's ``image_date`` (day-precise, well
within the required ~1-2 month timing precision) and its ``time_range`` is a 360-day window
centered on that date. The label is a **mask of where** the clear-cut happened. A completed
clear-cut persists in imagery, so a 1-year window centered on the confirmation date is
well-posed for change pairing.

Post-2016 rule: we keep only polygons whose ``image_date`` is on/after 2016-01-01 (the
Sentinel era). PRODES-year-2016 polygons imaged in late 2015 are dropped.

Encoding (spec 4 polygons, mirroring the DETER-B recipe): for each selected polygon center
a 64x64 UTM 10 m tile on the polygon centroid, rasterize that polygon (all_touched so small
polygons register) as class id 1 (deforestation), background 0 elsewhere. To keep visible
background context (and avoid pathological giant polygons filling the whole tile) we select
polygons with 0.002 <= area_km <= 0.4 km^2 (a 640 m tile is 0.41 km^2). Only the target
polygon is drawn; any co-located clear-cut of another date is left as background.

Class scheme (uint8):
  0 background     (no clear-cut in this pixel: standing forest / other cover)
  1 deforestation  (PRODES clear-cut / corte raso)

Up to TARGET_PER_CLASS tiles for the deforestation class, sampled round-robin across
PRODES years for temporal diversity and across states for spatial diversity (spec 5).
"""

import argparse
import json
import multiprocessing
import random
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

import shapely
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.mp import star_imap_unordered
from shapely.geometry import shape

from olmoearth_pretrain.open_set_segmentation_data import io
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)

SLUG = "prodes_brazilian_amazon_deforestation"
NAME = "PRODES (Brazilian Amazon deforestation)"
URL = "https://terrabrasilis.dpi.inpe.br/downloads/"
WFS = "https://terrabrasilis.dpi.inpe.br/geoserver/ows"
LAYER = "prodes-legal-amz:yearly_deforestation"

TILE = 64  # 64 px @ 10 m = 640 m
BACKGROUND_ID = 0
DEFOR_ID = 1
TARGET_PER_CLASS = 1000
FETCH_PER_QUERY = 80  # candidates fetched per (state, year)
YEARS = list(range(2016, 2026))
# Legal Amazon states.
STATES = ["AC", "AM", "AP", "MA", "MT", "PA", "RO", "RR", "TO"]
AREA_MIN_KM = 0.002  # ~0.2 ha -> registers at 10 m
AREA_MAX_KM = 0.4  # < 640 m tile (0.41 km^2) so background context remains
MIN_IMAGE_DATE = "2016-01-01"  # post-2016 / Sentinel era
SEED = 42
WINDOW_HALF_DAYS = 180  # +/- 180 d -> 360-day time range centered on image_date

CLASS_DEFS = [
    (
        0,
        "background",
        "No PRODES clear-cut in this pixel: standing forest, previously-cleared "
        "land, water, or other cover within the tile.",
    ),
    (
        1,
        "deforestation",
        "PRODES annual clear-cut deforestation (corte raso / DESMATAMENTO): complete "
        "removal of primary forest confirmed by expert photointerpretation in the "
        "Brazilian Legal Amazon. Includes exposed-soil and residual-vegetation "
        "clear-cut sub-classes.",
    ),
]


def _raw_path(state: str, year: int):
    return io.raw_dir(SLUG) / f"{state}__{year}.geojson"


def _fetch(state: str, year: int) -> dict[str, Any]:
    cql = (
        f"state='{state}' AND year={year} "
        f"AND area_km BETWEEN {AREA_MIN_KM} AND {AREA_MAX_KM}"
    )
    q = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": LAYER,
        "count": str(FETCH_PER_QUERY),
        "outputFormat": "application/json",
        "srsName": "EPSG:4326",
        "CQL_FILTER": cql,
    }
    url = WFS + "?" + urllib.parse.urlencode(q)
    last = None
    for _ in range(4):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"WFS fetch failed for {state}/{year}: {last}")


def download_all() -> None:
    """Fetch candidate GeoJSONs per (state, year) to raw/ (idempotent)."""
    io.raw_dir(SLUG).mkdir(parents=True, exist_ok=True)
    for state in STATES:
        for year in YEARS:
            dst = _raw_path(state, year)
            if dst.exists():
                continue
            data = _fetch(state, year)
            tmp = dst.parent / (dst.name + ".tmp")
            with tmp.open("w") as f:
                json.dump(data, f)
            tmp.rename(dst)
    with (io.raw_dir(SLUG) / "SOURCE.txt").open("w") as f:
        f.write(
            "INPE TerraBrasilis PRODES yearly clear-cut deforestation (Legal Amazon).\n"
            f"{URL}\n"
            f"WFS: {WFS}\n"
            f"Layer: {LAYER}\n"
            "Fetched per (state, year) with CQL area_km filter; srsName=EPSG:4326.\n"
            "Source native CRS EPSG:4674 (SIRGAS 2000), reprojected to EPSG:4326 by WFS.\n"
        )


def _load_candidates() -> dict[int, list[dict[str, Any]]]:
    """Load raw GeoJSONs -> {year: [record, ...]} (records carry WKB geom + image_date).

    Keyed by PRODES year so we can round-robin across years for temporal diversity.
    """
    by_year: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for state in STATES:
        for year in YEARS:
            p = _raw_path(state, year)
            if not p.exists():
                continue
            with p.open() as f:
                data = json.load(f)
            for feat in data.get("features", []):
                geom = feat.get("geometry")
                props = feat.get("properties", {})
                img = props.get("image_date")
                if geom is None or not img:
                    continue
                # post-2016 rule: change_time must be in the Sentinel era.
                if str(img)[:10] < MIN_IMAGE_DATE:
                    continue
                try:
                    g = shape(geom)
                except Exception:  # noqa: BLE001
                    continue
                if g.is_empty:
                    continue
                if not g.is_valid:
                    g = g.buffer(0)
                    if g.is_empty or not g.is_valid:
                        continue
                by_year[year].append(
                    {
                        "wkb": shapely.to_wkb(g),
                        "image_date": str(img)[:10],
                        "year": year,
                        "state": state,
                        "gid": feat.get("id", ""),
                    }
                )
    return by_year


def _sample(by_year: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Pick <= TARGET_PER_CLASS records, round-robin across years (shuffled within year)."""
    rng = random.Random(SEED)
    for yr in by_year:
        rng.shuffle(by_year[yr])
    years = sorted(by_year)
    idx = {yr: 0 for yr in years}
    picked: list[dict[str, Any]] = []
    while len(picked) < TARGET_PER_CLASS:
        progressed = False
        for yr in years:
            if idx[yr] < len(by_year[yr]):
                picked.append(by_year[yr][idx[yr]])
                idx[yr] += 1
                progressed = True
                if len(picked) >= TARGET_PER_CLASS:
                    break
        if not progressed:
            break
    return picked


def _time_range(image_date: str) -> tuple[datetime, datetime]:
    ct = datetime.strptime(image_date, "%Y-%m-%d").replace(tzinfo=UTC)
    return ct - timedelta(days=WINDOW_HALF_DAYS), ct + timedelta(days=WINDOW_HALF_DAYS)


def _write_tile(rec: dict[str, Any]) -> int:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return DEFOR_ID
    g = shapely.from_wkb(rec["wkb"])
    c = g.centroid
    proj, col, row = io.lonlat_to_utm_pixel(c.x, c.y)
    bounds = io.centered_bounds(col, row, TILE, TILE)
    px = geom_to_pixels(g, WGS84_PROJECTION, proj)
    if px.is_empty:
        return -1
    if not px.is_valid:
        px = px.buffer(0)
        if px.is_empty or not px.is_valid:
            return -1
    arr = rasterize_shapes(
        [(px, DEFOR_ID)], bounds, fill=BACKGROUND_ID, dtype="uint8", all_touched=True
    )
    if int(arr.max()) != DEFOR_ID:
        return -1  # polygon fell outside the centered tile (shouldn't happen); skip
    ct = datetime.strptime(rec["image_date"], "%Y-%m-%d").replace(tzinfo=UTC)
    tr = _time_range(rec["image_date"])
    present = (
        [BACKGROUND_ID, DEFOR_ID] if int((arr == BACKGROUND_ID).sum()) else [DEFOR_ID]
    )
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        tr,
        change_time=ct,
        source_id=f"{rec['state']}:{rec['year']}:{rec['gid']}",
        classes_present=present,
    )
    return DEFOR_ID


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    from olmoearth_pretrain.open_set_segmentation_data import manifest

    manifest.write_registry_entry(SLUG, "in_progress")
    io.check_disk()
    download_all()
    io.check_disk()

    by_year = _load_candidates()
    print(
        "candidates per year: "
        + ", ".join(f"{yr}:{len(v)}" for yr, v in sorted(by_year.items())),
        flush=True,
    )
    chosen = _sample(by_year)
    for i, r in enumerate(chosen):
        r["sample_id"] = f"{i:06d}"
    print(f"selected {len(chosen)} tiles (<= {TARGET_PER_CLASS})", flush=True)

    io.check_disk()
    with multiprocessing.Pool(args.workers) as p:
        results = list(
            tqdm.tqdm(
                star_imap_unordered(p, _write_tile, [dict(rec=r) for r in chosen]),
                total=len(chosen),
                desc="tiles",
            )
        )

    n_defor = sum(1 for r in results if r == DEFOR_ID)
    n_degenerate = sum(1 for r in results if r == -1)
    # tiles-per-class balanced counts: background is present in essentially every tile.
    year_counts = Counter(r["year"] for r in chosen[: len(results)])
    total = n_defor

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "INPE TerraBrasilis (PRODES)",
            "license": "CC-BY-SA-4.0",
            "provenance": {
                "url": URL,
                "have_locally": False,
                "annotation_method": "manual photointerpretation (INPE analysts)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": cid, "name": name, "description": desc}
                for cid, name, desc in CLASS_DEFS
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": total,
            "class_counts": {"deforestation": n_defor},
            "samples_per_year": {
                str(y): year_counts.get(y, 0) for y in sorted(year_counts)
            },
            "tile_size": TILE,
            "change_time_scheme": True,
            "time_range_days": 2 * WINDOW_HALF_DAYS,
            "notes": (
                "64x64 UTM 10 m tiles centered on each PRODES yearly clear-cut polygon "
                "centroid; the polygon is rasterized (all_touched) as class 1 "
                "(deforestation), background=0 elsewhere. Dated change labels: "
                "change_time=image_date (day-precise), time_range=+/-180 d (360 d) "
                "centered on it. Layer prodes-legal-amz:yearly_deforestation (Brazilian "
                "Legal Amazon). Only clear-cut polygons with 0.002<=area_km<=0.4 selected "
                "so a 640 m tile keeps background context and giant polygons are excluded. "
                "Only the target polygon is drawn (co-located clear-cuts of other dates "
                "left as background). Post-2016 filter: image_date >= 2016-01-01. Source "
                "CRS EPSG:4674 (SIRGAS 2000), reprojected to EPSG:4326 by WFS then to "
                f"local UTM. {n_degenerate} tiles dropped as degenerate."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=total
    )
    print(
        f"done: {total} tiles; dropped {n_degenerate}; per-year={dict(sorted(year_counts.items()))}",
        flush=True,
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()

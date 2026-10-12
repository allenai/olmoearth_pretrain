"""Earth Impact Database (EID) -> open-set-segmentation impact-structure footprints.

Source: Earth Impact Database, Planetary and Space Science Centre (PASSC), University of
New Brunswick, Canada (http://www.passc.net/EarthImpactDatabase/). The EID is the
definitive catalog of confirmed terrestrial impact structures (190 confirmed as of the
2018 web release). Each structure has a per-crater HTML page carrying a small data table:
name, location, latitude, longitude, diameter (km), age (Ma), exposed/drilled flags,
target rock and bolide type. License: "free scholarly use" (not-for-profit scientific
resource); attribution to PASSC/UNB recorded below. This research use is in scope.

There is no bulk download; the catalog is scraped from the HTML. We fetch the
"sorted by Name" index to enumerate the ~198 per-structure page URLs, then fetch each
per-structure page and parse its data table for the WGS84 coordinates (DMS) and diameter.

TRIAGE / suitability (spec 2, 4, 5, 8) -- ACCEPTED as a single-class PRESENCE
segmentation (per-pixel classification), NOT a change dataset:

  * Observability at 10 m: impact structures span 0.01-160 km. We FILTER to structures
    with a diameter >= 3 km (149 of the 197 parseable structures). The 3 km cutoff is
    derived from the encoding + coordinate precision (see below), and every retained
    structure is many hundreds of pixels across at 10 m -- clearly a resolvable circular
    landform (rim, central uplift, annular structure). Smaller structures (< 3 km) are
    dropped: either near the resolution limit or so small that the label tile cannot be
    guaranteed to land inside the structure given the catalog's arc-minute coordinates.
  * Coordinate precision: EID coordinates are given to the arc-minute (a few to the
    arc-second, e.g. Dhala). Worst-case rounding error is +/- 0.5' latitude ~= 0.93 km.
    For a 64 px (640 m) label tile centered on the catalog point, the farthest tile pixel
    is 0.93 km (coord error) + 0.45 km (tile half-diagonal, 320 m * sqrt(2)) = 1.38 km
    from the true structure center. Requiring that <= structure radius gives diameter >=
    2.77 km, rounded up to a clean 3 km cutoff -- so the whole 640 m footprint tile is
    guaranteed to lie within the impact structure despite coordinate imprecision.
  * Encoding (spec 4, polygons/footprint): impact structures are (roughly) circular and
    have a real footprint, so this is NOT a 1x1 point. Each structure is rasterized as a
    circular footprint of radius = diameter/2 into a 64x64 UTM tile at 10 m centered on
    the structure. Interior pixels = class 0 (impact_structure); pixels outside the
    circle within the tile = nodata/ignore (255). Because all retained structures are
    >= 3 km across (>= 300 px), the circle covers the entire 640 m tile: each output tile
    is a coherent 640 m patch of confirmed impact-structure surface. POSITIVE-ONLY, no
    fabricated background class -- the assembly step supplies negatives from other
    datasets (spec 5).
  * Time validity: impact structures are persistent landforms (ages Ma to Ga). The
    formation event is NOT an observable Sentinel-era change, so this is not a change
    dataset: change_time=null and a representative recent 1-year window in the Sentinel
    era (2020). The landform is present throughout.
  * Not used as classes: age, target rock, bolide type, exposed/drilled -- none of these
    subsurface/geological attributes are inferable from optical/SAR imagery at 10 m, so
    only present-day structure presence is labeled (single class).

Analogous to the accepted collapse_caldera_database_ccdb presence dataset (km-scale
volcanic collapse depressions), but here we encode the crater footprint (a coherent
positive tile), not a 1x1 point, because the retained structures are all >= 3 km.

Classes (presence-only; no background/negative class):
  0 impact_structure   <- interior of a confirmed EID structure with diameter >= 3 km

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.earth_impact_database
Idempotent: existing raw HTML pages and locations/{id}.tif are skipped on re-run.
"""

import argparse
import html
import json
import multiprocessing
import re
import urllib.parse
import urllib.request
from typing import Any

import shapely
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.rasterize import rasterize_shapes

SLUG = "earth_impact_database"
NAME = "Earth Impact Database"
BASE = "http://www.passc.net/EarthImpactDatabase/New%20website_05-2018/"
INDEX_PAGE = "Namesort.html"
UA = {"User-Agent": "Mozilla/5.0"}

MIN_DIAMETER_KM = (
    3.0  # see module docstring: tile guaranteed inside structure at this size
)
TILE = 64  # 64 px * 10 m = 640 m output tile.
STATIC_YEAR = 2020  # representative Sentinel-era year for these persistent landforms.

CLASS_IMPACT = 0
CLASSES = [
    (
        CLASS_IMPACT,
        "impact_structure",
        "Interior of a confirmed terrestrial impact structure (meteorite/asteroid/comet "
        "crater or eroded remnant) from the Earth Impact Database, filtered to diameter "
        ">= 3 km. A persistent, (roughly) circular landform (rim, annular structure, "
        "and/or central uplift) resolvable at 10-30 m. The impact event is geological "
        "(ages Ma-Ga) and is NOT treated as an observable change; only present-day "
        "landform presence is labeled.",
    ),
]

# Nav/section links on the sorted-index page that are NOT per-structure pages.
_MENU = {
    "Index.html",
    "World.html",
    "Namesort.html",
    "Diametersort.html",
    "Agesort.html",
    "NorthAmerica.html",
    "SouthAmerica.html",
    "Europe.html",
    "AsiaRussia.html",
    "Africa.html",
    "Australia.html",
}


def _fetch(url: str, timeout: float = 60.0) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _crater_hrefs() -> list[str]:
    """Fetch the sorted-by-name index and return the per-structure page filenames."""
    idx = _fetch(BASE + INDEX_PAGE).decode("utf-8", "replace")
    links = re.findall(r'href="([^"]*?)"[^>]*>([^<]+)</a>', idx)
    hrefs = {
        u
        for u, _n in links
        if u.endswith(".html") and "http" not in u and u not in _MENU
    }
    return sorted(hrefs)


def _parse_dms(s: str) -> float | None:
    """Parse an EID DMS coordinate like ``N 51° 23'`` or ``E 78°8' 3.1"`` to signed deg."""
    s = html.unescape(s).replace("\xa0", " ").strip()
    m = re.match(
        r"^([NSEW])\s*([\d.]+)\s*[°ºd]?\s*(?:([\d.]+)\s*['’]?)?\s*(?:([\d.]+)\s*[\"”]?)?",
        s,
    )
    if not m:
        return None
    hemi = m.group(1)
    val = (
        float(m.group(2)) + float(m.group(3) or 0) / 60 + float(m.group(4) or 0) / 3600
    )
    return -val if hemi in ("S", "W") else val


def _parse_diameter(s: str) -> float | None:
    s = html.unescape(s).replace("~", "").replace("<", "").replace(">", "").strip()
    nums = re.findall(r"[\d.]+", s)
    return float(nums[0]) if nums else None


def _parse_page(page_html: str) -> dict[str, Any] | None:
    """Parse a per-structure page's data table into a flat record (or None)."""
    i = page_html.find("Latitude")
    if i < 0:
        return None
    seg = page_html[i:]
    seg = seg[seg.find("</tr>") :]  # skip the header row
    tds = re.findall(r"<td[^>]*>(.*?)</td>", seg, re.S)
    tds = [
        html.unescape(re.sub(r"<[^>]+>", "", t)).replace("\xa0", " ").strip()
        for t in tds
    ]
    if len(tds) < 5:
        return None
    lat = _parse_dms(tds[2])
    lon = _parse_dms(tds[3])
    diam = _parse_diameter(tds[4])
    if lat is None or lon is None or diam is None:
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return {
        "name": tds[0],
        "location": tds[1],
        "lat": lat,
        "lon": lon,
        "diameter_km": diam,
        "age": tds[5] if len(tds) > 5 else None,
    }


def scrape_catalog() -> list[dict[str, Any]]:
    """Scrape all per-structure pages (idempotent) and return parsed records.

    Raw HTML pages are cached under raw/{slug}/pages/; a parsed catalog.json is written
    to raw/{slug}/ for provenance.
    """
    raw = io.raw_dir(SLUG)
    pages = raw / "pages"
    pages.mkdir(parents=True, exist_ok=True)

    hrefs = _crater_hrefs()
    print(f"index lists {len(hrefs)} per-structure pages")

    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for href in hrefs:
        dst = pages / href
        if dst.exists():
            page_html = dst.read_text(encoding="utf-8", errors="replace")
        else:
            try:
                data = _fetch(BASE + urllib.parse.quote(href))
            except Exception as e:  # noqa: BLE001
                missing.append(f"{href}: {e!r}")
                continue
            dst.write_bytes(data)
            page_html = data.decode("utf-8", "replace")
        rec = _parse_page(page_html)
        if rec is None:
            missing.append(f"{href}: unparseable table")
            continue
        rec["source_id"] = href[:-5]  # drop .html
        records.append(rec)

    print(f"parsed {len(records)} structures; {len(missing)} unavailable/unparseable")
    for m in missing:
        print("  skip", m)

    with (raw / "catalog.json").open("w") as f:
        json.dump(
            {"count": len(records), "records": records, "skipped": missing}, f, indent=2
        )
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "Earth Impact Database (EID), Planetary and Space Science Centre, "
            "University of New Brunswick.\n"
            f"Scraped from {BASE}{INDEX_PAGE} and per-structure pages.\n"
            "License: free scholarly use (not-for-profit scientific resource).\n"
        )
    return records


def _write_one(rec: dict[str, Any]) -> str | None:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return "skip"

    lon, lat = rec["lon"], rec["lat"]
    proj, col, row = io.lonlat_to_utm_pixel(lon, lat)
    bounds = io.centered_bounds(col, row, TILE, TILE)

    # Circular footprint (radius = diameter/2) in the tile's pixel space, centered on the
    # structure. Interior = class 0; outside-circle within tile = nodata (255). Retained
    # structures are all >= 3 km (>= 150 px radius) so the circle covers the whole tile.
    radius_px = rec["diameter_km"] * 1000.0 / io.RESOLUTION / 2.0
    circle = shapely.Point(col + 0.5, row + 0.5).buffer(radius_px)
    label = rasterize_shapes(
        [(circle, CLASS_IMPACT)],
        bounds,
        fill=io.CLASS_NODATA,
        dtype="uint8",
        all_touched=True,
    )[0]

    import numpy as np

    present = sorted(int(v) for v in np.unique(label) if int(v) != io.CLASS_NODATA)
    if not present:
        return None

    io.write_label_geotiff(SLUG, sample_id, label, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(STATIC_YEAR),
        change_time=None,
        source_id=rec["source_id"],
        classes_present=present,
    )
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    records = scrape_catalog()

    kept = [r for r in records if r["diameter_km"] >= MIN_DIAMETER_KM]
    kept.sort(key=lambda r: r["source_id"])
    for j, r in enumerate(kept):
        r["sample_id"] = f"{j:06d}"
    print(
        f"kept {len(kept)}/{len(records)} structures with diameter >= "
        f"{MIN_DIAMETER_KM} km"
    )

    io.check_disk()

    n_ok = 0
    with multiprocessing.Pool(min(args.workers, max(1, len(kept)))) as p:
        for res in star_imap_unordered(p, _write_one, [dict(rec=r) for r in kept]):
            if res is not None:
                n_ok += 1
    n_written = len(list(io.locations_dir(SLUG).glob("*.tif")))
    print(f"tiles ok this run: {n_ok}; total tif on disk: {n_written}")

    diams = [r["diameter_km"] for r in kept]
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Earth Impact Database, PASSC / University of New Brunswick",
            "license": "free scholarly use",
            "provenance": {
                "url": "http://www.passc.net/EarthImpactDatabase/",
                "have_locally": False,
                "annotation_method": (
                    "manual expert compilation; structures confirmed via shock-metamorphic "
                    "evidence (PDFs, shatter cones, high-pressure phases)"
                ),
                "attribution": (
                    "Earth Impact Database, Planetary and Space Science Centre, University "
                    "of New Brunswick, Canada (managed by J. Spray)."
                ),
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": cid, "name": name, "description": desc}
                for cid, name, desc in CLASSES
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": n_written,
            "class_counts": {"impact_structure": n_written},
            "diameter_cutoff_km": MIN_DIAMETER_KM,
            "diameter_stats_km": {
                "min": min(diams) if diams else None,
                "max": max(diams) if diams else None,
                "count": len(diams),
            },
            "notes": (
                "Confirmed terrestrial impact structures scraped from the Earth Impact "
                "Database (PASSC/UNB, 190 confirmed; 197 per-structure pages parseable, "
                "1 page (Rio Cuarto) 404). Filtered to diameter >= 3 km -> 149 structures. "
                "The 3 km cutoff is derived from the encoding + coordinate precision: EID "
                "coordinates are arc-minute (worst-case +/- 0.93 km lat); a 640 m label "
                "tile's farthest pixel is 0.93 + 0.45 (tile half-diagonal) = 1.38 km from "
                "the true center, so diameter >= 2.77 km (rounded to 3 km) guarantees the "
                "tile lies inside the structure. Each structure -> one 64x64 UTM tile at "
                "10 m centered on the point, with a circular footprint (radius = "
                "diameter/2) rasterized as class 0 (impact_structure) over a 255 (nodata) "
                "background; all retained structures are >= 3 km so the circle fills the "
                "whole tile (a coherent 640 m patch of impact-structure surface). "
                "POSITIVE-ONLY: no fabricated background/negative class (assembly adds "
                "negatives, spec 5). Persistent landform (ages Ma-Ga) -> NOT a change "
                "dataset: change_time=null, static 2020 1-year window. Age/target-rock/"
                "bolide-type/exposed flags are not observable at 10 m and are not used as "
                "classes. Some retained structures are deeply eroded or buried (weak "
                "surface expression) -- kept as valid presence labels per spec 5."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=n_written
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()

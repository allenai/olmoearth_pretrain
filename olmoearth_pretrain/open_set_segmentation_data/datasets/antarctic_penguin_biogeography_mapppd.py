"""Process the Antarctic Penguin Biogeography / MAPPPD database into open-set-segmentation
sparse-point labels.

Source: the Antarctic Penguin Biogeography Project "Count data" (MAPPPD, www.penguinmap.com),
published as a Darwin Core Archive on the SCAR/AADC IPT
(https://ipt.biodiversity.aq/resource?r=mapppd_count_data) and on GBIF as dataset
f7c30fac-cf80-471f-8343-4ec5d8594661. The DwC-A has an Event core (one survey at a breeding
site on a date, with WGS84 lon/lat) and an Occurrence extension (one penguin species observed
in that event, with a count). Six species: Adelie, chinstrap, gentoo, emperor, macaroni, king.

Penguin breeding colonies leave persistent guano stains detectable in Landsat / Sentinel-2 at
10-30 m, so a species presence at a colony is a usable species-presence label (class = species).
Colonies are persistent, so we treat presence as a static label and anchor each point on a
1-year Sentinel-era window (the survey year). We keep only surveys dated 2016+ (Sentinel era;
all records are dated) with occurrenceStatus=present, dedupe to one point per (site, species)
keeping the most recent survey year. Sparse points -> one dataset-wide points.geojson (spec 2a).
"""

import argparse
import collections
import csv
from typing import Any

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "antarctic_penguin_biogeography_mapppd"
NAME = "Antarctic Penguin Biogeography / MAPPPD"
DWCA_URL = "https://ipt.biodiversity.aq/archive.do?r=mapppd_count_data"
GBIF_DATASET = "f7c30fac-cf80-471f-8343-4ec5d8594661"
PER_CLASS = 1000
MIN_YEAR = 2016  # Sentinel era

# Manifest class order -> id. Map from the DwC vernacularName / scientificName.
CLASSES = [
    (
        "Adelie",
        "Pygoscelis adeliae",
        "Adelie penguin (Pygoscelis adeliae) breeding colony presence; a circumpolar "
        "pack-ice species nesting on ice-free coastal terrain.",
    ),
    (
        "chinstrap",
        "Pygoscelis antarctica",
        "Chinstrap penguin (Pygoscelis antarcticus) breeding colony presence; nests on "
        "ice-free slopes, concentrated on the Antarctic Peninsula and Scotia Arc.",
    ),
    (
        "gentoo",
        "Pygoscelis papua",
        "Gentoo penguin (Pygoscelis papua) breeding colony presence; nests on ice-free "
        "ground on the Peninsula and sub-Antarctic islands.",
    ),
    (
        "emperor",
        "Aptenodytes forsteri",
        "Emperor penguin (Aptenodytes forsteri) breeding colony presence; breeds on "
        "fast ice, largely detected from satellite guano staining.",
    ),
    (
        "macaroni",
        "Eudyptes chrysolophus",
        "Macaroni penguin (Eudyptes chrysolophus) breeding colony presence; crested "
        "penguin of the Scotia Arc and sub-Antarctic.",
    ),
    (
        "king penguin",
        "Aptenodytes patagonicus",
        "King penguin (Aptenodytes patagonicus) breeding colony presence; sub-Antarctic "
        "island breeder, rare south of 60S.",
    ),
]
NAME_TO_ID = {sci: i for i, (_n, sci, _d) in enumerate(CLASSES)}
# vernacularName -> scientificName, so we can key off either column robustly.
VERNACULAR_TO_SCI = {
    "adelie penguin": "Pygoscelis adeliae",
    "chinstrap penguin": "Pygoscelis antarctica",
    "gentoo penguin": "Pygoscelis papua",
    "emperor penguin": "Aptenodytes forsteri",
    "macaroni penguin": "Eudyptes chrysolophus",
    "king penguin": "Aptenodytes patagonicus",
}


def download_source() -> Any:
    """Download + extract the MAPPPD DwC-A into raw/{slug}/dwca (idempotent)."""
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    zip_path = raw / "mapppd_dwca.zip"
    download.download_http(
        DWCA_URL, zip_path, headers={"User-Agent": "Mozilla/5.0"}, timeout=180
    )
    dwca = download.extract_zip(zip_path, raw / "dwca")
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "Antarctic Penguin Biogeography Project / MAPPPD count data (Darwin Core Archive).\n"
            f"IPT: {DWCA_URL}\n"
            f"GBIF dataset: https://www.gbif.org/dataset/{GBIF_DATASET}\n"
            "Portal: https://www.penguinmap.com\n"
        )
    return dwca


def _class_id(occ: dict[str, str]) -> int | None:
    sci = occ.get("scientificName", "").strip()
    if sci in NAME_TO_ID:
        return NAME_TO_ID[sci]
    vern = occ.get("vernacularName", "").strip().lower()
    sci = VERNACULAR_TO_SCI.get(vern)
    return NAME_TO_ID.get(sci) if sci else None


def build_records(dwca_path: Any) -> list[dict[str, Any]]:
    """Parse the DwC-A into deduped (site, species) presence points, 2016+ only.

    One record per (locationID, species) keeping the most recent Sentinel-era survey.
    """
    events: dict[str, dict[str, str]] = {}
    with (dwca_path / "event.txt").open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            events[row["eventID"]] = row

    best: dict[tuple[str, int], dict[str, Any]] = {}
    with (dwca_path / "occurrence.txt").open() as f:
        for occ in csv.DictReader(f, delimiter="\t"):
            if occ.get("occurrenceStatus", "").strip() != "present":
                continue
            cid = _class_id(occ)
            if cid is None:
                continue
            ev = events.get(occ["eventID"])
            if ev is None:
                continue
            year = ev.get("year", "").strip()
            lat = ev.get("decimalLatitude", "").strip()
            lon = ev.get("decimalLongitude", "").strip()
            if not year or not lat or not lon:
                continue
            year = int(year)
            if year < MIN_YEAR:
                continue
            loc = ev.get("locationID", "").strip() or ev.get("eventID")
            key = (loc, cid)
            prev = best.get(key)
            if prev is None or year > prev["year"]:
                unc = ev.get("coordinateUncertaintyInMeters", "").strip()
                best[key] = {
                    "loc": loc,
                    "label": cid,
                    "year": year,
                    "lon": float(lon),
                    "lat": float(lat),
                    "locality": ev.get("locality", "").strip(),
                    "coord_uncertainty_m": float(unc) if unc else None,
                    "source_id": f"{loc}:{VERNACULAR_TO_SCI.get(occ.get('vernacularName', '').strip().lower(), occ.get('scientificName', ''))}:{year}",
                }
    return list(best.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    dwca = download_source()
    recs = build_records(dwca)
    print(f"built {len(recs)} deduped (site,species) presence points 2016+")

    selected = balance_by_class(recs, "label", per_class=PER_CLASS)
    print(f"selected {len(selected)} (<= {PER_CLASS}/class, 25k cap)")

    points = []
    for i, r in enumerate(sorted(selected, key=lambda x: (x["label"], x["loc"]))):
        p = {
            "id": f"{i:06d}",
            "lon": r["lon"],
            "lat": r["lat"],
            "label": r["label"],
            "time_range": io.year_range(r["year"]),
            "change_time": None,
            "source_id": r["source_id"],
            "coord_uncertainty_m": r["coord_uncertainty_m"],
            "locality": r["locality"],
        }
        points.append(p)
    io.write_points_table(SLUG, "classification", points)

    counts = collections.Counter(r["label"] for r in selected)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "GBIF / penguinmap.com (Antarctic Penguin Biogeography Project / MAPPPD)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://www.penguinmap.com",
                "dwca_url": DWCA_URL,
                "gbif_dataset": f"https://www.gbif.org/dataset/{GBIF_DATASET}",
                "have_locally": False,
                "annotation_method": "field counts + guano-stain photointerpretation (satellite/aerial)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, _sci, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {
                name: counts.get(i, 0) for i, (name, _s, _d) in enumerate(CLASSES)
            },
            "notes": (
                "Sparse species-presence points (1x1) at Antarctic penguin breeding "
                "colonies; class = species. Kept only surveys dated 2016+ (Sentinel era; "
                "all records are dated), occurrenceStatus=present, deduped to one point per "
                "(colony site, species) keeping the most recent survey year. Colonies are "
                "persistent, so each point is a static label with a 1-year window anchored on "
                "its survey year (change_time=null). Coordinate uncertainty (site gazetteer "
                "centroids) is recorded per point in coord_uncertainty_m (median ~1.15 km)."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG,
        "completed",
        task_type="classification",
        num_samples=len(selected),
        notes=(
            "Sparse penguin species-presence points from MAPPPD DwC-A; 6 species classes; "
            "kept only 2016+ present surveys, deduped to one point per (site,species), "
            "static 1-year window on survey year. Caveat: site-centroid coords, median "
            "~1.15 km uncertainty (recorded per-point)."
        ),
    )
    print("done")


if __name__ == "__main__":
    main()

"""Process EDDMapS-style invasive-species occurrences into a point-table label set.

Provenance / triage note
-------------------------
The manifest dataset is **EDDMapS Invasive Species** (University of Georgia / Bugwood),
a georeferenced database of invasive-plant occurrences across the US & Canada. Its bulk
data is licensed "viewable; bulk by request" — bulk download requires an
account/request approval that we do not have, and there is no EDDMapS credential in
``.env``. Per the task spec (§8) we therefore fall back to the
**open GBIF mirror** of invasive-plant occurrences: we take the set of introduced/invasive
plant taxa registered for the US & Canada in the GBIF **GRIIS** checklists (Global Register
of Introduced and Invasive Species — the same registry EDDMapS-tracked species belong to)
and pull their georeferenced occurrences (lon/lat + species + date) from the open GBIF
Occurrence API. This reproduces the same *signal* (where invasive plants were observed on
the ground, US & Canada, Sentinel era) without EDDMapS' gated bulk export. GBIF ingests
several EDDMapS-network/US invasive datasets (e.g. IPAMS, iNaturalist, USGS) so there is
real overlap, but this is a GBIF-sourced proxy, not the raw EDDMapS export — documented in
the summary and metadata.

Label type: **sparse points** (each occurrence is one on-the-ground observation at a
single 10 m pixel). Per spec §2a we therefore write ONE dataset-wide GeoJSON point table
``points.geojson`` (not per-point GeoTIFFs). Class = species; we keep the **top 254 species
by observation frequency** (uint8 class cap) and balance to the 25k per-dataset cap.

Observability caveat: a single invasive plant is usually sub-10 m and not resolvable from
Sentinel/Landsat, but dense infestations (kudzu mats, water-hyacinth rafts, Spartina
meadows, cheatgrass-dominated range, tamarisk stands) can be. We keep species as classes
per §5 and record the caveat; downstream assembly supplies negatives and drops too-rare
classes.

Run: ``python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.eddmaps_invasive_species``
Idempotent: caches every network result under ``raw/{slug}/`` and skips work already done.
"""

import argparse
import json
import multiprocessing
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from typing import Any

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    MAX_SAMPLES_PER_DATASET,
    balance_by_class,
)

SLUG = "eddmaps_invasive_species"
GBIF = "https://api.gbif.org/v1"

# GRIIS (Global Register of Introduced and Invasive Species) checklists on GBIF that
# define the invasive/introduced taxon universe for our region (US & Canada).
GRIIS_CHECKLISTS = {
    "us_contiguous": "32ad19ed-6b89-447a-9242-795c0897f345",
    "us_alaska": "7b091962-fdb2-49eb-9bfb-7d66561f1a8a",
    "us_hawaii": "6baf6a53-c106-40fb-bbde-f6d4e4051513",
    "canada": "b95e74e0-b772-430c-a729-9d56ce0182e2",
}
PLANT_KINGDOM_KEY = 6  # GBIF backbone kingdomKey for Plantae.

# Manifest-named EDDMapS flagship invasive plants. These are the classic dense-infestation
# species the task highlights as the most *observable* at 10 m (kudzu mats, tamarisk stands,
# water-hyacinth rafts, Spartina/cordgrass meadows) — the highest-value segmentation signal.
# Some are observed less often on GBIF than ubiquitous naturalized herbs, so they can fall
# just below the strict top-254-by-frequency cut. We force-include them (displacing the
# lowest-frequency otherwise-selected classes) so the manifest's named classes are present;
# documented as a judgment call in the summary. Resolved to backbone keys at runtime.
PRIORITY_SPECIES = {
    "Pueraria montana": "kudzu",
    "Tamarix ramosissima": "tamarisk / saltcedar",
    "Pontederia crassipes": "water hyacinth (syn. Eichhornia crassipes)",
    "Sporobolus alterniflorus": "smooth cordgrass (Spartina alterniflora)",
}
COUNTRIES = ["US", "CA"]
YEAR_MIN, YEAR_MAX = 2016, 2026  # Sentinel era; manifest time_range.
MAX_CLASSES = 254  # uint8 class cap (ids 0..253; 255 = nodata).
PER_SPECIES_FETCH = 200  # occurrences pulled per selected species.
PER_CLASS = 1000  # balance target before the 25k total cap kicks in.
FACET_PAGE = 1000  # speciesKey facet page size.
FACET_MAX_DEPTH = 20000  # how deep to scan the frequency-ranked plant facet.


# --------------------------------------------------------------------------- HTTP


def _get(path: str, params: list[tuple[str, str]], retries: int = 8) -> dict[str, Any]:
    """GET a GBIF API endpoint (query as an ordered param list to allow repeats).

    GBIF rate-limits anonymous clients (HTTP 429); on 429 we honor ``Retry-After`` when
    present and otherwise back off exponentially (longer than for generic errors), so a
    modest worker pool can keep going without hammering the API.
    """
    url = f"{GBIF}/{path}?" + urllib.parse.urlencode(params)
    last = ""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "olmoearth-osseg/1.0"}
            )
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            last = repr(e)
            if e.code == 429:
                ra = e.headers.get("Retry-After")
                delay = (
                    float(ra) if (ra and ra.isdigit()) else min(60.0, 5.0 * 2**attempt)
                )
                time.sleep(delay)
            else:
                time.sleep(2.0 * (attempt + 1))
        except Exception as e:  # noqa: BLE001 - retry any transient network error
            last = repr(e)
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"GBIF GET failed after {retries}: {url}: {last}")


def _occ_filter_params() -> list[tuple[str, str]]:
    p = [("country", c) for c in COUNTRIES]
    p += [
        ("hasCoordinate", "true"),
        ("hasGeospatialIssue", "false"),
        ("year", f"{YEAR_MIN},{YEAR_MAX}"),
    ]
    return p


# ----------------------------------------------------------------- invasive taxa


def load_invasive_plant_species() -> dict[int, dict[str, Any]]:
    """Return {backbone_species_key -> {canonicalName, scientificName, checklists}}.

    Pages every GRIIS checklist's name usages, keeping Plantae species that resolve to a
    GBIF backbone key (nubKey). Cached to raw/{slug}/checklist_species.json.
    """
    raw = io.raw_dir(SLUG)
    cache = raw / "checklist_species.json"
    if cache.exists():
        with cache.open() as f:
            return {int(k): v for k, v in json.load(f).items()}

    species: dict[int, dict[str, Any]] = {}
    for cl_name, key in GRIIS_CHECKLISTS.items():
        offset = 0
        while True:
            d = _get(
                "species/search",
                [
                    ("datasetKey", key),
                    ("rank", "SPECIES"),
                    ("limit", "1000"),
                    ("offset", str(offset)),
                ],
            )
            results = d.get("results", [])
            for r in results:
                if r.get("kingdom") != "Plantae":
                    continue
                nub = r.get("nubKey")
                if not nub:
                    continue
                nub = int(nub)
                entry = species.setdefault(
                    nub,
                    {
                        "canonicalName": r.get("canonicalName")
                        or r.get("scientificName"),
                        "scientificName": r.get("scientificName"),
                        "checklists": [],
                    },
                )
                if cl_name not in entry["checklists"]:
                    entry["checklists"].append(cl_name)
            offset += len(results)
            if d.get("endOfRecords") or not results:
                break
        print(f"  {cl_name}: cumulative plant species so far = {len(species)}")

    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "checklist_species.json.tmp").open("w") as f:
        json.dump({str(k): v for k, v in species.items()}, f)
    (raw / "checklist_species.json.tmp").rename(cache)
    return species


def rank_top_species(invasive: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank invasive plant species by US+CA 2016+ occurrence count via the plant facet.

    The GBIF speciesKey facet returns species in strictly descending occurrence count, so
    we scan it top-down, keep hits that are in ``invasive``, and stop once we have
    MAX_CLASSES — those are exactly the most-frequent invasive species. Cached.
    """
    raw = io.raw_dir(SLUG)
    cache = raw / "top_species.json"
    if cache.exists():
        with cache.open() as f:
            return json.load(f)

    top: list[dict[str, Any]] = []
    scanned = 0
    offset = 0
    while offset < FACET_MAX_DEPTH and len(top) < MAX_CLASSES:
        d = _get(
            "occurrence/search",
            _occ_filter_params()
            + [
                ("taxonKey", str(PLANT_KINGDOM_KEY)),
                ("limit", "0"),
                ("facet", "speciesKey"),
                ("facetLimit", str(FACET_PAGE)),
                ("facetOffset", str(offset)),
            ],
        )
        counts = d["facets"][0]["counts"] if d.get("facets") else []
        if not counts:
            break
        for c in counts:
            scanned += 1
            skey = int(c["name"])
            if skey in invasive:
                inv = invasive[skey]
                top.append(
                    {
                        "species_key": skey,
                        "canonicalName": inv["canonicalName"],
                        "scientificName": inv["scientificName"],
                        "checklists": inv["checklists"],
                        "occ_count": int(c["count"]),
                    }
                )
                if len(top) >= MAX_CLASSES:
                    break
        offset += len(counts)
    print(
        f"  scanned {scanned} ranked plant species; selected {len(top)} invasive by frequency"
    )

    top = _inject_priority_species(top, invasive)

    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "top_species.json.tmp").open("w") as f:
        json.dump(top, f)
    (raw / "top_species.json.tmp").rename(cache)
    return top


def _species_occ_count(species_key: int) -> int:
    d = _get(
        "occurrence/search",
        _occ_filter_params() + [("taxonKey", str(species_key)), ("limit", "0")],
    )
    return int(d.get("count", 0))


def _inject_priority_species(
    top: list[dict[str, Any]], invasive: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Force-include manifest-named flagship invasives, keeping <=MAX_CLASSES total.

    Each flagship name is resolved to a GBIF backbone species key; if not already among the
    frequency-selected classes it is appended (with its own US+CA occurrence count and a
    ``priority`` flag) and the lowest-frequency non-priority class is dropped to hold the
    254-class cap. Result is re-sorted by descending occurrence count.
    """
    have = {t["species_key"] for t in top}
    added = []
    for sci, common in PRIORITY_SPECIES.items():
        m = _get("species/match", [("name", sci)])
        skey = m.get("usageKey")
        if not skey:
            print(f"    priority '{sci}' did not match a backbone key; skipping")
            continue
        skey = int(skey)
        if skey in have:
            # Already selected by frequency: just tag it as a flagship for the summary.
            for t in top:
                if t["species_key"] == skey:
                    t["priority"] = True
                    t["common_name"] = common
            continue
        cnt = _species_occ_count(skey)
        inv = invasive.get(skey, {})
        added.append(
            {
                "species_key": skey,
                "canonicalName": (
                    inv.get("canonicalName") or m.get("canonicalName") or sci
                ),
                "scientificName": (
                    inv.get("scientificName") or m.get("scientificName") or sci
                ),
                "checklists": inv.get("checklists", []),
                "occ_count": cnt,
                "priority": True,
                "common_name": common,
                "in_griis": skey in invasive,
            }
        )
        have.add(skey)
        print(f"    + flagship {sci} ({common}): key={skey}, US+CA 2016+ count={cnt}")

    if added:
        # Drop the lowest-frequency NON-priority classes to make room, then re-rank.
        non_priority = [t for t in top if not t.get("priority")]
        priority = [t for t in top if t.get("priority")]
        keep_non = len(top) + len(added) - MAX_CLASSES
        non_priority.sort(key=lambda t: t["occ_count"], reverse=True)
        if keep_non > 0:
            dropped = non_priority[len(non_priority) - keep_non :]
            print(
                f"    displaced {len(dropped)} lowest-frequency classes "
                f"(counts {dropped[-1]['occ_count']}..{dropped[0]['occ_count']}) for flagships"
            )
            non_priority = non_priority[: len(non_priority) - keep_non]
        top = priority + non_priority + added
    top.sort(key=lambda t: t["occ_count"], reverse=True)
    return top[:MAX_CLASSES]


# --------------------------------------------------------------- occurrence pull


def _fetch_one_species(species_key: int) -> tuple[int, list[dict[str, Any]]]:
    """Fetch up to PER_SPECIES_FETCH georeferenced occurrences for one species."""
    d = _get(
        "occurrence/search",
        _occ_filter_params()
        + [
            ("taxonKey", str(species_key)),
            ("limit", str(PER_SPECIES_FETCH)),
            ("offset", "0"),
        ],
    )
    recs = []
    for r in d.get("results", []):
        lon, lat = r.get("decimalLongitude"), r.get("decimalLatitude")
        year = r.get("year")
        if lon is None or lat is None or year is None:
            continue
        if not (YEAR_MIN <= int(year) <= YEAR_MAX):
            continue
        recs.append(
            {
                "lon": float(lon),
                "lat": float(lat),
                "year": int(year),
                "gbif_id": r.get("key"),
                "dataset_key": r.get("datasetKey"),
            }
        )
    return species_key, recs


def fetch_occurrences(
    top: list[dict[str, Any]], workers: int
) -> dict[int, list[dict[str, Any]]]:
    """Parallel-fetch occurrences for every selected species. Cached."""
    raw = io.raw_dir(SLUG)
    cache = raw / "occurrences.json"
    if cache.exists():
        with cache.open() as f:
            return {int(k): v for k, v in json.load(f).items()}

    keys = [t["species_key"] for t in top]
    out: dict[int, list[dict[str, Any]]] = {}
    with multiprocessing.Pool(workers) as p:
        for skey, recs in p.imap_unordered(_fetch_one_species, keys, chunksize=1):
            out[skey] = recs
    total = sum(len(v) for v in out.values())
    print(f"  fetched {total} occurrences across {len(out)} species")

    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "occurrences.json.tmp").open("w") as f:
        json.dump({str(k): v for k, v in out.items()}, f)
    (raw / "occurrences.json.tmp").rename(cache)
    return out


# ----------------------------------------------------------------------- driver


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "EDDMapS bulk data is gated (license: viewable; bulk by request; no EDDMapS "
            "credential in .env). Open GBIF mirror used instead:\n"
            "  invasive/introduced plant taxa from GRIIS checklists (US contiguous, "
            "Alaska, Hawaii, Canada), occurrences from the GBIF Occurrence API.\n"
            f"  GRIIS checklists: {json.dumps(GRIIS_CHECKLISTS)}\n"
            f"  Occurrence filter: country in {COUNTRIES}, hasCoordinate=true, "
            f"hasGeospatialIssue=false, year {YEAR_MIN}-{YEAR_MAX}.\n"
        )

    print("1) loading invasive plant taxa from GRIIS checklists ...")
    invasive = load_invasive_plant_species()
    print(f"   {len(invasive)} unique invasive/introduced plant species (US & Canada)")

    print("2) ranking by GBIF occurrence frequency (US+CA, 2016+) ...")
    top = rank_top_species(invasive)
    if not top:
        manifest.write_registry_entry(
            SLUG,
            "temporary_failure",
            notes="GBIF facet returned no invasive plant species; retry (API issue).",
        )
        raise SystemExit("no species selected; recorded temporary_failure")

    print("3) fetching occurrences per selected species ...")
    occ = fetch_occurrences(top, args.workers)

    # Assign class ids by descending frequency (top[] is already frequency-ordered).
    classes = []
    records: list[dict[str, Any]] = []
    for cid, t in enumerate(top):
        skey = t["species_key"]
        recs = occ.get(skey, [])
        if not recs:
            continue
        classes.append(t)
        for r in recs:
            records.append(
                {
                    "lon": r["lon"],
                    "lat": r["lat"],
                    "year": r["year"],
                    "label": cid,
                    "species_key": skey,
                    "source_id": f"gbif:{r['gbif_id']}",
                    "gbif_dataset_key": r["dataset_key"],
                }
            )
    # Re-map class ids to be contiguous over species that actually yielded records.
    kept_keys = [c["species_key"] for c in classes]
    remap = {t["species_key"]: i for i, t in enumerate(classes)}
    for rec in records:
        rec["label"] = remap[rec["species_key"]]

    print(f"   {len(records)} raw occurrence records across {len(classes)} species")

    selected = balance_by_class(
        records, "label", per_class=PER_CLASS, total_cap=MAX_SAMPLES_PER_DATASET
    )
    print(f"   balanced to {len(selected)} samples (<=25k, class-balanced)")

    points = []
    for i, r in enumerate(sorted(selected, key=lambda x: (x["label"], x["source_id"]))):
        points.append(
            {
                "id": f"{i:06d}",
                "lon": r["lon"],
                "lat": r["lat"],
                "label": r["label"],
                "time_range": io.year_range(r["year"]),
                "source_id": r["source_id"],
                "species_key": r["species_key"],
                "gbif_dataset_key": r["gbif_dataset_key"],
            }
        )
    io.write_points_table(SLUG, "classification", points)

    counts = Counter(p["label"] for p in points)
    class_meta = []
    for cid, c in enumerate(classes):
        cls = c.get("checklists") or []
        if cls:
            listing = f"introduced/invasive plant listed in GRIIS for {', '.join(cls)}"
        else:
            listing = (
                "EDDMapS-tracked flagship invasive plant (not in the GRIIS US/CA lists)"
            )
        common = c.get("common_name")
        flagship = " Manifest-named flagship species." if c.get("priority") else ""
        common_str = f" (common name: {common})" if common else ""
        class_meta.append(
            {
                "id": cid,
                "name": c["canonicalName"],
                "description": (
                    f"{c['scientificName']}{common_str} — {listing}; GBIF backbone "
                    f"speciesKey {c['species_key']}. {c['occ_count']} georeferenced US+CA "
                    f"occurrences (2016+) available.{flagship}"
                ),
            }
        )

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "EDDMapS Invasive Species",
            "task_type": "classification",
            "source": "GBIF (open mirror of invasive-plant occurrences; EDDMapS bulk gated)",
            "license": "GBIF occurrences under source terms (mostly CC-BY / CC0); "
            "EDDMapS bulk itself is 'viewable; bulk by request'.",
            "provenance": {
                "url": "https://www.eddmaps.org/",
                "gbif_api": "https://api.gbif.org/v1/occurrence/search",
                "griis_checklists": GRIIS_CHECKLISTS,
                "have_locally": False,
                "annotation_method": "field observation (crowd-sourced + agency verified reports), via GBIF",
                "note": "EDDMapS bulk download requires account/request approval (no credential "
                "in .env); substituted open GBIF occurrences of GRIIS-listed invasive/introduced "
                "plant taxa for US & Canada. Proxy for EDDMapS, not the raw EDDMapS export.",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": class_meta,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(points),
            "num_classes": len(classes),
            "class_counts": {
                str(cid): counts.get(cid, 0) for cid in range(len(classes))
            },
            "notes": (
                "Sparse point segmentation (1x1). Class = invasive plant species; kept top "
                f"{len(classes)} species by GBIF US+CA occurrence frequency (uint8 254-class "
                "cap). Each point is one field observation; time_range = 1-year window on the "
                "observation year. Observability caveat: individual invasive plants are often "
                "sub-10 m; dense infestations (kudzu, water hyacinth, Spartina, cheatgrass, "
                "tamarisk) are observable. Negatives + rare-class filtering handled at assembly."
            ),
        },
    )

    manifest.write_registry_entry(
        SLUG,
        "completed",
        task_type="classification",
        num_samples=len(points),
        notes=(
            f"GBIF open mirror of EDDMapS-style invasive-plant occurrences (bulk EDDMapS "
            f"gated). {len(classes)} species classes (top-254 by freq), {len(points)} points, "
            f"US & Canada, {YEAR_MIN}-{YEAR_MAX}."
        ),
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()

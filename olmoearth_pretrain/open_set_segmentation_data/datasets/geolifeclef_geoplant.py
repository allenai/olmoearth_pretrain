"""Process GeoLifeCLEF / GeoPlant presence-only occurrences into open-set-segmentation
label patches.

Source: GeoPlant (GeoLifeCLEF 2024, Pl@ntNet-INRIA / NeurIPS D&B). The presence-only
(PO) metadata table ``PO_metadata_train.csv`` is ~5.08M opportunistic plant-species
observations (GBIF + Pl@ntNet + iNaturalist etc.) across Europe, 2017-2021, covering
9,709 anonymized species. Each row is a single species observed at one lon/lat with a
geolocation-uncertainty radius and an observation date. This is the natural
"one class per point" fit for the sparse-point recipe (one species label per record),
unlike the presence-absence surveys (many species share one location).

Access: fully open via the Pl@ntNet Seafile mirror (no Kaggle account required).

Label semantics (documented caveat): plant-species presence at a point is only *weakly*
observable from 10-30 m S2/S1/Landsat, and the source spans ~10k species. We therefore
treat these as **weak / contextual habitat labels** (the manifest's stated intent). Two
hard constraints shape the class set:

  * The label GeoTIFF is single-band uint8 (ids 0..254, 255=nodata), so at most ~254
    distinct classes can be encoded. We keep the **top 254 species by observation
    frequency** (each has >=~1.2k observations), assigning ids 0..253 in descending
    frequency; the remaining 9,455 rarer species are dropped (listed as a count in the
    summary).
  * We cap the total near the spec's ~50k default by taking up to ``PER_CLASS`` (200)
    randomly-sampled points per kept species -> up to ~50.8k 1x1 patches.

This is a sparse-point dataset, so it is written as one dataset-wide point table
(points.json, spec 2a): one row per observation with lon/lat, label = class id, and a
1-year time range anchored on the observation year.

Reproduce: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.geolifeclef_geoplant
"""

import argparse
import multiprocessing
import re
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "geolifeclef_geoplant"
NAME = "GeoLifeCLEF / GeoPlant"

# Open Seafile mirror (no Kaggle needed).
SEAFILE_REPO = "https://lab.plantnet.org/seafile/d/59325675470447b38add"
PO_METADATA_PATH = "PresenceOnlyOccurrences/PO_metadata_train.csv"

N_CLASSES = 254  # max that fits uint8 (ids 0..253; 255 = nodata)
PER_CLASS = 200  # keeps total near the ~50k default cap (254 * 200 = 50,800)
MAX_GEO_UNCERTAINTY_M = 100.0  # drop egregiously imprecise points (~1% of records)


def _resolve_seafile_url(file_path: str) -> str:
    """Resolve a Seafile share path to a direct raw-download URL."""
    r = requests.get(
        f"{SEAFILE_REPO}/files/?p=/{quote(file_path, safe='/')}", timeout=120
    )
    r.raise_for_status()
    m = re.search(r"rawPath: '([^']+)'", r.text)
    if not m:
        raise RuntimeError(f"could not resolve Seafile url for {file_path}")
    return m.group(1).replace("\\u002D", "-") + "?raw=1"


def _download_po_metadata() -> "io.UPath":
    """Download PO_metadata_train.csv into raw_dir (idempotent, atomic)."""
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    dst = raw / "PO_metadata_train.csv"
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "GeoLifeCLEF / GeoPlant (Pl@ntNet-INRIA, NeurIPS 2024 D&B).\n"
            f"Open Seafile mirror: {SEAFILE_REPO}\n"
            f"File: {PO_METADATA_PATH}\n"
            "GBIF extraction 2022-11-08: https://doi.org/10.15468/dl.4ysfh4\n"
        )
    if dst.exists():
        return dst
    url = _resolve_seafile_url(PO_METADATA_PATH)
    print(f"downloading {url}")
    tmp = raw / "PO_metadata_train.csv.tmp"
    with requests.get(url, timeout=1200, stream=True) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in resp.iter_content(1 << 20):
                f.write(chunk)
    tmp.rename(dst)
    return dst


def load_records() -> tuple[list[dict[str, Any]], dict[int, int], dict[int, int]]:
    """Load PO metadata, filter, keep top-N species, and return flat records.

    Returns (records, species_id_to_class_id, class_id_to_source_count).
    """
    csv_path = _download_po_metadata()
    df = pd.read_csv(
        str(csv_path),
        usecols=["lat", "lon", "year", "geoUncertaintyInM", "speciesId", "surveyId"],
    )
    n0 = len(df)
    df = df.dropna(subset=["lat", "lon", "year", "speciesId"])
    df = df[
        df["geoUncertaintyInM"].isna()
        | (df["geoUncertaintyInM"] <= MAX_GEO_UNCERTAINTY_M)
    ]
    df = df[(df["lat"].between(-90, 90)) & (df["lon"].between(-180, 180))]
    df["speciesId"] = df["speciesId"].astype(int)
    df["year"] = df["year"].astype(int)
    print(
        f"loaded {n0} rows; {len(df)} after filtering (geoUncertainty<= {MAX_GEO_UNCERTAINTY_M} m)"
    )

    counts = df["speciesId"].value_counts()
    top_species = counts.head(N_CLASSES)
    species_to_cid = {int(sid): i for i, sid in enumerate(top_species.index)}
    cid_source_count = {i: int(top_species.iloc[i]) for i in range(len(top_species))}
    print(
        f"{df['speciesId'].nunique()} species total; keeping top {len(species_to_cid)} "
        f"(min obs among kept = {int(top_species.min())}); dropping "
        f"{df['speciesId'].nunique() - len(species_to_cid)} rarer species"
    )

    df = df[df["speciesId"].isin(species_to_cid)]
    records = [
        {
            "lon": float(row.lon),
            "lat": float(row.lat),
            "class_id": species_to_cid[int(row.speciesId)],
            "species_id": int(row.speciesId),
            "year": int(row.year),
            "source_id": f"survey_{int(row.surveyId)}",
        }
        for row in df.itertuples(index=False)
    ]
    return records, species_to_cid, cid_source_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    records, species_to_cid, cid_source_count = load_records()

    selected = balance_by_class(records, "class_id", per_class=PER_CLASS)
    selected.sort(key=lambda r: (r["class_id"], r["source_id"]))
    print(
        f"selected {len(selected)} points (<= {PER_CLASS}/class over {len(species_to_cid)} classes)"
    )

    # Sparse point dataset -> one dataset-wide point table (spec 2a), not per-point tifs.
    points = [
        {
            "id": f"{i:06d}",
            "lon": r["lon"],
            "lat": r["lat"],
            "label": r["class_id"],
            "time_range": io.year_range(r["year"]),
            "source_id": r["source_id"],
        }
        for i, r in enumerate(selected)
    ]
    io.write_points_table(SLUG, "classification", points)

    from collections import Counter

    counts = Counter(r["class_id"] for r in selected)
    # Class list: id ordered by descending source frequency (id 0 = most observed).
    cid_to_species = {cid: sid for sid, cid in species_to_cid.items()}
    classes = [
        {
            "id": cid,
            "name": f"species_{cid_to_species[cid]}",
            "description": None,
            "source_species_id": cid_to_species[cid],
            "n_source_observations": cid_source_count[cid],
            "n_samples": counts.get(cid, 0),
        }
        for cid in range(len(species_to_cid))
    ]

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Kaggle / NeurIPS (Pl@ntNet-INRIA); open Seafile mirror",
            "license": "CC-BY",
            "provenance": {
                "url": "https://github.com/plantnet/GeoPlant",
                "seafile": SEAFILE_REPO,
                "have_locally": False,
                "annotation_method": "field / citizen-science (GBIF, Pl@ntNet, iNaturalist)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": classes,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "notes": (
                "Weak/contextual habitat label: plant-species presence points, written as "
                "a points.json table. Source has 9,709 species; classification is uint8 so "
                f"we cap at {N_CLASSES} classes and keep the top {len(species_to_cid)} "
                "species by observation frequency (ids 0..N-1 in descending frequency; "
                "class names are the source's anonymized integer speciesIds). Up to "
                f"{PER_CLASS} randomly-sampled points/class. Points with "
                f"geoUncertaintyInM > {MAX_GEO_UNCERTAINTY_M} dropped. 1-year time range "
                "anchored on each observation's year (2017-2021). Presence-only subset "
                "used (one species per point); presence-absence surveys not used because "
                "they place many species at one location."
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

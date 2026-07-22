"""Process IDTReeS individual tree crowns into open-set-segmentation labels.

Source: IDTReeS 2018/2020 competition data (Weinstein et al.), Zenodo record 3700197
(https://zenodo.org/records/3700197), CC-BY-4.0. Co-registered NEON RGB / LiDAR /
hyperspectral over three NEON sites; the labels are field-mapped **individual tree
crowns** delineated as polygons (``ITC/train_{SITE}.shp``) linked by ``indvdID`` to the
field table ``Field/train_data.csv`` (per-stem ``taxonID`` species code). We use ONLY the
crown geometries + the field species labels -- the multi-GB RemoteSensing imagery is not
needed (pretraining supplies its own imagery). The train release covers two sites: MLBS
(Mountain Lake Biological Station, VA, deciduous forest) and OSBS (Ordway-Swisher
Biological Station, FL, longleaf-pine flatwoods).

Triage / observability (spec 2 & 8): an individual tree crown is small -- median crown
footprint here is ~4.2 m, i.e. well under one 10 m Sentinel-2 pixel -- so per-crown
species identity is NOT directly resolvable at 10-30 m. This is the same posture as
``globalgeotree`` / ``geolifeclef_geoplant``: because the crowns sit in NATURAL forest
(NEON research sites), a point still acts as a **weak habitat/species label** (the
surrounding canopy correlates with the target), unlike the urban ``auto_arborist`` case
which was rejected (pavement-dominated pixels, no habitat proxy). So we ACCEPT as
weak sparse-point classification and let the downstream assembly step treat/filter it.

Recipe: sparse points (spec 2a/4). Each crown -> its centroid -> one Point feature with
``label`` = species class id, written to one dataset-wide ``points.geojson`` (NOT per-crown
GeoTIFFs; a crown is a sub-pixel 1x1 label). Class level = ``taxonID`` species code (33
codes, a few genus-level e.g. Betula sp.); this is far under the 254-class uint8 cap so we
keep every class (ids 0..N-1 by descending frequency). Long-tailed distribution is fine --
the assembly step drops too-small classes (spec 5), not this script.

Time range: a tree's species is effectively static; the competition field/flight campaign
is 2018 (manifest time_range 2018-2019, post-2016). Per spec 5 (static/seasonal labels) we
anchor every sample on a single 1-year Sentinel-era window (2018).

Reproduce: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.idtrees
"""

import argparse
import multiprocessing
import zipfile
from collections import Counter
from typing import Any

import geopandas as gpd
import pandas as pd

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.download import download_zenodo
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "idtrees"
NAME = "IDTReeS"
ZENODO_RECORD = "3700197"
SITES = ["MLBS", "OSBS"]  # train ITC crown shapefiles present in the record

PER_CLASS = 1000  # spec default; total (~1.1k crowns) is far under the 25k cap
ANCHOR_YEAR = (
    2018  # competition field/flight campaign; static species label (post-2016)
)

SITE_DESC = {
    "MLBS": "Mountain Lake Biological Station, VA (deciduous forest)",
    "OSBS": "Ordway-Swisher Biological Station, FL (longleaf-pine flatwoods)",
}


def _ensure_raw() -> Any:
    """Download the Zenodo train zip (if needed), extract Field + ITC, return raw dir."""
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "IDTReeS 2018/2020 competition data (Weinstein et al.).\n"
            f"Zenodo record: https://zenodo.org/records/{ZENODO_RECORD} (CC-BY-4.0)\n"
            "Files used: IDTREES_competition_train.zip -> Field/train_data.csv, "
            "Field/taxonID_ScientificName.csv, ITC/train_{MLBS,OSBS}.shp (crown polygons + "
            "species labels only; RemoteSensing imagery not used).\n"
        )
    download_zenodo(ZENODO_RECORD, raw)
    zp = raw / "IDTREES_competition_train.zip"
    z = zipfile.ZipFile(str(zp))
    for n in z.namelist():
        if n.startswith("Field/") or n.startswith("ITC/"):
            if not (raw / n).exists():
                z.extract(n, str(raw))
    return raw


def load_records() -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    """Return (records, class_id_to_meta).

    records: {lon, lat, class_id, source_id}. class_id_to_meta: {name, scientific_name,
    taxon_rank, n_source_crowns}. Class ids are assigned by descending crown frequency.
    """
    raw = _ensure_raw()

    field = pd.read_csv(str(raw / "Field/train_data.csv"))
    taxon = pd.read_csv(str(raw / "Field/taxonID_ScientificName.csv"))
    sci_by_code = dict(
        zip(taxon["taxonID"].astype(str), taxon["scientificName"].astype(str))
    )
    # One (taxonID, rank, scientificName) per stem id.
    stem = field.drop_duplicates("indvdID").set_index("indvdID")

    frames = []
    for site in SITES:
        g = gpd.read_file(str(raw / f"ITC/train_{site}.shp"))
        g = g.to_crs(4326)
        cent = g.geometry.to_crs(g.estimate_utm_crs()).centroid.to_crs(4326)
        g = g.assign(lon=cent.x.to_numpy(), lat=cent.y.to_numpy(), site=site)
        frames.append(g[["indvdID", "lon", "lat", "site"]])
    crowns = pd.concat(frames, ignore_index=True)

    crowns["taxonID"] = crowns["indvdID"].map(stem["taxonID"]).astype("string")
    crowns["taxonRank"] = crowns["indvdID"].map(stem["taxonRank"]).astype("string")
    n_all = len(crowns)
    crowns = crowns.dropna(subset=["taxonID", "lon", "lat"])
    print(
        f"{n_all} crowns; {len(crowns)} with a species label "
        f"({n_all - len(crowns)} crowns had no matching field taxonID -> dropped)"
    )

    counts = crowns["taxonID"].value_counts()
    code_to_cid = {str(code): i for i, code in enumerate(counts.index)}
    print(f"{len(code_to_cid)} species/taxon classes (uint8 cap 254; keeping all)")

    cid_meta: dict[int, dict[str, Any]] = {}
    for code, cid in code_to_cid.items():
        sub = crowns[crowns["taxonID"] == code]
        rank = sub["taxonRank"].dropna()
        cid_meta[cid] = {
            "name": code,
            "scientific_name": sci_by_code.get(code),
            "taxon_rank": (str(rank.iloc[0]) if len(rank) else None),
            "n_source_crowns": int(len(sub)),
        }

    records = [
        {
            "lon": float(r.lon),
            "lat": float(r.lat),
            "class_id": code_to_cid[str(r.taxonID)],
            "source_id": f"{r.site}/{r.indvdID}",
        }
        for r in crowns.itertuples(index=False)
    ]
    return records, cid_meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()
    _ = args

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    records, cid_meta = load_records()

    selected = balance_by_class(records, "class_id", per_class=PER_CLASS)
    selected.sort(key=lambda r: (r["class_id"], r["source_id"]))
    print(f"selected {len(selected)} points (<= {PER_CLASS}/class, 25k cap)")

    points = [
        {
            "id": f"{i:06d}",
            "lon": r["lon"],
            "lat": r["lat"],
            "label": r["class_id"],
            "time_range": io.year_range(ANCHOR_YEAR),
            "source_id": r["source_id"],
        }
        for i, r in enumerate(selected)
    ]
    io.write_points_table(SLUG, "classification", points)

    counts = Counter(r["class_id"] for r in selected)
    classes = []
    for cid in range(len(cid_meta)):
        m = cid_meta[cid]
        sci = m["scientific_name"]
        rank = m["taxon_rank"]
        desc = None
        if sci:
            desc = f"NEON field-mapped tree crowns of {sci}"
            if rank:
                desc += f" ({rank}-level taxon)"
            desc += "."
        classes.append(
            {
                "id": cid,
                "name": m["name"],
                "description": desc,
                "scientific_name": sci,
                "taxon_rank": rank,
                "n_source_crowns": m["n_source_crowns"],
                "n_samples": counts.get(cid, 0),
            }
        )

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Zenodo (IDTReeS 2018/2020 competition; idtrees.org)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": f"https://zenodo.org/records/{ZENODO_RECORD}",
                "have_locally": False,
                "annotation_method": (
                    "NEON field plots (per-stem taxonID) + photointerpreted individual "
                    "tree-crown polygons, co-registered to NEON RGB/LiDAR/hyperspectral"
                ),
                "sites": {s: SITE_DESC[s] for s in SITES},
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": classes,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "notes": (
                "Sparse-point tree-species segmentation written as points.geojson (1x1 "
                "sub-pixel labels): each individual tree-crown polygon -> its centroid, "
                "labelled with the field taxonID species code. 33 taxon classes (a few "
                "genus-level, e.g. Betula sp.), all kept (well under the 254-class uint8 "
                "cap); ids 0..N-1 by descending crown frequency; long-tailed (PIPA2 / "
                "Pinus palustris most common), sparse classes kept per spec 5 (assembly "
                "filters too-small classes downstream). Two train sites: MLBS (VA) + OSBS "
                "(FL). WEAK habitat/species label: a ~4.2 m crown is sub-pixel at 10 m, so "
                "species is not directly resolvable at 10-30 m -- treated as a weak "
                "contextual label (same posture as globalgeotree), valid because the crowns "
                "sit in natural forest (not the urban auto_arborist case). Static species "
                f"label anchored on a 1-year window ({ANCHOR_YEAR}, post-2016)."
            ),
        },
    )

    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print(f"done: {len(selected)} points across {len(cid_meta)} classes")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()

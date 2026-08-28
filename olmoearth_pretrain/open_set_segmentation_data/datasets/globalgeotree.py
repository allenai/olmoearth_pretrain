"""Process GlobalGeoTree tree-occurrence records into open-set-segmentation labels.

Source: GlobalGeoTree (Yang et al., ESSD), a global vision-language dataset of ~6.3M
geolocated tree occurrences paired with Sentinel-2 time series and environmental
variables. We use only the occurrence metadata table ``files/GlobalGeoTree.csv`` on the
Hugging Face dataset repo ``yann111/GlobalGeoTree`` (not the WebDataset imagery tars).
Each row is one tree observed at a single lon/lat, with a full taxonomic hierarchy
(level0 leaf type / level1_family / level2_genus / level3_species), a GBIF species_key,
the observation source (iNaturalist / GBIF / forest inventories), and an observation year.

This is the natural "one class per point" fit for the sparse-point recipe (one species
label per record). Two constraints from the task spec shape the class set:

  * Labels are single-band uint8 (ids 0..254, 255=nodata), so at most 254 distinct
    classes. The source has ~20.7k species (post-2016), so we keep the **top 254 species
    by observation frequency** (ids 0..253 in descending frequency; each kept species has
    >= ~4.2k observations) and drop the remaining ~20.5k rarer species (recorded as a
    count in the summary). Species is the class level (per the task).
  * Pre-2016 labels are outside the Sentinel era. The source spans 2015-2024 with ~205k
    pre-2016 rows; we keep only year >= 2016 and drop the pre-2016 subset.

Sparse-point dataset -> one dataset-wide GeoJSON point table (points.geojson, spec 2a):
one Point feature per observation with lon/lat, label = class id, and a 1-year time range
anchored on the observation year. Balanced to the 25k per-dataset cap (~98/class).

Reproduce: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.globalgeotree
"""

import argparse
import multiprocessing
from collections import Counter
from typing import Any

import pandas as pd

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "globalgeotree"
NAME = "GlobalGeoTree"
HF_REPO = "yann111/GlobalGeoTree"
CSV_FILE = "files/GlobalGeoTree.csv"

N_CLASSES = 254  # max that fits uint8 (ids 0..253; 255 = nodata)
PER_CLASS = 1000  # spec default; total_cap=25000 lowers it to 25000//254 = 98/class
MIN_YEAR = 2016  # Sentinel-2 era; drop pre-2016 rows


def load_records() -> tuple[
    list[dict[str, Any]], dict[str, int], dict[int, dict[str, Any]]
]:
    """Load the occurrence CSV, filter, keep top-N species, return flat records.

    Returns (records, species_to_class_id, class_id_to_meta) where class_id_to_meta maps
    each kept class id to {name, family, genus, level0, species_key, n_source_obs}.
    """
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "GlobalGeoTree (Yang et al., ESSD).\n"
            f"Hugging Face dataset repo: {HF_REPO}\n"
            f"File used: {CSV_FILE} (occurrence metadata only; imagery tars not needed).\n"
            "GitHub: https://github.com/MUYang99/GlobalGeoTree\n"
        )
    csv_path = download.hf_download(HF_REPO, CSV_FILE, raw)

    df = pd.read_csv(
        str(csv_path),
        usecols=[
            "sample_id",
            "level0",
            "level1_family",
            "level2_genus",
            "level3_species",
            "species_key",
            "source",
            "year",
            "longitude",
            "latitude",
        ],
    )
    n0 = len(df)
    df = df.dropna(subset=["level3_species", "year", "longitude", "latitude"])
    df = df[(df["latitude"].between(-90, 90)) & (df["longitude"].between(-180, 180))]
    df["year"] = df["year"].astype(int)
    n_pre = int((df["year"] < MIN_YEAR).sum())
    df = df[df["year"] >= MIN_YEAR]
    print(
        f"loaded {n0} rows; {len(df)} after coord/species/year filters "
        f"(dropped {n_pre} pre-{MIN_YEAR} rows)"
    )

    counts = df["level3_species"].value_counts()
    top = counts.head(N_CLASSES)
    species_to_cid = {str(sp): i for i, sp in enumerate(top.index)}
    print(
        f"{df['level3_species'].nunique()} species total (>= {MIN_YEAR}); keeping top "
        f"{len(species_to_cid)} (min obs among kept = {int(top.min())}); dropping "
        f"{df['level3_species'].nunique() - len(species_to_cid)} rarer species"
    )

    df = df[df["level3_species"].isin(species_to_cid)]

    # Per-class taxonomy metadata (family/genus/leaf-type) from the first row per species.
    cid_meta: dict[int, dict[str, Any]] = {}
    for sp, g in df.groupby("level3_species"):
        cid = species_to_cid[str(sp)]
        r0 = g.iloc[0]
        sk = r0["species_key"]
        cid_meta[cid] = {
            "name": str(sp),
            "family": (
                str(r0["level1_family"]).strip()
                if pd.notna(r0["level1_family"])
                else None
            ),
            "genus": (
                str(r0["level2_genus"]).strip()
                if pd.notna(r0["level2_genus"])
                else None
            ),
            "level0": (str(r0["level0"]).strip() if pd.notna(r0["level0"]) else None),
            "species_key": (int(sk) if pd.notna(sk) else None),
            "n_source_obs": int(top.iloc[cid]),
        }

    records = [
        {
            "lon": float(row.longitude),
            "lat": float(row.latitude),
            "class_id": species_to_cid[str(row.level3_species)],
            "year": int(row.year),
            "source_id": f"ggt_{int(row.sample_id)}",
        }
        for row in df.itertuples(index=False)
    ]
    return records, species_to_cid, cid_meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    records, species_to_cid, cid_meta = load_records()

    selected = balance_by_class(records, "class_id", per_class=PER_CLASS)
    selected.sort(key=lambda r: (r["class_id"], r["source_id"]))
    print(
        f"selected {len(selected)} points (<= {PER_CLASS}/class, 25k cap -> "
        f"~{25000 // len(species_to_cid)}/class over {len(species_to_cid)} classes)"
    )

    # Sparse point dataset -> one dataset-wide GeoJSON point table (spec 2a).
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

    counts = Counter(r["class_id"] for r in selected)
    classes = []
    for cid in range(len(species_to_cid)):
        m = cid_meta[cid]
        desc_parts = []
        if m["genus"]:
            desc_parts.append(f"genus {m['genus']}")
        if m["family"]:
            desc_parts.append(f"family {m['family']}")
        if m["level0"]:
            desc_parts.append(m["level0"].lower() + " tree")
        description = (
            ("Tree species " + m["name"] + " (" + ", ".join(desc_parts) + ").")
            if desc_parts
            else None
        )
        classes.append(
            {
                "id": cid,
                "name": m["name"],
                "description": description,
                "family": m["family"],
                "genus": m["genus"],
                "leaf_type": m["level0"],
                "gbif_species_key": m["species_key"],
                "n_source_observations": m["n_source_obs"],
                "n_samples": counts.get(cid, 0),
            }
        )

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "GitHub / ESSD (GlobalGeoTree; Hugging Face yann111/GlobalGeoTree)",
            "license": "CC-BY",
            "provenance": {
                "url": "https://github.com/MUYang99/GlobalGeoTree",
                "huggingface": HF_REPO,
                "have_locally": False,
                "annotation_method": "tree occurrence records (iNaturalist / GBIF / forest inventories) paired with S2",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": classes,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "notes": (
                "Sparse-point tree-species segmentation, written as a points.geojson table "
                "(1x1 pixel labels). Class level = species (level3_species). Source has "
                f"~20.7k species post-{MIN_YEAR}; classification is uint8 so we cap at "
                f"{N_CLASSES} classes and keep the top {len(species_to_cid)} species by "
                "observation frequency (ids 0..N-1 descending; each kept species has "
                ">=~4.2k source obs), dropping ~20.5k rarer species. Pre-2016 rows (~205k, "
                f"years 2015) dropped; kept year >= {MIN_YEAR} (Sentinel-2 era). Balanced to "
                "the 25k per-dataset cap (~98 randomly-sampled points/class). 1-year time "
                "range anchored on each observation's year (2016-2024). All observation "
                "sources used. Tree-species presence is only weakly observable at 10-30 m "
                "from S2/S1/Landsat -> treat as weak/contextual habitat labels."
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

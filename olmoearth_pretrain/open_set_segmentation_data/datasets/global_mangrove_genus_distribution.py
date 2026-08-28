"""Global Mangrove Genus Distribution -> sparse-point genus classification.

Source: Twomey & Lovelock (2024), "Global spatial dataset of mangrove genus distribution
in seaward and riverine margins", Scientific Data 11, 306
(https://doi.org/10.1038/s41597-024-03134-1). Data on PANGAEA
(https://doi.pangaea.de/10.1594/PANGAEA.942481), CC-BY-4.0.

The release has two products:
  * ``FrontalMangroveGenus`` shapefile: 250 Marine-Ecoregions-of-the-World polygons, each
    tagged with the dominant *frontal* (seaward-margin) mangrove genus. These polygons are
    whole marine ecoregions (median ~625,000 km2, mostly open ocean) -> NOT usable as a
    per-pixel label.
  * ``MangroveZonationData.xlsx`` "Original Data" sheet: 733 mangrove-zonation studies
    compiled from the literature, each with a Frontal Mangrove Genus/Species, a country /
    location, and a per-record Latitude/Longitude (precision flagged "Specific" vs
    "Estimated"). 473 rows carry usable coordinates.

We use the georeferenced zonation records (label_type ``points`` per the manifest): each is
one 10 m point carrying the observed frontal mangrove genus at a real coastal mangrove
location. Written as one dataset-wide ``points.geojson`` (spec 2a). Static literature
compilation with no per-record date -> a representative Sentinel-era 1-year window (2020);
mangrove forests / genus composition are persistent, so this is a static-label window.

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_mangrove_genus_distribution
"""

import argparse
from collections import Counter

import pandas as pd

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "global_mangrove_genus_distribution"
NAME = "Global Mangrove Genus Distribution"
URL = "https://doi.pangaea.de/10.1594/PANGAEA.942481"
XLSX_REL = "MangroveZonationData.xlsx"
SHEET = "Original Data"
PER_CLASS = 1000
STATIC_YEAR = (
    2020  # representative Sentinel-era window (compilation has no per-record date)
)

# Genus-name normalization (fix source typos / trailing whitespace).
GENUS_FIX = {
    "aviennia": "Avicennia",
    "brugueira": "Bruguiera",
    "bruguiera": "Bruguiera",
    "luminitzera": "Lumnitzera",
    "lumnitzera": "Lumnitzera",
    "lagucularia": "Laguncularia",
}

# Short definitions for the common mangrove genera (source gives none per record).
GENUS_DESC = {
    "Rhizophora": "Red mangroves; stilt/prop-rooted trees typically dominating the seaward pioneer fringe.",
    "Avicennia": "Grey/black mangroves; pneumatophore-bearing trees, often the frontal genus on many coasts and cold-tolerant.",
    "Sonneratia": "Mangrove apple; large pneumatophore-bearing trees of the low intertidal seaward front (Indo-West Pacific).",
    "Laguncularia": "White mangrove (Atlantic-East Pacific); often a frontal/pioneer genus in the Americas and West Africa.",
    "Bruguiera": "Orange/large-leafed mangroves with knee roots, typically mid-to-landward zones.",
    "Ceriops": "Yellow mangroves; small knee-rooted trees of the mid/landward intertidal.",
    "Conocarpus": "Buttonwood; landward-margin mangrove associate (Atlantic-East Pacific).",
    "Nypa": "Nipa palm; estuarine/riverine mangrove palm of the Indo-West Pacific.",
}


def normalize_genus(v: object) -> str | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    if not s:
        return None
    return GENUS_FIX.get(s.lower(), s[0].upper() + s[1:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "Twomey & Lovelock (2024) Scientific Data; PANGAEA 942481 (CC-BY-4.0)\n"
            f"{URL}\n"
            "Downloaded: Genus_Shapefiles.zip (ecoregion polygons, unused) + "
            "MangroveZonationData.xlsx (georeferenced zonation records, used).\n"
        )

    xlsx_path = raw / XLSX_REL
    df = pd.read_excel(str(xlsx_path), sheet_name=SHEET)

    recs = []
    for _, row in df.iterrows():
        lat, lon = row.get("Latitude"), row.get("Longitude")
        if pd.isna(lat) or pd.isna(lon):
            continue
        genus = normalize_genus(row.get("Frontal Mangrove Genus"))
        if genus is None:
            continue
        lat, lon = float(lat), float(lon)
        if not (-180 <= lon <= 180):
            continue
        # Global mangrove latitude band is ~33 N to ~40 S; drop out-of-band coords
        # (one erroneous record: "Punta Arenas, Chile" at -53 S).
        if not (-40.0 <= lat <= 33.0):
            print(
                f"  drop out-of-band lat={lat:.3f} lon={lon:.3f} genus={row.get('Frontal Mangrove Genus')}"
            )
            continue
        species = row.get("Frontal Mangrove Species")
        prec = row.get("Location Coordinates")
        recs.append(
            {
                "lon": lon,
                "lat": lat,
                "genus": genus,
                "species": None if pd.isna(species) else str(species).strip(),
                "coord_precision": None if pd.isna(prec) else str(prec).strip(),
                "row_id": int(row["Unnamed: 0"])
                if not pd.isna(row.get("Unnamed: 0"))
                else _,
            }
        )
    print(f"{len(recs)} georeferenced genus records")

    # Class map: genera ordered by frequency (descending) -> ids 0..N (well under 254 cap).
    counts = Counter(r["genus"] for r in recs)
    ordered = [g for g, _ in counts.most_common()]
    genus_to_id = {g: i for i, g in enumerate(ordered)}
    print("genus counts:", dict(counts.most_common()))

    selected = balance_by_class(recs, "genus", per_class=PER_CLASS)
    print(f"selected {len(selected)} (<= {PER_CLASS}/class)")

    points = []
    for i, r in enumerate(selected):
        points.append(
            {
                "id": f"{i:06d}",
                "lon": r["lon"],
                "lat": r["lat"],
                "label": genus_to_id[r["genus"]],
                "time_range": io.year_range(STATIC_YEAR),
                "source_id": f"zonation_row_{r['row_id']}",
                "genus": r["genus"],
                "species": r["species"],
                "coord_precision": r["coord_precision"],
            }
        )
    io.write_points_table(SLUG, "classification", points)

    sel_counts = Counter(r["genus"] for r in selected)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Scientific Data (PANGAEA 942481)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": URL,
                "have_locally": False,
                "annotation_method": "manual literature/field compilation of mangrove zonation diagrams",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": genus_to_id[g], "name": g, "description": GENUS_DESC.get(g)}
                for g in ordered
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {g: sel_counts.get(g, 0) for g in ordered},
            "notes": (
                "Sparse 1x1 point-segmentation labels (points.geojson, spec 2a). Label = "
                "observed dominant *frontal* (seaward-margin) mangrove genus at each site. "
                "472 georeferenced zonation records (one out-of-band coord dropped); 68 have "
                "'Specific' precise coordinates, the rest 'Estimated' from location names "
                "(see per-point coord_precision). "
                "Static literature compilation -> representative 2020 1-year window "
                "(mangrove genus composition is persistent). Species kept in properties for "
                "reference but not used as the class (genus is the target). Ecoregion-polygon "
                "product in the release is NOT used (whole marine ecoregions, not per-pixel)."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("done")


if __name__ == "__main__":
    main()

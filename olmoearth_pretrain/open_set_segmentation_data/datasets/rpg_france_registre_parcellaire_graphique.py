"""Process RPG France (Registre Parcellaire Graphique) into label patches.

Source: IGN France / ASP, the anonymized national LPIS of French agricultural parcels
(farmer CAP declarations), distributed openly under Licence Ouverte / Etalab via
``data.geopf.fr`` (Géoplateforme). From the 2015 edition on, the "RPG 2.0" product carries
the crop type down to the individual **parcelle** level: each parcel polygon has a 3-letter
``CODE_CULTU`` (detailed crop code, ~300 nationally) and a numeric ``CODE_GROUP`` (grouped
crop, ~28 groups). This is the largest single-country LPIS and is the annual, national
analogue of the EuroCrops snapshots -- so this script mirrors ``eurocrops.py``.

RPG is huge (national, ~9.5M parcels/year). We download a **bounded, geographically
diverse subset of administrative regions** for one recent snapshot year (2022, within the
manifest's 2016-2024 range), covering all French agroclimatic zones and every major crop:

  R24 Centre-Val de Loire (Beauce cereals, rapeseed), R32 Hauts-de-France (sugar beet,
  potato, wheat, flax), R44 Grand Est (Champagne/Alsace vineyards, sugar beet, wheat),
  R53 Bretagne (maize, grassland, vegetables), R75 Nouvelle-Aquitaine (maize, sunflower,
  vineyard), R76 Occitanie (durum wheat, vineyard, orchards, sunflower), R84
  Auvergne-Rhone-Alpes (grassland, orchards, maize), R93 PACA (vineyard, orchards, rice in
  the Camargue, lavender).

Task: per-pixel **classification** (crop type). Each selected parcel is rasterized into a
<=64x64 local-UTM 10 m tile: the parcel's ``CODE_CULTU`` class id is burned inside the
polygon, everything outside is nodata (255) -- we only have a ground-truth crop label
inside declared parcels, so outside is "ignore", not a background class (spec 5).

Classes are the distinct ``CODE_CULTU`` codes present in the sampled parcels. Labels are
uint8 (ids 0-253, 255=nodata) so at most 254 classes: if more than 254 codes appear we keep
the top 254 by global frequency and drop the rest (logged). Class ids are assigned 0..N-1 in
descending global frequency. Names come from the IGN/etalab RPG culture nomenclature
(CODE_CULTU -> French libelle); the RPG crop-group name is attached as the class description.

Sampling: tiles-per-class balanced with the 25k per-dataset cap. With N classes the
effective per-class limit is min(1000, 25000 // N) (``balance_by_class`` default). Rare
classes are prioritized to reach the (reduced) target; truncation is logged.

Time range: 1-year window anchored on the snapshot year (2022).

Run (idempotent; skips already-written {sample_id}.tif):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.rpg_france_registre_parcellaire_graphique
"""

import argparse
import csv
import multiprocessing
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import py7zr
import pyogrio
import shapely
import tqdm
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "rpg_france_registre_parcellaire_graphique"
NAME = "RPG France (Registre Parcellaire Graphique)"
YEAR = 2022
GEOPF_BASE = "https://data.geopf.fr/telechargement/download/RPG"

# Bounded, geographically diverse subset of French administrative regions (metropolitan,
# Lambert-93), all for the 2022 snapshot. Each is a single .7z archive on the Geoplateforme.
REGIONS = [
    {"code": "R24", "label": "Centre-Val de Loire", "note": "Beauce cereals, rapeseed"},
    {
        "code": "R32",
        "label": "Hauts-de-France",
        "note": "sugar beet, potato, wheat, flax",
    },
    {
        "code": "R44",
        "label": "Grand Est",
        "note": "Champagne/Alsace vineyards, sugar beet",
    },
    {"code": "R53", "label": "Bretagne", "note": "maize, grassland, vegetables"},
    {
        "code": "R75",
        "label": "Nouvelle-Aquitaine",
        "note": "maize, sunflower, vineyard",
    },
    {"code": "R76", "label": "Occitanie", "note": "durum wheat, vineyard, orchards"},
    {
        "code": "R84",
        "label": "Auvergne-Rhone-Alpes",
        "note": "grassland, orchards, maize",
    },
    {"code": "R93", "label": "PACA", "note": "vineyard, orchards, rice (Camargue)"},
]

CODE_PROPERTY = "CODE_CULTU"
GROUP_PROPERTY = "CODE_GROUP"

# CODE_CULTU -> French libelle nomenclature (IGN/ASP RPG), mirrored by etalab/api-rpg.
CULTURE_CSV_URL = (
    "https://raw.githubusercontent.com/etalab/api-rpg/master/codes/CULTURE.csv"
)

# RPG "groupe de cultures" nomenclature (CODE_GROUP -> group libelle), used as the class
# description. Standard RPG 2.0 28-group scheme.
GROUP_NAMES = {
    "1": "Ble tendre",
    "2": "Mais grain et ensilage",
    "3": "Orge",
    "4": "Autres cereales",
    "5": "Colza",
    "6": "Tournesol",
    "7": "Autres oleagineux",
    "8": "Proteagineux",
    "9": "Plantes a fibres",
    "10": "Semences",
    "11": "Gel (surfaces gelees sans production)",
    "12": "Gel industriel",
    "13": "Autres gels",
    "14": "Riz",
    "15": "Legumineuses a grains",
    "16": "Fourrage",
    "17": "Estives et landes",
    "18": "Prairies permanentes",
    "19": "Prairies temporaires",
    "20": "Vergers",
    "21": "Vignes",
    "22": "Fruits a coque",
    "23": "Oliviers",
    "24": "Autres cultures industrielles",
    "25": "Legumes ou fleurs",
    "26": "Canne a sucre",
    "27": "Arboriculture",
    "28": "Divers",
}

# The Geoplateforme download host rejects urllib's default agent (HTTP 403); send a browser UA.
_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

# uint8 class labels -> at most 254 classes (255 = nodata).
MAX_CLASSES = 254
PER_CLASS = (
    1000  # spec target; lowered automatically to 25000 // N by balance_by_class.
)
MAX_TILE = io.MAX_TILE  # 64

_WGS84_SRC = Projection(CRS.from_epsg(4326), 1, 1)


# --------------------------------------------------------------------------------------
# Nomenclature.
# --------------------------------------------------------------------------------------
def load_culture_names() -> dict[str, str]:
    """Return {CODE_CULTU: french_libelle} from the RPG culture nomenclature CSV."""
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    csv_path = raw / "CULTURE.csv"
    download.download_http(CULTURE_CSV_URL, csv_path)
    names: dict[str, str] = {}
    with open(csv_path.path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader, None)
        for row in reader:
            if len(row) >= 2 and row[0]:
                names[row[0].strip()] = row[1].strip()
    return names


# --------------------------------------------------------------------------------------
# Download + extract (.7z from the Geoplateforme).
# --------------------------------------------------------------------------------------
def region_archive_name(code: str) -> str:
    return f"RPG_2-0__SHP_LAMB93_{code}_{YEAR}-01-01"


def ensure_data() -> dict[str, str]:
    """Download + extract each region's .7z; return {code: PARCELLES_GRAPHIQUES.shp path}."""
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    shp_by_code: dict[str, str] = {}
    for r in REGIONS:
        io.check_disk()
        name = region_archive_name(r["code"])
        archive = raw / f"{name}.7z.001"
        url = f"{GEOPF_BASE}/{name}/{name}.7z.001"
        download.download_http(url, archive, headers={"User-Agent": _UA})
        dest = Path(raw.path) / "unzip" / r["code"]
        dest.mkdir(parents=True, exist_ok=True)
        shps = list(dest.rglob("PARCELLES_GRAPHIQUES.shp"))
        if not shps:
            with py7zr.SevenZipFile(archive.path, "r") as z:
                z.extractall(dest)
            shps = list(dest.rglob("PARCELLES_GRAPHIQUES.shp"))
        if not shps:
            raise RuntimeError(
                f"no PARCELLES_GRAPHIQUES.shp for {r['code']} after extract"
            )
        shp_by_code[r["code"]] = str(shps[0])
        print(f"  {r['code']} ({r['label']}): {shps[0]}")
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "RPG France 2.0 (Registre Parcellaire Graphique), IGN France / ASP.\n"
            "Licence Ouverte / Etalab. https://geoservices.ign.fr/rpg\n"
            "Downloaded from the Geoplateforme (data.geopf.fr/telechargement).\n"
            f"Snapshot year {YEAR}. Regions (bounded diverse subset): "
            + ", ".join(f"{r['code']} {r['label']}" for r in REGIONS)
            + "\nCulture nomenclature: "
            + CULTURE_CSV_URL
            + "\n"
        )
    return shp_by_code


# --------------------------------------------------------------------------------------
# Pass 1: read crop codes (no geometry) for frequency + code->group.
# --------------------------------------------------------------------------------------
def read_codes(shp_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (CODE_CULTU, CODE_GROUP) string arrays per feature in fid order."""
    df = pyogrio.read_dataframe(
        shp_path,
        columns=[CODE_PROPERTY, GROUP_PROPERTY],
        read_geometry=False,
        fid_as_index=True,
    )
    cultu = df[CODE_PROPERTY].fillna("").astype(str).to_numpy()
    group = df[GROUP_PROPERTY].fillna("").astype(str).to_numpy()
    return cultu, group


# --------------------------------------------------------------------------------------
# Pass 2 worker: rasterize one parcel into a <=64x64 UTM tile.
# --------------------------------------------------------------------------------------
def _write_tile(rec: dict[str, Any]) -> tuple[str, str]:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return sample_id, "skip"
    try:
        geom = shapely.from_wkb(rec["geom_wkb"])  # WGS84 (lon/lat) geometry
        proj = io.utm_projection_for_lonlat(rec["lon"], rec["lat"])
        pix = geom_to_pixels(geom, _WGS84_SRC, proj)
        minx, miny, maxx, maxy = pix.bounds
        cx = int(round((minx + maxx) / 2))
        cy = int(round((miny + maxy) / 2))
        w = min(MAX_TILE, max(1, int(np.ceil(maxx - minx))))
        h = min(MAX_TILE, max(1, int(np.ceil(maxy - miny))))
        bounds = io.centered_bounds(cx, cy, w, h)
        arr = rasterize_shapes(
            [(pix, int(rec["class_id"]))],
            bounds,
            fill=io.CLASS_NODATA,
            dtype="uint8",
            all_touched=True,
        )
        if not (arr != io.CLASS_NODATA).any():
            return sample_id, "empty"
        io.write_label_geotiff(
            SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA
        )
        io.write_sample_json(
            SLUG,
            sample_id,
            proj,
            bounds,
            io.year_range(rec["year"]),
            source_id=rec["source_id"],
            classes_present=sorted(set(np.unique(arr).tolist()) - {io.CLASS_NODATA}),
        )
        return sample_id, "ok"
    except Exception as e:  # noqa: BLE001
        print(f"error on {sample_id}: {e}")
        return sample_id, "error"


# --------------------------------------------------------------------------------------
# Main.
# --------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    culture_names = load_culture_names()
    shp_by_code = ensure_data()

    # ---- Pass 1: codes per region -------------------------------------------------
    codes_by_code: dict[str, np.ndarray] = {}
    global_freq: Counter = Counter()
    code_to_group: dict[str, str] = {}
    for r in REGIONS:
        cultu, group = read_codes(shp_by_code[r["code"]])
        codes_by_code[r["code"]] = cultu
        valid = cultu != ""
        for code, n in Counter(cultu[valid].tolist()).items():
            global_freq[code] += n
        for code, grp in zip(cultu[valid].tolist(), group[valid].tolist()):
            code_to_group.setdefault(code, grp)
        print(
            f"  {r['code']}: {len(cultu)} parcels, "
            f"{len(set(cultu[valid].tolist()))} distinct CODE_CULTU"
        )
        io.check_disk()

    # ---- Keep top-N codes by frequency, assign ids 0..N-1 (descending freq) --------
    ranked = [code for code, _ in global_freq.most_common()]
    kept = ranked[:MAX_CLASSES]
    dropped = ranked[MAX_CLASSES:]
    code_to_id = {code: i for i, code in enumerate(kept)}
    print(
        f"total distinct CODE_CULTU: {len(ranked)}; kept: {len(kept)}; "
        f"dropped: {len(dropped)}"
    )

    # ---- Build candidate (region, fid) lists per class, then balance ---------------
    records: list[dict[str, Any]] = []
    for r in REGIONS:
        cultu = codes_by_code[r["code"]]
        for code, cid in code_to_id.items():
            fids = np.nonzero(cultu == code)[0]
            for fid in fids.tolist():
                records.append(
                    {"code": code, "class_id": cid, "region": r["code"], "fid": fid}
                )
    print(f"candidate parcels for kept classes: {len(records)}")

    selected = balance_by_class(
        records, key="class_id", per_class=PER_CLASS, total_cap=25000
    )
    n_classes = len(code_to_id)
    eff_per_class = max(1, min(PER_CLASS, 25000 // n_classes))
    print(f"selected {len(selected)} parcels (eff per-class cap = {eff_per_class})")

    # ---- Pass 2: read geometries for selected fids (grouped by region) -------------
    by_region: dict[str, list[dict[str, Any]]] = {}
    for r in selected:
        by_region.setdefault(r["region"], []).append(r)

    tile_recs: list[dict[str, Any]] = []
    for region_code, recs in by_region.items():
        fids = sorted({r["fid"] for r in recs})
        gdf = pyogrio.read_dataframe(
            shp_by_code[region_code],
            columns=[CODE_PROPERTY],
            fids=fids,
            fid_as_index=True,
        )
        gdf_wgs = gdf.to_crs(4326)
        geom_by_fid = {int(fid): geom for fid, geom in gdf_wgs.geometry.items()}
        for r in recs:
            geom = geom_by_fid.get(int(r["fid"]))
            if geom is None or geom.is_empty:
                continue
            cent = geom.centroid
            if not np.isfinite(cent.x) or not np.isfinite(cent.y):
                continue
            tile_recs.append(
                {
                    "class_id": r["class_id"],
                    "lon": float(cent.x),
                    "lat": float(cent.y),
                    "geom_wkb": shapely.to_wkb(geom),
                    "year": YEAR,
                    "source_id": f"{region_code}/{r['fid']}",
                }
            )
        print(f"  read {len(recs)} geometries for {region_code}")
        io.check_disk()

    for i, r in enumerate(tile_recs):
        r["sample_id"] = f"{i:06d}"

    # ---- Write tiles in parallel ---------------------------------------------------
    results: Counter = Counter()
    written_by_class: Counter = Counter()
    id_to_rec = {r["sample_id"]: r for r in tile_recs}
    with multiprocessing.Pool(args.workers) as p:
        for sample_id, res in tqdm.tqdm(
            star_imap_unordered(p, _write_tile, [dict(rec=r) for r in tile_recs]),
            total=len(tile_recs),
        ):
            results[res] += 1
            if res in ("ok", "skip"):
                written_by_class[id_to_rec[sample_id]["class_id"]] += 1
    print("write results:", dict(results))
    io.check_disk()

    # ---- Metadata ------------------------------------------------------------------
    def class_name(code: str) -> str:
        return culture_names.get(code, code)

    def class_desc(code: str) -> str | None:
        grp = code_to_group.get(code)
        gname = GROUP_NAMES.get(grp) if grp else None
        if gname:
            return f"RPG CODE_CULTU '{code}'; crop group {grp} ({gname})."
        return f"RPG CODE_CULTU '{code}'."

    classes = [
        {"id": cid, "name": class_name(code), "description": class_desc(code)}
        for code, cid in sorted(code_to_id.items(), key=lambda kv: kv[1])
    ]
    class_counts = {
        class_name(code): int(written_by_class.get(cid, 0))
        for code, cid in sorted(code_to_id.items(), key=lambda kv: kv[1])
    }
    num_written = int(results.get("ok", 0) + results.get("skip", 0))
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "IGN France / ASP (Geoplateforme)",
            "license": "Licence Ouverte / Etalab",
            "provenance": {
                "url": "https://geoservices.ign.fr/rpg",
                "have_locally": False,
                "annotation_method": "farmer declaration (CAP), anonymized LPIS",
                "snapshot_year": YEAR,
                "regions": [
                    {"code": r["code"], "label": r["label"], "note": r["note"]}
                    for r in REGIONS
                ],
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": classes,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": num_written,
            "class_counts": class_counts,
            "dropped_code_cultu": dropped,
            "notes": (
                "French national LPIS crop parcels (RPG 2.0), bounded diverse subset of 8 "
                "administrative regions (R24, R32, R44, R53, R75, R76, R84, R93) for the "
                f"{YEAR} snapshot. Each parcel rasterized into a <=64x64 local-UTM 10 m "
                "tile: CODE_CULTU class id inside the polygon, 255 (nodata/ignore) outside "
                "(no true background class; unlabeled land is ignore). Class ids assigned "
                "0..N-1 by descending global CODE_CULTU frequency; kept top "
                f"{len(kept)} of {len(ranked)} codes (dropped {len(dropped)}). "
                "Tiles-per-class balanced with the 25k cap. Time range = 1-year window on "
                f"the {YEAR} snapshot year."
            ),
        },
    )

    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=num_written
    )
    print(f"done: {num_written} samples across {n_classes} classes")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()

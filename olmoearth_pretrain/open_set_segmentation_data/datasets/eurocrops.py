"""Process EuroCrops into open-set-segmentation label patches (rasterized crop polygons).

Source: EuroCrops (Zenodo record 10118572), the largest harmonized open EU crop-type
parcel dataset. National LPIS / CAP farmer declarations from many countries, each mapped
to the shared **HCAT** (Hierarchical Crop and Agriculture Taxonomy) via the per-feature
``EC_hcat_c`` 10-digit code. Licensed CC-BY-4.0.

EuroCrops is very large (dozens of country GeoPackages/shapefiles, tens of GB). We
download a **bounded, geographically diverse subset of countries** covering the main
European biogeographic regions and crop mixes (Iberia/Mediterranean, Alpine, Nordic,
Baltic, Balkans, maritime NW-Europe):

  PT (Portugal 2021), ES_NA (Spain / Navarra 2020), AT (Austria 2021), DK (Denmark 2019),
  EE (Estonia 2021), HR (Croatia 2020), NL (Netherlands 2020), SE (Sweden 2021).

Task: per-pixel **classification** (crop type). Each selected field parcel is rasterized
into a <=64x64 UTM 10 m tile (tile sized to the parcel footprint, capped at 64): the
parcel's crop class id is burned inside the polygon, everything outside is nodata (255) --
we only have a ground-truth crop label inside declared parcels, so outside is "ignore",
not a background class.

Classes are the HCAT leaf codes present in the sampled parcels. HCAT has ~175+ nodes;
classification labels are uint8 (ids 0-253, 255=nodata), so if more than 254 distinct
codes appear we keep the top 254 by frequency and drop the rest (documented in the
summary). Class ids are assigned 0..N-1 in **descending global frequency**. Names come
from the repo's HCAT3 mapping (data/eurocrops_hcat3_mapping.json).

Sampling: tiles-per-class balanced with the 25k per-dataset cap. With N classes the
effective per-class limit is min(1000, 25000 // N) (``balance_by_class`` default). Rare
classes are prioritized to reach the (reduced) target; truncation is logged.

Time range: 1-year window anchored on each country snapshot's year.

Run (idempotent; skips already-written {sample_id}.tif):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.eurocrops
"""

import argparse
import json
import multiprocessing
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
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

SLUG = "eurocrops"
NAME = "EuroCrops"
ZENODO_RECORD = "10118572"

# Bounded, geographically diverse country subset (zip name, short code, snapshot year).
COUNTRIES = [
    {"zip": "PT.zip", "code": "PT", "year": 2021, "region": "Portugal (Mediterranean)"},
    {
        "zip": "ES_NA_2020.zip",
        "code": "ES_NA",
        "year": 2020,
        "region": "Spain / Navarra (Mediterranean)",
    },
    {"zip": "AT_2021.zip", "code": "AT", "year": 2021, "region": "Austria (Alpine)"},
    {
        "zip": "DK_2019.zip",
        "code": "DK",
        "year": 2019,
        "region": "Denmark (Nordic lowland)",
    },
    {"zip": "EE_2021.zip", "code": "EE", "year": 2021, "region": "Estonia (Baltic)"},
    {"zip": "HR_2020.zip", "code": "HR", "year": 2020, "region": "Croatia (Balkans)"},
    {
        "zip": "NL_2020.zip",
        "code": "NL",
        "year": 2020,
        "region": "Netherlands (maritime NW-Europe)",
    },
    {"zip": "SE_2021.zip", "code": "SE", "year": 2021, "region": "Sweden (Nordic)"},
]

HCAT_CODE_PROPERTY = "EC_hcat_c"
HCAT_MAPPING_PATH = "data/eurocrops_hcat3_mapping.json"

# uint8 class labels -> at most 254 classes (255 = nodata).
MAX_CLASSES = 254
PER_CLASS = (
    1000  # spec target; lowered automatically to 25000 // N by balance_by_class.
)
MAX_TILE = io.MAX_TILE  # 64


# --------------------------------------------------------------------------------------
# HCAT name lookup.
# --------------------------------------------------------------------------------------
def load_hcat_names() -> dict[int, str]:
    """Return {hcat_code(int): hcat_name} from the repo HCAT3 mapping."""
    with open(HCAT_MAPPING_PATH) as f:
        entries = json.load(f)
    return {int(e["hcat_code"]): e["hcat_name"] for e in entries}


# --------------------------------------------------------------------------------------
# Download + unzip.
# --------------------------------------------------------------------------------------
def ensure_data() -> dict[str, str]:
    """Download the chosen country zips + HCAT3.csv, unzip, return {code: shp_path}."""
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    want = [c["zip"] for c in COUNTRIES] + ["HCAT3.csv"]
    download.download_zenodo(ZENODO_RECORD, raw, want)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "EuroCrops (Zenodo record 10118572), CC-BY-4.0.\n"
            "https://zenodo.org/records/10118572 ; https://github.com/maja601/EuroCrops\n"
            "Countries downloaded (bounded diverse subset): "
            + ", ".join(f"{c['zip']}" for c in COUNTRIES)
            + "\n"
        )
    shp_by_code: dict[str, str] = {}
    unzip_root = Path(raw.path) / "unzip"
    for c in COUNTRIES:
        dest = unzip_root / c["code"]
        dest.mkdir(parents=True, exist_ok=True)
        shps = list(dest.rglob("*.shp"))
        if not shps:
            with zipfile.ZipFile(Path(raw.path) / c["zip"]) as zf:
                zf.extractall(dest)
            shps = list(dest.rglob("*.shp"))
        if not shps:
            raise RuntimeError(f"no .shp found for {c['code']} after unzip")
        # Prefer a shapefile whose name references the country/EC (skip stray aux).
        shps.sort(key=lambda p: (len(p.name), p.name))
        shp_by_code[c["code"]] = str(shps[0])
    return shp_by_code


# --------------------------------------------------------------------------------------
# Pass 1: read HCAT codes (no geometry) to compute frequency + candidates.
# --------------------------------------------------------------------------------------
def read_codes(shp_path: str) -> np.ndarray:
    """Return an int64 array of HCAT codes per feature (fid order); -1 where missing."""
    import pandas as pd

    df = pyogrio.read_dataframe(
        shp_path, columns=[HCAT_CODE_PROPERTY], read_geometry=False, fid_as_index=True
    )
    nums = pd.to_numeric(df[HCAT_CODE_PROPERTY], errors="coerce")
    return nums.fillna(-1).to_numpy().astype(np.int64)


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


_WGS84_SRC = Projection(CRS.from_epsg(4326), 1, 1)


# --------------------------------------------------------------------------------------
# Main.
# --------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    hcat_names = load_hcat_names()
    shp_by_code = ensure_data()

    # ---- Pass 1: codes per country -----------------------------------------------
    codes_by_code: dict[str, np.ndarray] = {}
    global_freq: Counter = Counter()
    for c in COUNTRIES:
        codes = read_codes(shp_by_code[c["code"]])
        codes_by_code[c["code"]] = codes
        valid = codes[codes >= 0]
        # Only count codes that exist in the HCAT taxonomy.
        for code, n in Counter(valid.tolist()).items():
            if code in hcat_names:
                global_freq[code] += n
        print(f"  {c['code']}: {len(codes)} parcels, {len(set(valid.tolist()))} codes")

    io.check_disk()

    # ---- Keep top-N codes by frequency, assign ids 0..N-1 (descending freq) -------
    ranked = [code for code, _ in global_freq.most_common()]
    kept = ranked[:MAX_CLASSES]
    dropped = ranked[MAX_CLASSES:]
    code_to_id = {code: i for i, code in enumerate(kept)}
    print(
        f"total distinct HCAT codes: {len(ranked)}; kept: {len(kept)}; dropped: {len(dropped)}"
    )

    # ---- Build candidate (country, fid) lists per class, then balance -------------
    records: list[dict[str, Any]] = []
    for c in COUNTRIES:
        codes = codes_by_code[c["code"]]
        for code, cid in code_to_id.items():
            fids = np.nonzero(codes == code)[0]
            for fid in fids.tolist():
                records.append(
                    {
                        "code": code,
                        "class_id": cid,
                        "country": c["code"],
                        "fid": fid,
                        "year": c["year"],
                    }
                )
    print(f"candidate parcels for kept classes: {len(records)}")

    selected = balance_by_class(
        records, key="class_id", per_class=PER_CLASS, total_cap=25000
    )
    n_classes = len(code_to_id)
    eff_per_class = max(1, min(PER_CLASS, 25000 // n_classes))
    print(f"selected {len(selected)} parcels (eff per-class cap = {eff_per_class})")

    # ---- Pass 2: read geometries for selected fids (grouped by country) -----------
    by_country: dict[str, list[dict[str, Any]]] = {}
    for r in selected:
        by_country.setdefault(r["country"], []).append(r)

    tile_recs: list[dict[str, Any]] = []
    for code_cc, recs in by_country.items():
        fids = sorted({r["fid"] for r in recs})
        gdf = pyogrio.read_dataframe(
            shp_by_code[code_cc],
            columns=[HCAT_CODE_PROPERTY],
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
                    "year": r["year"],
                    "source_id": f"{code_cc}/{r['fid']}",
                }
            )
        print(f"  read {len(recs)} geometries for {code_cc}")
        io.check_disk()

    for i, r in enumerate(tile_recs):
        r["sample_id"] = f"{i:06d}"

    # ---- Write tiles in parallel --------------------------------------------------
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

    # ---- Metadata -----------------------------------------------------------------
    classes = [
        {
            "id": cid,
            "name": hcat_names.get(code, str(code)),
            "description": f"HCAT code {code} ({hcat_names.get(code, 'unknown')}).",
        }
        for code, cid in sorted(code_to_id.items(), key=lambda kv: kv[1])
    ]
    class_counts = {
        hcat_names.get(code, str(code)): int(written_by_class.get(cid, 0))
        for code, cid in sorted(code_to_id.items(), key=lambda kv: kv[1])
    }
    num_written = int(results.get("ok", 0) + results.get("skip", 0))
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Zenodo",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://zenodo.org/records/10118572",
                "have_locally": False,
                "annotation_method": "farmer declaration (CAP/LPIS), harmonized to HCAT",
                "zenodo_record": ZENODO_RECORD,
                "countries": [
                    {"code": c["code"], "year": c["year"], "region": c["region"]}
                    for c in COUNTRIES
                ],
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": classes,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": num_written,
            "class_counts": class_counts,
            "dropped_hcat_codes": [int(x) for x in dropped],
            "notes": (
                "Crop-type parcels from a bounded diverse subset of EuroCrops countries "
                "(PT, ES_NA, AT, DK, EE, HR, NL, SE). Each parcel rasterized into a "
                "<=64x64 UTM 10 m tile: HCAT class id inside the polygon, 255 (nodata/"
                "ignore) outside (no true background class; unlabeled land is ignore). "
                "Class ids assigned 0..N-1 by descending global HCAT-code frequency; "
                f"kept top {len(kept)} of {len(ranked)} codes (dropped {len(dropped)}). "
                "Tiles-per-class balanced with the 25k cap. Time range = 1-year window on "
                "each country's snapshot year."
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

"""Process the JECAM Harmonized In-Situ Datasets into open-set-segmentation labels.

Source: "Harmonized in situ JECAM datasets for agricultural land use mapping and
monitoring in tropical countries" (Jolivot et al. 2021, Earth Syst. Sci. Data 13,
5951-5967, https://doi.org/10.5194/essd-13-5951-2021). Originally on CIRAD Dataverse
(doi:10.18167/DVN1/P7OLAP) but that version is **deaccessioned** ("transferred to another
repository"); the current authoritative copy lives on the CIRAD GeoNetwork/GeoServer at
geode.cirad.fr as WFS layer ``TETIS:BD_JECAM_CIRAD_2023``. Licensed CC-BY-4.0.

It is a set of quality-controlled, field-scale land-use/land-cover **polygons** collected
by local experts under the GEOGLAM/JECAM initiative in seven tropical/subtropical
countries (Burkina Faso, Madagascar, Brazil, Senegal, Kenya, Cambodia, South Africa)
between 2013 and 2022. 31,879 records total (24,287 cropland + 7,592 non-crop). Each
record carries a precise field polygon (WGS84), an acquisition date, a broad ``LandCover``
class and, for cropland, up to three ``CropType`` attributes.

Task: per-pixel **classification** (crop type + land cover). We build ONE unified class
scheme: for ``LandCover == "Cropland"`` the class is the field's ``CropType1`` (the crop),
otherwise the class is the ``LandCover`` value (Built-up surface, Pasture, Forest, Water
body, ...). Class ids are assigned 0..N-1 in **descending frequency**. 86 classes appear
in the post-2016 subset -- comfortably under the 254 uint8 cap, so nothing is dropped.

Each selected field polygon is rasterized into a <=64x64 UTM 10 m tile (tile sized to the
parcel footprint, centered on its centroid, capped at 64): the class id is burned inside
the polygon, everything outside is nodata/ignore (255) -- we only have a ground-truth
label inside surveyed fields, there is no true background class.

Time range: 1-year window anchored on each record's acquisition year (labeled season).
Post-2016 rule: only records with acquisition year >= 2016 are kept (Sentinel era); the
pre-2016 subset (2013-2015, ~8k records) is filtered out.

Sampling: tiles-per-class balanced with the 25k per-dataset cap. With N classes the
effective per-class limit is min(1000, 25000 // N) (``balance_by_class`` default). Rare
classes are prioritized to reach the (reduced) target.

Run (idempotent; skips already-written {sample_id}.tif):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.jecam_harmonized_in_situ_datasets
"""

import argparse
import json
import multiprocessing
from collections import Counter
from typing import Any

import numpy as np
import shapely
import shapely.geometry
import tqdm
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "jecam_harmonized_in_situ_datasets"
NAME = "JECAM Harmonized In-Situ Datasets"

WFS_LAYER = "TETIS:BD_JECAM_CIRAD_2023"
WFS_URL = (
    "https://geode.cirad.fr/geoserver/ows?service=WFS&version=2.0.0"
    "&request=GetFeature&typeNames="
    + WFS_LAYER
    + "&outputFormat=application/json&srsName=EPSG:4326"
)
GEOJSON_NAME = "BD_JECAM_CIRAD_2023.geojson"

MIN_YEAR = 2016  # Sentinel era; drop records acquired before 2016.
MAX_CLASSES = 254  # uint8 (255 = nodata); keep top-N by frequency if exceeded.
PER_CLASS = 1000  # lowered automatically to 25000 // N by balance_by_class.
MAX_TILE = io.MAX_TILE  # 64

_WGS84_SRC = Projection(CRS.from_epsg(4326), 1, 1)


# --------------------------------------------------------------------------------------
# Download (WFS -> one GeoJSON of all field polygons; label-only, no imagery).
# --------------------------------------------------------------------------------------
def ensure_data() -> "object":
    """Download the JECAM WFS layer as a single GeoJSON to raw_dir; return its path."""
    import urllib.request

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    dst = raw / GEOJSON_NAME
    if not dst.exists():
        tmp = raw / (GEOJSON_NAME + ".tmp")
        with urllib.request.urlopen(WFS_URL, timeout=600) as r, tmp.open("wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        tmp.rename(dst)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "JECAM Harmonized In-Situ Datasets (Jolivot et al. 2021), CC-BY-4.0.\n"
            "Original DOI doi:10.18167/DVN1/P7OLAP (CIRAD Dataverse) is DEACCESSIONED;\n"
            "current copy: CIRAD GeoNetwork record 6855571d-677a-4852-afa8-7d7084ed2de8,\n"
            "served as WFS layer TETIS:BD_JECAM_CIRAD_2023 on geode.cirad.fr.\n"
            f"Downloaded via: {WFS_URL}\n"
            "Data paper: https://doi.org/10.5194/essd-13-5951-2021\n"
        )
    return dst


def _label_for(props: dict[str, Any]) -> str | None:
    """Unified class label: CropType1 for cropland, else the LandCover class."""
    lc = (props.get("LandCover") or "").strip()
    if not lc:
        return None
    if lc == "Cropland":
        ct1 = (props.get("CropType1") or "").strip()
        return ct1 or None
    return lc


# --------------------------------------------------------------------------------------
# Pass 2 worker: rasterize one field polygon into a <=64x64 UTM tile.
# --------------------------------------------------------------------------------------
def _write_tile(rec: dict[str, Any]) -> tuple[str, str]:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return sample_id, "skip"
    try:
        geom = shapely.from_wkb(rec["geom_wkb"])  # WGS84 (lon/lat)
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

    geojson_path = ensure_data()
    with geojson_path.open() as f:
        fc = json.load(f)
    feats = fc["features"]
    print(f"loaded {len(feats)} features")

    # ---- Build records: unified label + acquisition year, post-2016 only ----------
    raw_records: list[dict[str, Any]] = []
    n_nogeom = n_prewindow = n_nolabel = 0
    for ft in feats:
        g = ft.get("geometry")
        if g is None:
            n_nogeom += 1
            continue
        p = ft["properties"]
        label = _label_for(p)
        if label is None:
            n_nolabel += 1
            continue
        acq = p.get("AcquiDate")
        year = int(acq[:4]) if acq else None
        if year is None or year < MIN_YEAR:
            n_prewindow += 1
            continue
        raw_records.append({"label": label, "year": year, "geom": g, "id": p.get("Id")})
    print(
        f"kept {len(raw_records)} post-{MIN_YEAR} labeled records "
        f"(dropped: no-geom={n_nogeom}, pre-{MIN_YEAR}={n_prewindow}, no-label={n_nolabel})"
    )

    # ---- Assign class ids by descending frequency (cap at 254) ---------------------
    freq = Counter(r["label"] for r in raw_records)
    ranked = [name for name, _ in freq.most_common()]
    kept = ranked[:MAX_CLASSES]
    dropped = ranked[MAX_CLASSES:]
    label_to_id = {name: i for i, name in enumerate(kept)}
    print(
        f"distinct classes: {len(ranked)}; kept: {len(kept)}; dropped: {len(dropped)}"
    )

    records = [
        {
            "class_id": label_to_id[r["label"]],
            "label": r["label"],
            "year": r["year"],
            "geom": r["geom"],
            "id": r["id"],
        }
        for r in raw_records
        if r["label"] in label_to_id
    ]

    # ---- Tiles-per-class balanced selection with the 25k cap -----------------------
    selected = balance_by_class(
        records, key="class_id", per_class=PER_CLASS, total_cap=25000
    )
    n_classes = len(label_to_id)
    eff_per_class = max(1, min(PER_CLASS, 25000 // n_classes))
    print(f"selected {len(selected)} parcels (eff per-class cap = {eff_per_class})")

    # ---- Prepare tile records (geometry -> wkb + centroid lon/lat) -----------------
    tile_recs: list[dict[str, Any]] = []
    for r in selected:
        geom = shapely.geometry.shape(r["geom"])
        if geom.is_empty:
            continue
        cent = geom.centroid
        if not (np.isfinite(cent.x) and np.isfinite(cent.y)):
            continue
        tile_recs.append(
            {
                "class_id": r["class_id"],
                "lon": float(cent.x),
                "lat": float(cent.y),
                "geom_wkb": shapely.to_wkb(geom),
                "year": r["year"],
                "source_id": f"Id={r['id']}",
            }
        )
    for i, r in enumerate(tile_recs):
        r["sample_id"] = f"{i:06d}"

    io.check_disk()

    # ---- Write tiles in parallel ---------------------------------------------------
    results: Counter = Counter()
    written_by_class: Counter = Counter()
    id_to_rec = {r["sample_id"]: r for r in tile_recs}
    with multiprocessing.Pool(args.workers) as pool:
        for sample_id, res in tqdm.tqdm(
            star_imap_unordered(pool, _write_tile, [dict(rec=r) for r in tile_recs]),
            total=len(tile_recs),
        ):
            results[res] += 1
            if res in ("ok", "skip"):
                written_by_class[id_to_rec[sample_id]["class_id"]] += 1
    print("write results:", dict(results))

    io.check_disk()

    # ---- Metadata ------------------------------------------------------------------
    id_to_label = {i: name for name, i in label_to_id.items()}
    classes = [
        {"id": i, "name": id_to_label[i], "description": None} for i in range(n_classes)
    ]
    class_counts = {
        id_to_label[i]: int(written_by_class.get(i, 0)) for i in range(n_classes)
    }
    num_written = int(results.get("ok", 0) + results.get("skip", 0))
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "CIRAD GeoNetwork/GeoServer (formerly CIRAD Dataverse)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://doi.org/10.5194/essd-13-5951-2021",
                "have_locally": False,
                "annotation_method": "manual field survey (GEOGLAM/JECAM, local experts)",
                "wfs_layer": WFS_LAYER,
                "geonetwork_record": "6855571d-677a-4852-afa8-7d7084ed2de8",
                "deaccessioned_doi": "10.18167/DVN1/P7OLAP",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": classes,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": num_written,
            "class_counts": class_counts,
            "dropped_classes": dropped,
            "notes": (
                "Field-scale crop/land-use polygons from the JECAM harmonized in-situ "
                "database (7 tropical countries, 2013-2022). Unified class scheme: cropland "
                "-> CropType1 (crop type), non-cropland -> LandCover class. Each field "
                "rasterized into a <=64x64 UTM 10 m tile (class id inside polygon, 255 "
                "nodata/ignore outside; no true background class). Class ids assigned "
                f"0..N-1 by descending frequency ({n_classes} classes). Post-2016 records "
                "only (pre-2016 filtered out). Time range = 1-year window on each record's "
                "acquisition year. Tiles-per-class balanced with the 25k cap "
                f"(eff per-class = {eff_per_class})."
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

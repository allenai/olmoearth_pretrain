"""Process Colombia SIMCI coca-cultivation density grid into open-set-segmentation patches.

Source: UNODC-Colombia SIMCI (Sistema Integrado de Monitoreo de Cultivos Ilicitos),
"Densidad de Cultivos de Coca", published on the Colombian open-data portal datos.gov.co
(Socrata dataset id ``v3rx-q7t3``). It is a national 1 km grid: each feature is one
~1 km x 1 km cell (a MultiPolygon in WGS84) carrying, for every census year 2001-2024, a
column ``areaCoca_YYYY`` = the area (hectares) of coca cultivation detected inside that
cell that year (0-100 ha; a cell is 1 km^2 = 100 ha). Coca is detected by SIMCI from
very-high-resolution satellite imagery interpretation with field verification. 119,154
cells cover the monitored coca belt of Colombia (not the whole country).

ACCESS (no credential): openly downloadable from datos.gov.co via the Socrata GeoJSON
export ``https://www.datos.gov.co/resource/v3rx-q7t3.geojson?$limit=150000`` -> one
FeatureCollection cached in raw/{slug}/coca_grid.geojson. Label-only; no imagery pulled.

TASK (spec section 4, derived-product map -> prefer homogeneous / high-confidence cells).
The product is a coarse (1 km) *density* grid, so we treat it as a **weak coca
presence/absence classification** rather than regression: the density is a cell-level
aggregate and cannot be localized within the cell, so a per-pixel regression value would be
spuriously precise. Two classes:

  0 no_coca  <- cell with areaCoca == 0 ha that year (monitored, no coca detected).
                A *hard* negative: these cells lie inside the same coca-suitable belt as the
                positives (the grid only covers historically-monitored areas), not random
                background.
  1 coca     <- cell with areaCoca >= COCA_HA_THRESH (=50 ha, i.e. >= 50% of the 1 km cell
                is coca) that year: a coca-dominated, spatially-homogeneous high-confidence
                cell. The wide excluded gap (1-49 ha) is intentional so both classes stay
                high-confidence/homogeneous per the derived-product-map guidance.

Each qualifying (cell, year) becomes one sample: a 64x64 (640 m) local-UTM 10 m tile,
centered on the cell centroid, **filled uniformly** with the class id (the cell is a
homogeneous ~1 km unit, larger than the 640 m tile). This is a coarse WEAK label:
"this ~640 m area sits in a coca-cultivation-dominated (or coca-free) 1 km cell". See the
summary for the resolution caveat.

TIME (spec section 5): annual product -> 1-year window anchored on each labeled year.
Only post-2016 years are kept (2016-2024; Sentinel era), one (cell, year) sample per
qualifying cell/year. change_time=null (annual presence state, not a dated event).

SAMPLING (spec section 5): classification, up to 1000 samples per class, tiles balanced by
class (25k cap). Positives (>=50 ha) are rarer than the target -> all are kept (spec: keep
sparse classes; downstream assembly filters/negatives). Negatives are drawn from a bounded
random pool of the (very many) zero cells.

task_type=classification, label_type=dense_raster (homogeneous cell tiles).
Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.colombia_simci_coca_monitoring
Idempotent: the raw GeoJSON and existing locations/{id}.tif are skipped on re-run.
"""

import argparse
import multiprocessing
import random
from collections import Counter
from typing import Any

import numpy as np
import tqdm
from rslearn.utils.mp import star_imap_unordered
from shapely.geometry import shape

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "colombia_simci_coca_monitoring"
NAME = "Colombia SIMCI Coca Monitoring"
SOCRATA_ID = "v3rx-q7t3"
GEOJSON_URL = f"https://www.datos.gov.co/resource/{SOCRATA_ID}.geojson?$limit=150000"
PAGE_URL = (
    "https://www.datos.gov.co/Justicia-y-Derecho/"
    "Densidad-de-Cultivos-de-Coca-Subdirecci-n-Estrat-g/v3rx-q7t3"
)

YEARS = list(range(2016, 2025))  # post-2016 (Sentinel era) census years only
# Socrata field name per year (column naming is inconsistent across years in the source).
YEAR_FIELD = {
    2016: "areacoca_2016",
    2017: "areacoca_2017",
    2018: "areacoca_2018",
    2019: "areacoca_2019",
    2020: "areacoca_2020",
    2021: "areacoca_2021",
    2022: "coca2022_",
    2023: "areacoca2023",
    2024: "areacoca2024",
}

TILE = 64  # output tile: 64 px @ 10 m = 640 m (< the 1 km source cell)
COCA_HA_THRESH = 50.0  # >= 50 ha (>=50% of the 1 km cell) => coca-dominated positive
PER_CLASS = 1000
NEG_POOL = 6000  # bounded random pool of zero (no_coca) cell-years to balance from
SEED = 42

CLASSES = [
    (
        0,
        "no_coca",
        "A 1 km SIMCI grid cell in which no coca cultivation was detected that census year "
        "(areaCoca = 0 ha). These cells lie inside Colombia's monitored coca belt, so they are "
        "hard negatives (coca-suitable terrain with no coca that year), not random background.",
    ),
    (
        1,
        "coca",
        "A 1 km SIMCI grid cell whose detected coca cultivation area was >= 50 ha that census "
        "year (>= 50% of the 1 km^2 cell), i.e. a coca-cultivation-dominated cell. Coca is "
        "detected by UNODC-SIMCI from very-high-resolution satellite imagery interpretation "
        "with field verification (annual illicit-crop census).",
    ),
]
CLASS_NAME = {c: n for c, n, _ in CLASSES}


def _val(props: dict[str, Any], field: str) -> float:
    v = props.get(field)
    if v in (None, ""):
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def download_raw() -> Any:
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    dst = raw / "coca_grid.geojson"
    download.download_http(
        GEOJSON_URL, dst, headers={"User-Agent": "Mozilla/5.0"}, timeout=600
    )
    (raw / "SOURCE.txt").write_text(
        "Colombia SIMCI 'Densidad de Cultivos de Coca' (annual coca-cultivation density\n"
        "grid). UNODC-Colombia SIMCI, published on datos.gov.co (Socrata id v3rx-q7t3).\n"
        f"Portal: {PAGE_URL}\n"
        f"Downloaded via Socrata GeoJSON export: {GEOJSON_URL}\n"
        "One MultiPolygon feature per ~1 km x 1 km cell; per-year areaCoca_YYYY = hectares\n"
        "of coca detected in the cell (0-100 ha). License: Colombia open government data.\n"
    )
    return dst


def build_candidates(geojson_path: str) -> list[dict[str, Any]]:
    """Return (cell, year) candidate records for the coca (>=50 ha) and no_coca (0 ha) classes.

    Positives: every qualifying coca cell-year (rare -> keep all). Negatives: a bounded
    random pool of zero cell-years (there are ~721k; we only need <=1000 after balancing).
    Centroids are computed only for the candidates we keep.
    """
    import json

    with open(geojson_path) as f:
        fc = json.load(f)
    feats = fc["features"]

    coca: list[dict[str, Any]] = []
    zero_idx: list[tuple[int, int]] = []  # (feature_index, year) for zero cells
    for fi, feat in enumerate(feats):
        pr = feat["properties"]
        grilla = pr.get("grilla1")
        for y in YEARS:
            area = _val(pr, YEAR_FIELD[y])
            if area >= COCA_HA_THRESH:
                coca.append(
                    {
                        "fi": fi,
                        "grilla": grilla,
                        "year": y,
                        "area": area,
                        "label": 1,
                    }
                )
            elif area == 0.0:
                zero_idx.append((fi, y))

    rng = random.Random(SEED)
    rng.shuffle(zero_idx)
    zero_idx = zero_idx[:NEG_POOL]
    neg = [
        {
            "fi": fi,
            "grilla": feats[fi]["properties"].get("grilla1"),
            "year": y,
            "area": 0.0,
            "label": 0,
        }
        for (fi, y) in zero_idx
    ]

    # Compute centroid (WGS84 lon/lat) once per kept candidate feature.
    need = {r["fi"] for r in coca} | {r["fi"] for r in neg}
    centroid: dict[int, tuple[float, float]] = {}
    for fi in need:
        c = shape(feats[fi]["geometry"]).centroid
        centroid[fi] = (float(c.x), float(c.y))

    out = []
    for r in coca + neg:
        lon, lat = centroid[r["fi"]]
        out.append(
            {
                "lon": lon,
                "lat": lat,
                "year": r["year"],
                "label": r["label"],
                "area": r["area"],
                "grilla": r["grilla"],
                "source_id": f"{r['grilla']}|{r['year']}",
            }
        )
    return out


def write_one(rec: dict[str, Any]) -> None:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return
    dst_proj, col, row = io.lonlat_to_utm_pixel(rec["lon"], rec["lat"])
    bounds = io.centered_bounds(col, row, TILE, TILE)
    label = int(rec["label"])
    arr = np.full((TILE, TILE), label, dtype=np.uint8)
    io.write_label_geotiff(
        SLUG, sample_id, arr, dst_proj, bounds, nodata=io.CLASS_NODATA
    )
    io.write_sample_json(
        SLUG,
        sample_id,
        dst_proj,
        bounds,
        io.year_range(rec["year"]),
        change_time=None,
        source_id=rec["source_id"],
        classes_present=[label],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    geojson_path = download_raw()
    io.check_disk()

    cands = build_candidates(geojson_path.path)
    cand_counts = Counter(r["label"] for r in cands)
    print(
        f"candidates: coca(>= {COCA_HA_THRESH:.0f} ha)={cand_counts.get(1, 0)}, "
        f"no_coca pool={cand_counts.get(0, 0)}"
    )

    selected = balance_by_class(cands, "label", per_class=PER_CLASS, seed=SEED)
    for i, r in enumerate(sorted(selected, key=lambda x: x["source_id"])):
        r["sample_id"] = f"{i:06d}"
    sel_counts = Counter(r["label"] for r in selected)
    print(
        f"selected {len(selected)}: "
        f"{ {CLASS_NAME[k]: sel_counts.get(k, 0) for k in (0, 1)} }"
    )

    io.check_disk()
    with multiprocessing.Pool(args.workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            pass

    # Per-year positive breakdown for the summary.
    year_pos = Counter(r["year"] for r in selected if r["label"] == 1)

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "UNODC-Colombia SIMCI (via datos.gov.co / ODC)",
            "license": "Colombia open government data (datos.gov.co)",
            "provenance": {
                "url": PAGE_URL,
                "socrata_id": SOCRATA_ID,
                "have_locally": False,
                "annotation_method": (
                    "UNODC-SIMCI annual coca census: coca cultivation detected from "
                    "very-high-resolution satellite imagery interpretation with field "
                    "verification, aggregated to a 1 km grid (areaCoca ha per cell/year)."
                ),
                "accessed_via": f"Socrata GeoJSON export {GEOJSON_URL}",
                "native_grid_m": 1000,
                "years": YEARS,
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [{"id": i, "name": n, "description": d} for i, n, d in CLASSES],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {CLASS_NAME[k]: sel_counts.get(k, 0) for k in (0, 1)},
            "positives_per_year": {str(y): year_pos.get(y, 0) for y in YEARS},
            "notes": (
                "WEAK presence/absence label from the SIMCI 1 km coca-density grid "
                "(datos.gov.co v3rx-q7t3). Each sample is a 64x64 (640 m) local-UTM 10 m "
                "tile centered on a 1 km cell and filled uniformly with the cell's class. "
                f"coca(1) = cell areaCoca >= {COCA_HA_THRESH:.0f} ha (>=50% of the 1 km "
                "cell that year); no_coca(0) = cell areaCoca == 0 ha (hard negative within "
                "the monitored coca belt). The 1-49 ha gap is excluded so both classes are "
                "homogeneous/high-confidence. RESOLUTION CAVEAT: the source is a 1 km "
                "aggregate, so coca is not localized within the cell and the uniform tile "
                "label is region-level, not per-pixel exact; positives are coca-DOMINATED "
                "cells. Annual product: 1-year window anchored on each labeled year, "
                "post-2016 only (2016-2024), change_time=null. Up to 1000/class, tiles "
                "balanced by class; positives are all kept (below target)."
            ),
        },
    )
    print(f"num_samples={len(selected)} task_type=classification")
    manifest.write_registry_entry(
        SLUG,
        "completed",
        task_type="classification",
        num_samples=len(selected),
        notes=(
            f"coca={sel_counts.get(1, 0)} (>= {COCA_HA_THRESH:.0f} ha/1km cell), "
            f"no_coca={sel_counts.get(0, 0)}; 640 m uniform weak-label tiles from SIMCI "
            "1 km density grid, 2016-2024 annual windows."
        ),
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()

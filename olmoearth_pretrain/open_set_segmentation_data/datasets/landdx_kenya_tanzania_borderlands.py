"""landDX (Kenya-Tanzania borderlands) -> open-set-segmentation pastoral-structure masks.

Source: "Landscape Dynamics (landDX): an open-access spatial-temporal database for the
Kenya-Tanzania borderlands" (Tyrrell et al. 2022, Scientific Data 9:8,
doi 10.1038/s41597-021-01100-9; data DOI 10.5287/bodleian:qqv4EdRnQ, Oxford University
Research Archive, CC-BY-4.0). Manual VHR (Google Earth / Bing, ~0.5 m, a few areas 30 m
Landsat) digitization of anthropogenic structures across ~31,000 km2 of southern Kenya
(Kajiado + Narok counties) by SORALO, Kenya Wildlife Trust, Aarhus University and the
Mara Elephant Project. The static ORA release ships four ESRI shapefiles:
  - landDx_polygons     : 57,192 polygons -> type Settlement_Boma (37,040 livestock
                          enclosures) + Agriculture (20,152 farmland polygons).
  - landDx_polylines    : 96,879 lines    -> Fence_* (94,546) + Road_* (2,324).
  - landDx_points       : 31,024 points   -> boma centroids (redundant w/ polygons).
  - landDx_polygons_centroids : 57,080 polygon centroids (redundant).

Unified class scheme (spec S5 "combine multi-modality into ONE dataset"):
  0 = livestock_enclosure  (Settlement_Boma polygons; boma/kraal/enkang)
  1 = agricultural_land    (Agriculture polygons)
This is a **classification** (per-pixel segmentation) task, **positive-only** (spec S5):
non-labeled pixels are nodata/ignore (255); we do NOT fabricate a background class -- the
assembly step supplies negatives from other datasets.

Dropped modalities / classes (documented, per the task's observability judgment):
  * Fencing (Fence_* polylines): a **thin line** feature. Brush fences are a few metres
    wide and wire fences are invisible even in VHR (mapped only via land-use edges); the
    source further carries a ~39.7 m Google-Earth positional RMSE (Tyrrell et al. 2022,
    citing Potere 2008). A ~40 m location error on a sub-10 m-wide line means a dilated
    10 m mask would frequently not overlie the real feature -> not reliably observable /
    alignable at 10-30 m from Sentinel/Landsat. Dropped (spec S4 "lines: reject if the
    feature is not observable at 10-30 m").
  * Roads (Road_* polylines): out of the manifest's 3-class scope; also thin. Dropped.
  * Boma points / polygon centroids: redundant with the boma polygons (which give the
    real footprint). Not used.

Observability of the kept classes at 10 m: bomas are 30-150 m cleared enclosures
(equiv-side median ~25 m ~ 2-3 px, p95 ~65 m ~ 6-7 px) with a distinctive bare-earth /
manure spectral signature -> discernible at 10-30 m (per the manifest note); the ~40 m
positional RMSE offsets small bomas but larger ones and the field-scale agriculture
polygons (equiv-side median ~102 m) tolerate it. ACCEPTED for bomas + agriculture.

Time range (spec S5): each feature carries collect_da (the digitized imagery / ground
date). Dated features in [2016, 2022] use a 1-year window on their year. Dated features
BEFORE 2016 (pre-Sentinel imagery, e.g. 2003-2015 Google Earth) are dropped (spec S2
triage: keep only the post-2016 subset of a mixed dataset). Undated features (KWT, whose
imagery is <=2017; some SORALO with no GE date stamp, imagery <=2020) are kept with a
DEFAULT_YEAR window -- undated is not known-pre-2016, and the SORALO weighted-mean date is
2016-09. These are persistent-ish land features, so change_time is null and a static
1-year window is used.

Tiling (spec S4 "polygons ... sampled sub-windows for large/dense coverage"): the study
area is partitioned onto a 640 m grid in World Mollweide (ESRI:54009); each occupied cell
becomes one 64x64 (640 m) UTM 10 m tile centered on the cell center, into which every boma
/ agriculture polygon overlapping the cell is rasterized (clipped, agriculture first then
bomas on top so bomas win). The 1-year window is anchored on the modal effective year of a
cell's features. Tiles-per-class balanced selection (spec S5) keeps up to 1000 tiles per
class.

Run (idempotent; skips already-written tiles):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.landdx_kenya_tanzania_borderlands
"""

import argparse
import math
import multiprocessing
import zipfile
from collections import Counter, defaultdict
from typing import Any

import fiona
import shapely
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered
from shapely.geometry import shape
from shapely.ops import transform as shp_transform

from olmoearth_pretrain.open_set_segmentation_data import (
    download,
    io,
    manifest,
    sampling,
)
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)

SLUG = "landdx_kenya_tanzania_borderlands"
NAME = "landDX (Kenya-Tanzania Borderlands)"

# Oxford University Research Archive static release (CC-BY-4.0).
ORA_UUID = "a733ec4f-20e3-4989-acba-5f85cfd6d0eb"
ORA_FILE_ID = "ddv13zt283"  # active_public_uncategorized_shpfiles.zip
ZIP_NAME = "active_public_uncategorized_shpfiles.zip"
SHP_DIRNAME = "active_shp"
POLY_SHP = "landDx_polygons.shp"

# Source polygons are WGS84 (EPSG:4326); we grid in World Mollweide (metric) exactly as
# the congo_basin_forest_roads script does, so a Projection with res (1,1) makes geometry
# coords == metres for geom_to_pixels.
MOLL_CRS = CRS.from_string("ESRI:54009")
MOLL_PROJ = Projection(MOLL_CRS, 1, 1)

TILE = 64  # 640 m tiles.
CELL_M = TILE * io.RESOLUTION  # 640 m grid cells in the source (Mollweide) CRS.

YEAR_MIN = 2016  # Sentinel era / manifest lower bound.
YEAR_MAX = 2022  # manifest upper bound.
DEFAULT_YEAR = 2017  # undated features: KWT imagery <=2017, SORALO wtd-mean 2016-09.

CID_BOMA = 0
CID_AG = 1
CLASSES = [
    {
        "id": CID_BOMA,
        "name": "livestock_enclosure",
        "description": (
            "A livestock enclosure (Maasai boma / enkang / kraal): a 30-150 m cleared, "
            "fenced holding area for cattle/shoats, recognisable by a distinctive "
            "bare-earth / manure spectral signature and a continuous fenced perimeter. "
            "Manually digitized from VHR Google Earth / Bing imagery (SORALO, Kenya "
            "Wildlife Trust). Rasterized from the digitized polygon footprint; only "
            "areas clearly in use to hold livestock at the imagery date were mapped."
        ),
    },
    {
        "id": CID_AG,
        "name": "agricultural_land",
        "description": (
            "Agricultural land use (subsistence and mechanized cropland) in the "
            "southern-Kenya rangelands, manually digitized as polygons from VHR Google "
            "Earth / Bing imagery (SORALO + Kenya Wildlife Trust). Field-scale patches "
            "(equiv-side median ~100 m) delineating cultivated land distinct from the "
            "surrounding savanna vegetation."
        ),
    },
]

_TO_WGS84 = None  # lazily-built pyproj transformer (per process): Mollweide -> lon/lat.
_TO_MOLL = None  # lazily-built pyproj transformer: WGS84 -> Mollweide.


def _lonlat(x: float, y: float) -> tuple[float, float]:
    """ESRI:54009 (x, y) metres -> (lon, lat) degrees."""
    global _TO_WGS84
    if _TO_WGS84 is None:
        from pyproj import Transformer

        _TO_WGS84 = Transformer.from_crs("ESRI:54009", 4326, always_xy=True)
    return _TO_WGS84.transform(x, y)


def _to_moll(geom: Any) -> Any:
    """Reproject a WGS84 shapely geometry to World Mollweide (metres)."""
    global _TO_MOLL
    if _TO_MOLL is None:
        from pyproj import Transformer

        _TO_MOLL = Transformer.from_crs(4326, "ESRI:54009", always_xy=True)
    return shp_transform(lambda xx, yy: _TO_MOLL.transform(xx, yy), geom)


def _download_and_extract() -> None:
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    shp = raw / SHP_DIRNAME / POLY_SHP
    if not shp.exists():
        zip_path = raw / ZIP_NAME
        if not zip_path.exists():
            print(
                "downloading landDX active shapefiles from Oxford Research Archive ..."
            )
            download.download_http(
                f"https://ora.ox.ac.uk/objects/uuid:{ORA_UUID}/files/{ORA_FILE_ID}",
                zip_path,
            )
        print("extracting shapefiles ...")
        with zipfile.ZipFile(zip_path.path) as z:
            z.extractall((raw / SHP_DIRNAME).path)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "Landscape Dynamics (landDX) database -- Kenya-Tanzania borderlands.\n"
            "Tyrrell, P. et al. (2022). Scientific Data 9, 8. "
            "doi:10.1038/s41597-021-01100-9\n"
            "Data (static release): Oxford University Research Archive, "
            "doi:10.5287/bodleian:qqv4EdRnQ  (CC-BY-4.0).\n"
            f"File: {ZIP_NAME} (active public uncategorized shapefiles).\n"
            "Manual VHR (Google Earth/Bing) digitization of livestock enclosures "
            "(bomas), agricultural land (polygons), and fencing/roads (polylines) over "
            "~31,000 km2 of southern Kenya (Kajiado + Narok). WGS84 (EPSG:4326).\n"
        )


def _effective_year(collect_da: Any) -> int | None:
    """Map a feature's collect date to its effective 1-year-window year, or None to drop.

    - dated in [YEAR_MIN, YEAR_MAX] -> that year.
    - dated before YEAR_MIN (pre-Sentinel imagery) -> None (drop the feature).
    - dated after YEAR_MAX -> clamp to YEAR_MAX (none occur in practice).
    - undated -> DEFAULT_YEAR (kept; not known-pre-2016).
    """
    if not collect_da:
        return DEFAULT_YEAR
    try:
        yr = int(str(collect_da)[:4])
    except (ValueError, TypeError):
        return DEFAULT_YEAR
    if yr < YEAR_MIN:
        return None
    if yr > YEAR_MAX:
        return YEAR_MAX
    return yr


def build_cells() -> dict[tuple[int, int], dict[str, Any]]:
    """Partition boma + agriculture polygons onto a 640 m Mollweide grid.

    Returns cell (ix, iy) -> {"shapes": [(moll_wkb, class_id)], "years": Counter,
    "classes": set}. A polygon is added to every grid cell its bbox overlaps (bomas span
    ~1 cell; large ag polygons span several -- clipped at rasterization).
    """
    shp = (io.raw_dir(SLUG) / SHP_DIRNAME / POLY_SHP).path
    cells: dict[tuple[int, int], dict[str, Any]] = defaultdict(
        lambda: {"shapes": [], "years": Counter(), "classes": set()}
    )
    kept = Counter()
    dropped_year = Counter()
    dropped_null = 0
    with fiona.open(shp) as src:
        for feat in src:
            pr = feat["properties"]
            t = pr.get("type")
            if t == "Settlement_Boma":
                cid = CID_BOMA
            elif t == "Agriculture":
                cid = CID_AG
            else:
                continue
            year = _effective_year(pr.get("collect_da"))
            if year is None:
                dropped_year[t] += 1
                continue
            if feat["geometry"] is None:
                dropped_null += 1
                continue
            geom = shape(feat["geometry"])
            if geom.is_empty:
                dropped_null += 1
                continue
            geom = _to_moll(geom)
            if geom.is_empty:
                dropped_null += 1
                continue
            wkb = shapely.to_wkb(geom)
            kept[t] += 1
            minx, miny, maxx, maxy = geom.bounds
            ix0, ix1 = math.floor(minx / CELL_M), math.floor(maxx / CELL_M)
            iy0, iy1 = math.floor(miny / CELL_M), math.floor(maxy / CELL_M)
            for ix in range(ix0, ix1 + 1):
                for iy in range(iy0, iy1 + 1):
                    c = cells[(ix, iy)]
                    c["shapes"].append((wkb, cid))
                    c["years"][year] += 1
                    c["classes"].add(cid)
    print(f"  kept polygons: {dict(kept)}")
    print(f"  dropped (pre-{YEAR_MIN} date): {dict(dropped_year)}")
    print(f"  dropped (null/empty geom): {dropped_null}")
    return cells


def _write_one(rec: dict[str, Any]) -> str:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return "skip"

    ix, iy = rec["cell"]
    cx = (ix + 0.5) * CELL_M
    cy = (iy + 0.5) * CELL_M
    lon, lat = _lonlat(cx, cy)
    proj = io.utm_projection_for_lonlat(lon, lat)
    _, col, row = io.lonlat_to_utm_pixel(lon, lat, proj)
    bounds = io.centered_bounds(col, row, TILE, TILE)

    # Rasterize agriculture first, bomas second, so bomas win on overlap.
    ag_shapes: list[tuple[Any, int]] = []
    boma_shapes: list[tuple[Any, int]] = []
    for wkb, cid in rec["shapes"]:
        geom = shapely.from_wkb(wkb)
        try:
            pix = geom_to_pixels(geom, MOLL_PROJ, proj)
        except Exception:
            continue
        if pix.is_empty:
            continue
        (boma_shapes if cid == CID_BOMA else ag_shapes).append((pix, cid))
    shapes = ag_shapes + boma_shapes
    if not shapes:
        return "empty"

    arr = rasterize_shapes(
        shapes, bounds, fill=io.CLASS_NODATA, dtype="uint8", all_touched=True
    )[0]
    present = [int(c) for c in (CID_BOMA, CID_AG) if int((arr == c).sum()) > 0]
    if not present:
        return "empty"

    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(rec["year"]),
        change_time=None,
        source_id=f"cell_{ix}_{iy}",
        classes_present=present,
    )
    return "written"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--per-class", type=int, default=1000)
    parser.add_argument(
        "--probe", action="store_true", help="scan/report only, no writes"
    )
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    _download_and_extract()

    print("partitioning boma + agriculture polygons onto a 640 m grid ...")
    cells = build_cells()
    print(f"  {len(cells)} occupied cells")

    io.check_disk()

    # One candidate record per occupied cell; anchor the window on the modal effective
    # year of the cell's features; classes_present is the candidate class set in the cell.
    records: list[dict[str, Any]] = []
    for (ix, iy), c in cells.items():
        modal_year = c["years"].most_common(1)[0][0]
        records.append(
            {
                "cell": (ix, iy),
                "shapes": c["shapes"],
                "year": int(modal_year),
                "classes_present": sorted(c["classes"]),
            }
        )

    # Tiles-per-class balanced selection: up to per_class tiles per class (spec S5).
    selected = sampling.balance_tiles_by_class(
        records, classes_key="classes_present", per_class=args.per_class
    )
    print(f"selected {len(selected)} candidate cells (<= {args.per_class}/class)")

    selected.sort(key=lambda r: r["cell"])  # stable id assignment
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"

    cand_class_counts = Counter()
    for r in selected:
        for c in r["classes_present"]:
            cand_class_counts[c] += 1
    print("candidate tiles-per-class:", dict(sorted(cand_class_counts.items())))
    print(
        "anchor-year distribution:",
        dict(sorted(Counter(r["year"] for r in selected).items())),
    )

    if args.probe:
        print("probe only; exiting before writes")
        return

    results: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for res in star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]):
            results[res] += 1
    print("write results:", dict(results))

    io.check_disk()

    # Recompute per-class / anchor-year counts from the tiles actually on disk (some
    # selected cells rasterize empty), so metadata is accurate and stable across re-runs.
    import json as _json

    class_counts: Counter = Counter()
    year_hist: Counter = Counter()
    num_samples = 0
    for jp in io.locations_dir(SLUG).glob("*.json"):
        with jp.open() as _f:
            meta = _json.load(_f)
        num_samples += 1
        year_hist[int(meta["time_range"][0][:4])] += 1
        for c in meta.get("classes_present", []):
            class_counts[int(c)] += 1

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Mara Elephant Project / SORALO / KWT / Aarhus (Sci Data)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://doi.org/10.1038/s41597-021-01100-9",
                "data_doi": "https://doi.org/10.5287/bodleian:qqv4EdRnQ",
                "have_locally": False,
                "annotation_method": (
                    "manual VHR (Google Earth / Bing, ~0.5 m; a few areas 30 m Landsat) "
                    "digitization of livestock enclosures and agricultural land"
                ),
                "citation": (
                    "Tyrrell, P., Amoke, I., Betjes, K. et al. (2022). Landscape "
                    "Dynamics (landDX) an open-access spatial-temporal database for the "
                    "Kenya-Tanzania borderlands. Scientific Data 9, 8. "
                    "doi:10.1038/s41597-021-01100-9"
                ),
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": CLASSES,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": num_samples,
            "class_tile_counts": {str(k): v for k, v in sorted(class_counts.items())},
            "anchor_year_counts": {str(k): v for k, v in sorted(year_hist.items())},
            "notes": (
                "Positive-only per-pixel segmentation of pastoral structures in the "
                "southern-Kenya rangelands. Unified 2-class scheme: 0=livestock_enclosure "
                "(Settlement_Boma polygons), 1=agricultural_land (Agriculture polygons); "
                "non-labeled pixels = nodata (255). Bomas + agriculture polygons "
                "rasterized (all_touched, ag first then bomas on top) into 64x64 UTM 10 m "
                "tiles on a 640 m grid (World Mollweide). Fencing (Fence_* polylines, "
                "94,546) and roads (Road_*, 2,324) were DROPPED: thin line features with "
                "a ~39.7 m Google-Earth positional RMSE are not reliably observable / "
                "alignable at 10-30 m from Sentinel/Landsat (wire fences are invisible "
                "even in VHR). Boma points / polygon centroids are redundant with the "
                "polygons and unused. Time range: 1-year static window (change_time=null) "
                "on each feature's collect date; dated features before 2016 (pre-Sentinel "
                "imagery, 2003-2015) dropped, undated features (KWT <=2017; some SORALO "
                f"<=2020) kept with a {DEFAULT_YEAR} window (SORALO weighted-mean date "
                "2016-09). Per-cell window anchored on the modal feature year. "
                "Tiles-per-class balanced to <=1000 tiles/class. Caveat: manual VHR "
                "digitization with ~40 m positional error; small bomas (~25 m median "
                "equiv-side, 2-3 px) are near the 10 m limit and may be offset, though "
                "larger bomas and field-scale agriculture tolerate it. Boma occupancy is "
                "seasonal, so a boma mapped in one year may be absent in the imagery of "
                "an adjacent year."
            ),
        },
    )

    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=num_samples
    )
    print(
        f"done: {num_samples} samples; class_tile_counts="
        f"{dict(sorted(class_counts.items()))}; "
        f"anchor-year={dict(sorted(year_hist.items()))}"
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()

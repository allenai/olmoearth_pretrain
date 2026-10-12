"""Process USGS ASTER Hydrothermal Alteration Maps (OFR 2013-1139) into label patches.

Source: USGS Open-File Report 2013-1139, "Hydrothermal Alteration Maps of the Central and
Southern Basin and Range Province of the United States Compiled From ASTER Data" (Mars,
2013; https://mrdata.usgs.gov/surficial-mineralogy/ofr-2013-1139/, CC0/public domain).
ASTER VNIR-SWIR data + IDL logical-operator band-ratio algorithms were used to map
surficial minerals diagnostic of hydrothermal alteration (permissive of gold/copper
deposits) across the Basin and Range (approx. lon -120.4..-107.4, lat 30.6..42.4). Native
ASTER resolution 15-90 m (SWIR 30 m); the manifest notes alteration is discernible in
S2/Landsat SWIR, so labels are resampled to 10 m UTM.

Distribution / access (frugal, no auth): the product is published only as **polygon
shapefiles** (not rasters), one shapefile PER alteration type -- the alteration type is a
property of the whole layer, not a per-feature attribute (see the FGDC eainfo). We download
the five small per-type zips (~588 MB total) and rasterize them; no GeoTIFF layer exists to
download. The five layers map to a UNIFIED class scheme (spec 5, "combine mineral groups
into one class map"):

  0 argillic        advanced-argillic (alunite-pyrophyllite-kaolinite)
  1 phyllic         phyllic/sericitic (sericite-muscovite/illite)
  2 epi_chlor       propylitic, epidote-chlorite(-albite) mineral group (report: propylitic)
  3 carbonate       calcite-dolomite mineral group (part of propylitic in the report)
  4 hydro_silica    hydrothermal silica-rich (hydrous quartz, chalcedony, opal, am. silica)

Note on the manifest class list ("advanced argillic, phyllic, propylitic, clays,
carbonates, iron oxides"): it is aspirational -- the actual OFR 2013-1139 product has
exactly the five mineral-group layers above (no distinct "clays" or "iron oxide" layer;
kaolinite/clays fall inside the argillic layer, and there is no iron-oxide layer). We use
the five real layers and document the deviation in the summary.

Positive-only (spec 5): this is a foreground alteration map -- polygons mark WHERE
alteration is detected; unaltered ground is left as nodata (255), not a fabricated
background class. The pretraining-assembly step supplies negatives from other datasets.

Method:
  1. Candidate windows: snap every polygon centroid to a ~640 m lon/lat grid cell (= one
     64 px @ 10 m tile footprint) and count centroids per (cell, class). A class counts as
     present in a cell when >= MIN_POLYS centroids fall in it (a homogeneity / high-
     confidence proxy: ~>=10% of the cell's native pixels).
  2. Tiles-per-class balanced selection (spec 5) over candidate cells, <= 1000 tiles/class,
     25k total cap (well under -- 5 classes => <= 5000 tiles).
  3. Rasterize: for each selected cell, query the actual polygons of every layer that
     intersect the tile (shapely STRtree per layer) and burn them into a 64x64 local-UTM
     10 m tile. Overlapping alteration (co-occurring minerals) is resolved rarest-class-
     wins (layers burned most-common -> rarest so rare classes survive overlaps). Native
     WGS84 polygons -> local UTM at 10 m; categorical => exact polygon burn (no bilinear).

Time range: static geologic label -> a representative 1-year Sentinel-era window (2016;
manifest time_range [2016, 2016]); change_time = null.

Run:  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.usgs_aster_hydrothermal_alteration_maps
"""

import argparse
import multiprocessing
import time
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pyogrio
import shapely
import tqdm
from pyproj import Transformer
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io
from olmoearth_pretrain.open_set_segmentation_data.rasterize import rasterize_shapes
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    select_tiles_per_class,
)

SLUG = "usgs_aster_hydrothermal_alteration_maps"

RAW = io.raw_dir(SLUG)

# (class_id, layer_name, human_name, description). class_id is the label value written.
LAYERS: list[tuple[int, str, str, str]] = [
    (
        0,
        "argillic",
        "argillic",
        "Advanced-argillic hydrothermal alteration: alunite-pyrophyllite-kaolinite mineral "
        "assemblage, mapped from ASTER VNIR-SWIR band-ratio logical operators.",
    ),
    (
        1,
        "phyllic",
        "phyllic",
        "Phyllic (sericitic) hydrothermal alteration: sericite-muscovite (illite) assemblage.",
    ),
    (
        2,
        "epi_chlor",
        "propylitic_epidote_chlorite",
        "Propylitic alteration mapped as the epidote-chlorite(-albite) mineral group (labeled "
        "epi_chlor in the source shapefiles; the report text refers to this as propylitic).",
    ),
    (
        3,
        "carbonate",
        "carbonate",
        "Calcite-dolomite (carbonate) mineral group; treated as part of propylitic alteration "
        "in the report.",
    ),
    (
        4,
        "hydro_silica",
        "hydrothermal_silica",
        "Hydrothermal silica-rich rocks: hydrous quartz, chalcedony, opal, and amorphous "
        "silica.",
    ),
]
CID_TO_NAME = {
    cid: name for cid, name, _h, _d in [(l[0], l[1], l[2], l[3]) for l in LAYERS]
}
CID_TO_HUMAN = {cid: h for cid, _n, h, _d in LAYERS}

# Grid cell ~= a 64 px @ 10 m (=640 m) tile footprint, in degrees at the dataset mid-lat
# (~36.5 deg): lat 640/111320 ~= 0.00575, lon 640/(111320*cos36.5) ~= 0.00715.
LON_CELL = 0.00715
LAT_CELL = 0.00575
TILE = 64
MIN_POLYS = 10  # >= this many centroids of a class in a cell => class "present"
PER_CLASS = 1000  # tiles-per-class target (25k cap; 5 classes => well under)
YEAR = 2016  # representative Sentinel-era 1-year window (manifest [2016,2016])

# Rasterization order: most-common layer first so rarest is burned last and wins overlaps.
# Polygon counts: epi_chlor 1.72M > phyllic 1.09M > hydro_silica 0.94M > argillic 0.93M >
# carbonate 0.50M.
BURN_ORDER = ["epi_chlor", "phyllic", "hydro_silica", "argillic", "carbonate"]
NAME_TO_CID = {name: cid for cid, name, _h, _d in LAYERS}

# Positive int cell-key encoding (ix can be negative in the western US; iy positive).
_IXOFF = 100000
_IYOFF = 100000
_IYSPAN = 1000000


def _layer_path(name: str):
    return RAW / name / f"{name}.shp"


def _encode(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    return (ix + _IXOFF) * _IYSPAN + (iy + _IYOFF)


def _decode(key: int) -> tuple[int, int]:
    return key // _IYSPAN - _IXOFF, key % _IYSPAN - _IYOFF


def _scan_layer(cid: int, name: str) -> tuple[int, dict[int, int]]:
    """Return (cid, {cell_key: centroid_count}) for one alteration layer."""
    b = pyogrio.read_dataframe(str(_layer_path(name)), columns=[]).geometry.bounds
    cx = ((b.minx + b.maxx) / 2).to_numpy()
    cy = ((b.miny + b.maxy) / 2).to_numpy()
    ix = np.floor(cx / LON_CELL).astype(np.int64)
    iy = np.floor(cy / LAT_CELL).astype(np.int64)
    keys = _encode(ix, iy)
    u, c = np.unique(keys, return_counts=True)
    return cid, dict(zip(u.tolist(), c.tolist()))


def _cell_center(key: int) -> tuple[float, float]:
    ix, iy = _decode(key)
    return (ix + 0.5) * LON_CELL, (iy + 0.5) * LAT_CELL


def _tile_geom(lon: float, lat: float) -> dict[str, Any]:
    """Compute UTM projection, integer pixel bounds and the lon/lat query bbox for a tile."""
    proj, col, row = io.lonlat_to_utm_pixel(lon, lat)
    bounds = io.centered_bounds(col, row, TILE, TILE)
    # Tile corners in UTM metres (pixel*res; y_res=-10) -> lon/lat bbox for spatial query.
    xs = [bounds[0] * io.RESOLUTION, bounds[2] * io.RESOLUTION]
    ys = [bounds[1] * -io.RESOLUTION, bounds[3] * -io.RESOLUTION]
    tf = Transformer.from_crs(proj.crs, "EPSG:4326", always_xy=True)
    lons, lats = tf.transform(
        [xs[0], xs[1], xs[0], xs[1]], [ys[0], ys[1], ys[1], ys[0]]
    )
    pad = 0.0015  # ~150 m so polygons straddling the edge are captured
    return {
        "proj": proj,
        "bounds": bounds,
        "qbox": (min(lons) - pad, min(lats) - pad, max(lons) + pad, max(lats) + pad),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()

    # --- Phase 1: per-cell per-class centroid counts (parallel over the 5 layers) ---
    print("Scanning layer polygon centroids into grid cells...")
    cell_counts: dict[int, dict[int, int]] = defaultdict(dict)
    t0 = time.time()
    with multiprocessing.Pool(5) as p:
        for cid, counts in star_imap_unordered(
            p, _scan_layer, [dict(cid=cid, name=name) for cid, name, _h, _d in LAYERS]
        ):
            for key, cnt in counts.items():
                cell_counts[key][cid] = cnt
    print(f"  {len(cell_counts)} occupied cells ({time.time() - t0:.0f}s)")

    # --- Phase 2: candidate records + tiles-per-class balanced selection ---
    candidates: list[dict[str, Any]] = []
    for key, dd in cell_counts.items():
        present = sorted(cid for cid, cnt in dd.items() if cnt >= MIN_POLYS)
        if not present:
            continue
        lon, lat = _cell_center(key)
        candidates.append(
            {"key": int(key), "lon": lon, "lat": lat, "present_ids": present}
        )
    print(f"{len(candidates)} candidate cells (>= {MIN_POLYS} centroids of some class)")

    selected = select_tiles_per_class(
        candidates, classes_key="present_ids", per_class=PER_CLASS
    )
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
        g = _tile_geom(r["lon"], r["lat"])
        r.update(g)
    print(
        f"selected {len(selected)} tiles (tiles-per-class, <= {PER_CLASS}/class, 25k cap)"
    )

    # --- Phase 3: rasterize actual polygons into per-tile arrays (rarest-class wins) ---
    # accumulator per selected tile, initialised to nodata (255 = unaltered / not mapped).
    acc: dict[str, np.ndarray] = {
        r["sample_id"]: np.full((TILE, TILE), io.CLASS_NODATA, dtype=np.uint8)
        for r in selected
    }
    # cache one WGS84->UTM transformer per CRS (only ~3 UTM zones across Basin & Range).
    tf_cache: dict[str, Transformer] = {}

    def to_pixels(geoms: np.ndarray, crs: str) -> np.ndarray:
        tf = tf_cache.get(crs)
        if tf is None:
            tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            tf_cache[crs] = tf

        def _fn(coords: np.ndarray) -> np.ndarray:
            x, y = tf.transform(coords[:, 0], coords[:, 1])
            return np.column_stack(
                [np.asarray(x) / io.RESOLUTION, np.asarray(y) / -io.RESOLUTION]
            )

        return shapely.transform(geoms, _fn)

    for name in BURN_ORDER:
        cid = NAME_TO_CID[name]
        t0 = time.time()
        gdf = pyogrio.read_dataframe(str(_layer_path(name)), columns=[])
        geoms = gdf.geometry.to_numpy()
        tree = shapely.STRtree(geoms)
        n_hit = 0
        for r in tqdm.tqdm(selected, desc=f"burn {name}"):
            hits = tree.query(shapely.box(*r["qbox"]), predicate="intersects")
            if len(hits) == 0:
                continue
            crs = r["proj"].crs.to_string()
            px = to_pixels(geoms[hits], crs)
            mask = rasterize_shapes(
                [(g, 1) for g in px],
                r["bounds"],
                fill=0,
                dtype="uint8",
                all_touched=True,
            )[0]
            a = acc[r["sample_id"]]
            a[mask == 1] = cid  # rarest layer burned last -> wins overlaps
            n_hit += 1
        del gdf, geoms, tree
        print(f"  {name}: burned into {n_hit} tiles ({time.time() - t0:.0f}s)")

    # --- Phase 4: write tiles + JSON in parallel; report actual class distribution ---
    io.check_disk()
    write_args = []
    for r in selected:
        write_args.append(dict(rec=r, arr=acc[r["sample_id"]]))
    print(f"writing {len(write_args)} tiles...")
    class_counts: Counter = Counter()
    n_written = 0
    with multiprocessing.Pool(args.workers) as p:
        for present in tqdm.tqdm(
            star_imap_unordered(p, _write_one, write_args), total=len(write_args)
        ):
            for cid in present:
                class_counts[cid] += 1
            n_written += 1

    # metadata.json
    classes_meta = [
        {"id": cid, "name": human, "layer": name, "description": desc}
        for cid, name, human, desc in LAYERS
    ]
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "USGS ASTER Hydrothermal Alteration Maps",
            "task_type": "classification",
            "source": "USGS",
            "license": "CC0",
            "provenance": {
                "url": "https://mrdata.usgs.gov/surficial-mineralogy/ofr-2013-1139/",
                "publication": "USGS Open-File Report 2013-1139 (Mars, 2013); doi:10.3133/ofr20131139",
                "have_locally": False,
                "annotation_method": "automated/spectral (ASTER VNIR-SWIR band-ratio logical operators)",
                "access": "per-alteration-type polygon shapefiles (5 layers), rasterized to 10 m UTM",
                "native_resolution_m": 30,
            },
            "sensors_relevant": ["sentinel2", "landsat"],
            "classes": classes_meta,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": n_written,
            "class_counts": {
                CID_TO_HUMAN[cid]: class_counts[cid] for cid in sorted(class_counts)
            },
            "notes": (
                "Foreground-only hydrothermal-alteration map over the central/southern Basin "
                "and Range. Source is five per-type polygon shapefiles (argillic, phyllic, "
                "epi_chlor=propylitic, carbonate, hydro_silica); alteration type is a property "
                "of the whole layer. Unaltered/unmapped ground is nodata (255), not a "
                "background class (spec 5 positive-only; assembly supplies negatives). "
                "Candidate 640 m grid cells were generated from polygon centroids; a class "
                f"counts as present in a cell at >= {MIN_POLYS} centroids (homogeneity/"
                "confidence proxy). Tiles-per-class balanced selection <= 1000 tiles/class "
                "(25k cap). Selected tiles were rasterized from the actual polygons of every "
                "layer intersecting the tile (STRtree query), overlaps resolved rarest-class-"
                "wins. Native WGS84 polygons burned into local UTM at 10 m (exact polygon "
                "burn, all_touched). Static geologic label -> 1-year window anchored on "
                f"{YEAR}; change_time=null. Manifest classes 'clays'/'iron oxides' have no "
                "distinct source layer and are not represented (kaolinite/clays fall within "
                "the argillic layer)."
            ),
        },
    )
    print(f"done: {n_written} tiles")
    for cid in sorted(class_counts):
        print(f"  {CID_TO_HUMAN[cid]}: {class_counts[cid]}")


def _write_one(rec: dict[str, Any], arr: np.ndarray) -> list[int]:
    sample_id = rec["sample_id"]
    present = sorted(int(v) for v in np.unique(arr) if v != io.CLASS_NODATA)
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return present
    io.write_label_geotiff(
        SLUG, sample_id, arr, rec["proj"], rec["bounds"], nodata=io.CLASS_NODATA
    )
    io.write_sample_json(
        SLUG,
        sample_id,
        rec["proj"],
        rec["bounds"],
        io.year_range(YEAR),
        source_id=f"cell_{'_'.join(map(str, _decode(rec['key'])))}",
        classes_present=present,
    )
    return present


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()

"""Process the Circumpolar Arctic Vegetation Map (CAVM, raster version) into open-set
segmentation label patches.

Source: Raynolds et al. (2019), "A raster version of the Circumpolar Arctic Vegetation
Map (CAVM)", Remote Sensing of Environment; distributed on Mendeley Data
(doi:10.17632/c4xj5rv6kv.2) as a single file "Raster CAVM GIS data.zip" containing
`raster_cavm_v1.tif` (+ legend CSV). The raster is a single-band int8 grid at **1000 m**
native resolution in a north-polar Sphere Lambert Azimuthal Equal Area projection
(latitude_of_center=90). Each pixel encodes one Arctic tundra vegetation unit (expert
photointerpretation grouping >400 plant communities into 16 vegetation types) plus
glacier/water/non-arctic codes:

    1=B1  2=B2a 3=B3  4=B4  5=B2b        (barren / mountain complexes)
    21=G1 22=G2 23=G3 24=G4              (graminoid tundras)
    31=P1 32=P2                          (prostrate dwarf-shrub tundras)
    33=S1 34=S2                          (erect/low-shrub tundras)
    41=W1 42=W2 43=W3                    (wetland complexes)
    91=FW 92=SW                          (fresh water / sea water)   -> merged "water"
    93=GL                                (glacier / permanent ice)
    99=NA                                (non-arctic; outside tundra) -> nodata
    127                                  (raster nodata)              -> nodata

We treat the vegetation unit as a per-pixel **classification** label. This is a GLOBAL
derived-product map, so per the spec we do BOUNDED-TILE sampling: the single (~118 MB
uncompressed) 1 km circumpolar file is downloaded once, then up to 1000 grid cells per
class are sampled circumpolar and class-balanced. Around each selected cell a 64x64 label
tile in a local UTM/UPS projection at **10 m** is cut and reprojected from the 1 km source
with **nearest** resampling (categorical labels). Because a 64x64 @10 m tile (640 m) is
smaller than one native 1 km cell, each tile is essentially the homogeneous vegetation
class at that location -- this heavy 1 km -> 10 m upsampling is intentional and documented
(the CAVM class is defined on the 1 km grid). High-latitude cells (>84N) fall on the UPS
polar projection, handled automatically by get_utm_ups_projection.

Time range: the CAVM vegetation label is quasi-static (expert map, raster v1 built 2018,
manifest range 2016-2019). Per spec we assign a representative 1-year Sentinel-era window
anchored on 2018.

Run:  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.circumpolar_arctic_vegetation_map_cavm
"""

import argparse
import multiprocessing
import zipfile
from collections import Counter
from typing import Any

import numpy as np
import rasterio
import tqdm
from rasterio.warp import Resampling, reproject, transform, transform_bounds
from rasterio.windows import from_bounds
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import get_transform_from_projection_and_bounds

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest

SLUG = "circumpolar_arctic_vegetation_map_cavm"

YEAR = 2018  # representative Sentinel-era year (manifest range 2016-2019; raster v1 = 2018)
PER_CLASS = 1000
TILE = 64
SRC_NODATA = 127  # raster nodata sentinel

MENDELEY_URL = (
    "https://data.mendeley.com/public-files/datasets/c4xj5rv6kv/files/"
    "5223c414-234a-498c-ae08-3100cb38510f/file_downloaded"
)
ZIP_NAME = "Raster_CAVM_GIS_data.zip"
TIF_NAME = "raster_cavm_v1.tif"

# (name, description, [source codes]) in manifest/legend order. id = position in list.
# 16 vegetation types, then glacier (GL), then water (FW+SW merged). NA(99)/127 -> nodata.
CLASSES: list[tuple[str, str, list[int]]] = [
    (
        "Cryptogam, herb barren",
        "CAVM unit B1. Dry to wet barren landscapes with very sparse, very low-growing plant "
        "cover; scattered herbs, lichens, mosses and liverworts. Zonal type in dry, "
        "continental portions of Arctic Bioclimate Subzones A and B.",
        [1],
    ),
    (
        "Cryptogam, barren complex",
        "CAVM unit B2a. Areas of exposed rock and lichens interspersed with lakes and "
        "graminoid areas. Subzones C and D on the Canadian Shield.",
        [2],
    ),
    (
        "Non-carbonate mountain complex",
        "CAVM unit B3. Sparse alpine vegetation and rocks on non-carbonate bedrock; variety "
        "and size of plants decrease with elevation and latitude.",
        [3],
    ),
    (
        "Carbonate mountain complex",
        "CAVM unit B4. Sparse alpine vegetation and rocks on carbonate bedrock; variety and "
        "size of plants decrease with elevation and latitude.",
        [4],
    ),
    (
        "Cryptogam, barren, dwarf-shrub complex",
        "CAVM unit B2b. Areas of exposed rock and lichens interspersed with lakes and shrubby "
        "areas. Subzones E and D on the Canadian Shield.",
        [5],
    ),
    (
        "Graminoid, forb, cryptogam tundra",
        "CAVM unit G1. Moist tundra with moderate to complete cover of very low-growing "
        "plants (grasses, rushes, forbs, mosses, lichens, liverworts). Zonal type in maritime "
        "portions of Subzones A and B.",
        [21],
    ),
    (
        "Graminoid, prostrate dwarf-shrub, forb, moss tundra",
        "CAVM unit G2. Moist to dry tundra, open to continuous cover. Rushes dominant in "
        "Subzone B, sedges in Subzone C, with prostrate shrubs < 5 cm tall. Zonal type in "
        "continental portions of Subzones B and C.",
        [22],
    ),
    (
        "Non-tussock sedge, dwarf-shrub, moss tundra",
        "CAVM unit G3. Moist tundra dominated by sedges and dwarf shrubs < 40 cm tall with a "
        "well-developed moss layer; frost-boil barren patches common. Zonal type on nonacidic "
        "soils in Subzones D, some C and E.",
        [23],
    ),
    (
        "Tussock-sedge, dwarf-shrub, moss tundra",
        "CAVM unit G4. Moist tundra dominated by tussock cottongrass (Eriophorum vaginatum) "
        "and dwarf shrubs < 40 cm tall; mosses abundant. Zonal type on acidic soils in "
        "Subzone E, some D.",
        [24],
    ),
    (
        "Prostrate dwarf-shrub, herb, lichen tundra",
        "CAVM unit P1. Dry tundra with patchy vegetation; prostrate shrubs < 5 cm tall (Dryas "
        "spp., Salix arctica) dominant with graminoids, forbs and lichens. Zonal type in dry "
        "continental portions of Subzones B and C and higher elevations of D and E.",
        [31],
    ),
    (
        "Prostrate/hemi-prostrate dwarf-shrub, lichen tundra",
        "CAVM unit P2. Moist to dry tundra dominated by prostrate and hemiprostrate shrubs "
        "< 15 cm tall, particularly Cassiope spp. Zonal type in maritime, acidic portions of "
        "Subzone C.",
        [32],
    ),
    (
        "Erect dwarf-shrub, moss tundra",
        "CAVM unit S1. Tundra dominated by erect dwarf-shrubs, mostly < 40 cm tall. Zonal "
        "type in continental areas with acidic soils of Subzone D.",
        [33],
    ),
    (
        "Low-shrub, moss tundra",
        "CAVM unit S2. Tundra dominated by low shrubs > 40 cm tall. Zonal type in warmer, "
        "maritime portions of Subzone E and areas with deep, moist active layers.",
        [34],
    ),
    (
        "Sedge/grass, moss wetland complex",
        "CAVM unit W1. Wetland complexes in the colder Arctic, dominated by sedges, grasses "
        "and mosses. Subzones B and C.",
        [41],
    ),
    (
        "Sedge, moss, dwarf-shrub wetland complex",
        "CAVM unit W2. Wetland complexes in milder Arctic areas, dominated by sedges and "
        "mosses, including erect dwarf-shrubs < 40 cm tall. Subzone D.",
        [42],
    ),
    (
        "Sedge, moss, low-shrub wetland complex",
        "CAVM unit W3. Wetland complexes in the warmer Arctic, dominated by sedges and shrubs "
        "> 40 cm tall. Subzone E.",
        [43],
    ),
    ("glacier", "CAVM unit GL. Glacier / permanent ice and snow.", [93]),
    (
        "water",
        "CAVM units FW + SW: fresh water (lakes and rivers) and sea water (ocean), merged "
        "into a single water class.",
        [91, 92],
    ),
]
SRC_TO_ID: dict[int, int] = {}
for _cid, (_n, _d, _codes) in enumerate(CLASSES):
    for _code in _codes:
        SRC_TO_ID[_code] = _cid


def raw_tif() -> Any:
    return io.raw_dir(SLUG) / TIF_NAME


def download_source() -> None:
    """Download the Mendeley zip and extract raster_cavm_v1.tif (idempotent, disk-guarded)."""
    io.check_disk()
    tif = raw_tif()
    if tif.exists():
        print(f"  [skip] {tif.name} already present")
        return
    zip_dst = io.raw_dir(SLUG) / ZIP_NAME
    print(f"  downloading {MENDELEY_URL}")
    download.download_http(MENDELEY_URL, zip_dst, headers={"User-Agent": "Mozilla/5.0"})
    print("  unzipping")
    with zipfile.ZipFile(zip_dst.path) as zf:
        zf.extractall(io.raw_dir(SLUG).path)


# ---- worker: source raster opened once per process ----
_SRC = None


def _init_worker() -> None:
    global _SRC
    _SRC = rasterio.open(str(raw_tif()))


def _write_one(rec: dict[str, Any]) -> None:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return
    lon, lat = rec["lon"], rec["lat"]
    dst_proj, col, row = io.lonlat_to_utm_pixel(lon, lat)
    bounds = io.centered_bounds(col, row, TILE, TILE)
    dst_transform = get_transform_from_projection_and_bounds(dst_proj, bounds)

    # Geographic extent of the UTM/UPS tile (metres) -> window in the source projection.
    xs = [bounds[0] * io.RESOLUTION, bounds[2] * io.RESOLUTION]
    ys = [bounds[1] * -io.RESOLUTION, bounds[3] * -io.RESOLUTION]
    left, right = min(xs), max(xs)
    bottom, top = min(ys), max(ys)

    ds = _SRC
    l2, b2, r2, t2 = transform_bounds(dst_proj.crs, ds.crs, left, bottom, right, top)
    pad = 2000.0  # ~2 native 1 km cells of margin so the tile is fully covered
    win = from_bounds(l2 - pad, b2 - pad, r2 + pad, t2 + pad, ds.transform)
    src = ds.read(1, window=win, boundless=True, fill_value=SRC_NODATA).astype(np.int16)
    win_transform = ds.window_transform(win)

    dst_arr = np.full((TILE, TILE), SRC_NODATA, dtype=np.int16)
    reproject(
        source=src,
        destination=dst_arr,
        src_transform=win_transform,
        src_crs=ds.crs,
        dst_transform=dst_transform,
        dst_crs=dst_proj.crs,
        resampling=Resampling.nearest,
        src_nodata=SRC_NODATA,
        dst_nodata=SRC_NODATA,
    )
    out = np.full((TILE, TILE), io.CLASS_NODATA, np.uint8)
    for v, cid in SRC_TO_ID.items():
        out[dst_arr == v] = cid

    io.write_label_geotiff(
        SLUG, sample_id, out, dst_proj, bounds, nodata=io.CLASS_NODATA
    )
    present = sorted(int(x) for x in np.unique(out) if x != io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        dst_proj,
        bounds,
        io.year_range(YEAR),
        source_id=rec["source_id"],
        classes_present=present,
    )


def _sample_candidates() -> list[dict[str, Any]]:
    """Draw up to PER_CLASS grid-cell centres per class, circumpolar, from the source raster."""
    with rasterio.open(str(raw_tif())) as ds:
        a = ds.read(1)
        st = ds.transform
        width = ds.width
        src_crs = ds.crs
    rng = np.random.default_rng(42)
    recs: list[dict[str, Any]] = []
    for cid, (name, _desc, codes) in enumerate(CLASSES):
        idx = (
            np.flatnonzero(np.isin(a, codes))
            if len(codes) > 1
            else np.flatnonzero(a == codes[0])
        )
        n_total = len(idx)
        if n_total > PER_CLASS:
            idx = rng.choice(idx, PER_CLASS, replace=False)
        rows = (idx // width).astype(np.int64)
        cols = (idx % width).astype(np.int64)
        mx = st.c + st.a * (cols + 0.5)
        my = st.f + st.e * (rows + 0.5)
        lons, lats = transform(src_crs, "EPSG:4326", mx.tolist(), my.tolist())
        for r, c, lon, lat in zip(rows.tolist(), cols.tolist(), lons, lats):
            recs.append(
                {
                    "lon": float(lon),
                    "lat": float(lat),
                    "label": cid,
                    "source_id": f"r{r}_c{c}",
                }
            )
        print(
            f"  class {cid} ({name}): {n_total} cells -> {min(n_total, PER_CLASS)} sampled"
        )
    return recs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    print("Downloading CAVM raster...")
    download_source()
    io.check_disk()

    print("Sampling class-balanced grid cells...")
    selected = _sample_candidates()
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(f"selected {len(selected)} cells (<= {PER_CLASS}/class)")

    io.check_disk()
    with multiprocessing.Pool(args.workers, initializer=_init_worker) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            pass

    counts = Counter(r["label"] for r in selected)
    class_counts = {name: counts.get(i, 0) for i, (name, _d, _c) in enumerate(CLASSES)}
    print("class counts:", class_counts)

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "Circumpolar Arctic Vegetation Map (CAVM)",
            "task_type": "classification",
            "source": "Mendeley Data / Remote Sensing of Environment (Raynolds et al. 2019)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://doi.org/10.17632/c4xj5rv6kv.2",
                "have_locally": False,
                "annotation_method": "expert photointerpretation (raster CAVM v1)",
                "product": "raster_cavm_v1.tif",
                "native_resolution_m": 1000,
                "source_crs": "Sphere Lambert Azimuthal Equal Area (lat_center=90)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc, _c) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": class_counts,
            "notes": (
                "Bounded-tile sampling of the circumpolar CAVM raster (Raynolds et al. 2019). "
                "The single 1 km int8 file (Sphere LAEA, north-polar) was downloaded from "
                "Mendeley; up to 1000 grid cells per class were sampled circumpolar and "
                "class-balanced. Around each cell a 64x64 tile in local UTM/UPS at 10 m was "
                "cut and reprojected from 1 km with NEAREST resampling (categorical). "
                "16 vegetation types kept as classes 0-15; glacier (GL)=16; water=17 merges "
                "fresh water (FW) + sea water (SW). Non-arctic (NA=99) and raster nodata "
                "(127) map to 255 (ignore). NOTE the heavy 1 km -> 10 m upsampling: a "
                "64x64 @10 m tile (640 m) is smaller than one native cell, so each tile is "
                f"essentially the homogeneous CAVM class at that location. Time range = "
                f"1-year window anchored on {YEAR} (quasi-static expert vegetation map)."
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

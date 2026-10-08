"""Process CORINE Land Cover 2018 (CLC2018) into open-set-segmentation label patches.

Source: EEA / Copernicus Land Monitoring Service, "CORINE Land Cover 2018 (raster 100 m),
Europe" -- version V2020_20u1 (product ``U2018_CLC2018_V2020_20u1``, EPSG:3035, 100 m,
25 ha minimum mapping unit). CLC is a photointerpreted (visual) pan-European land-cover /
land-use inventory with a hierarchical 3-level nomenclature whose level 3 has **44 thematic
classes** (grid codes 111..523). DOI: 10.2909/960998c1-1870-4e82-8051-6485205ebbac.

ACCESS. The authoritative full-coverage download from land.copernicus.eu is gated behind a
free EEA/Copernicus **Land Portal** login (an EU-Login account), which is *not* covered by
the Copernicus Data Space credentials in .env (those are for the
Sentinel Data Space, a different system), and the EEA discomap ArcGIS services expose only
*styled* MapServer renderings (RGB, not raw class codes). We therefore access the identical
CLC2018 100 m product through **Google Earth Engine** (asset
``COPERNICUS/CORINE/V20/100m/2018``, band ``landcover`` = the raw 3-digit CLC grid code),
authenticated with the authorized GEE service-account key referenced by .env
(``/etc/credentials/gcp_credentials.json``; spec section 8 authorizes GEE creds).

DERIVED PRODUCT -> BOUNDED-TILE, HOMOGENEOUS-WINDOW SAMPLING (spec sections 4 & 5). CLC is a
large European derived map, so we do not attempt full coverage. We fetch a curated set of
native-100 m EPSG:3035 region **blocks** (via ``ee.data.computePixels``) spread across every
European biogeographic zone and country -- chosen so all 44 classes appear, including the
geographically-restricted ones (rice, olive groves, dehesa/agro-forestry, glaciers, bare
rock, intertidal flats, salt marshes, salines, coastal lagoons, estuaries, peat bogs). Each
block is scanned on its native 100 m grid for spatially-homogeneous ~600 m windows where a
single CLC class occupies a strong majority (>= DOM_MIN of the window) -- the section 4
guidance to prefer homogeneous/high-confidence windows for derived-product maps. Candidate
windows are balanced **tiles-per-class** by their dominant class (<= PER_CLASS/class, subject
to the 25k per-dataset cap -> 25000 // 44 = 568/class) and each is reprojected from native
EPSG:3035 100 m to a local UTM projection at 10 m with **nearest** resampling (categorical
labels). The output tile keeps the *true* CLC class of every pixel (a full multi-class
segmentation patch), not just the dominant class; only genuine source nodata (outside
coverage / unclassified codes 990/995/999) becomes 255.

task_type = classification, label_type = polygons/dense_raster (accessed as the 100 m
raster). 2018 is a static per-year land-cover state, so change_time is null and the time
range is a 1-year window on 2018 (section 5).

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.corine_land_cover
Idempotent: fetched blocks are cached under raw/{slug}/blocks/, and existing
locations/{id}.tif are skipped on re-run.
"""

import argparse
import json
import multiprocessing
import random
from collections import Counter
from typing import Any

import numpy as np
import rasterio
import tqdm
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import from_bounds
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import get_transform_from_projection_and_bounds

from olmoearth_pretrain.open_set_segmentation_data import io
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "corine_land_cover"
NAME = "CORINE Land Cover"
URL = "https://land.copernicus.eu/en/products/corine-land-cover"
EE_ASSET = "COPERNICUS/CORINE/V20/100m/2018"
GEE_KEY = "/etc/credentials/gcp_credentials.json"

YEAR = 2018
PER_CLASS = 1000  # <= 1000/class; balance_by_class(total_cap) lowers to 25000//44
TILE = 64  # output UTM tile: 64 px @ 10 m = 640 m
BLOCK = 6  # native 100 m scan window: 6 px = 600 m ~ one output tile
DOM_MIN = 0.6  # a candidate window's dominant class must be >= 60% of the window
BLOCK_PX = 1500  # region-block side in native 100 m px = 150 km
CAP_PER_BLOCK_CLASS = (
    200  # cap qualifying windows kept per (block, class) to bound memory
)
PAD_M = 400.0  # native-metre pad so the reprojected UTM tile is fully covered
SEED = 42

# EPSG:3035 (LAEA Europe) native grid origin of the CORINE asset (from its projection
# transform [100,0,900000,0,-100,5500000]); block reads are snapped to this grid so the
# fetch is aligned to native pixels (no resampling on the way in).
LAEA_ORIGIN_X = 900000
LAEA_ORIGIN_Y = 5500000

# --- CLC level-3 nomenclature: grid code -> (name, description). 44 classes. Ascending
# code order defines the output class id (0..43). Descriptions condensed from the CLC2018
# nomenclature guidelines (Copernicus / EEA). ---
CLC_CLASSES: list[tuple[int, str, str]] = [
    (
        111,
        "Continuous urban fabric",
        "Land mostly (>80%) covered by buildings, roads and artificial surfaces; vegetation "
        "and bare soil are exceptional. Dense city cores.",
    ),
    (
        112,
        "Discontinuous urban fabric",
        "Land where buildings, roads and artificial surfaces are associated with vegetated "
        "areas and bare soil, occupying a discontinuous but significant part of the surface "
        "(suburban/residential).",
    ),
    (
        121,
        "Industrial or commercial units",
        "Artificially surfaced areas (concrete, asphalt, tarmacadam) without vegetation, "
        "occupied by industry, commerce, public utilities or associated infrastructure.",
    ),
    (
        122,
        "Road and rail networks and associated land",
        "Motorways, railways and associated installations (stations, platforms, embankments); "
        "minimum width ~100 m included.",
    ),
    (
        123,
        "Port areas",
        "Infrastructure of port areas, including quays, dockyards and marinas.",
    ),
    (124, "Airports", "Airport installations: runways, buildings and associated land."),
    (
        131,
        "Mineral extraction sites",
        "Open-pit extraction of construction material (sandpits, quarries) or other minerals "
        "(open-cast mines), including flooded gravel pits (except riverbed extraction).",
    ),
    (132, "Dump sites", "Public, industrial or mine dump sites and landfills."),
    (
        133,
        "Construction sites",
        "Spaces under construction development, soil or bedrock excavation, earthworks.",
    ),
    (
        141,
        "Green urban areas",
        "Areas with vegetation within or partly embraced by urban fabric: parks, cemeteries "
        "with vegetation, mansion grounds.",
    ),
    (
        142,
        "Sport and leisure facilities",
        "Camping grounds, sports grounds, leisure parks, golf courses, racecourses, etc., "
        "including formal parks not surrounded by urban zones.",
    ),
    (
        211,
        "Non-irrigated arable land",
        "Cereals, legumes, fodder crops, root crops and fallow land under rain-fed cultivation, "
        "including vegetables under the open air or plastic/glass. No permanent crops.",
    ),
    (
        212,
        "Permanently irrigated land",
        "Crops irrigated permanently or periodically using a permanent infrastructure "
        "(irrigation channels, drainage network); most of these crops could not be cultivated "
        "without water supply.",
    ),
    (
        213,
        "Rice fields",
        "Land developed for rice cultivation: flat surfaces with irrigation channels, "
        "flooded surfaces during part of the year.",
    ),
    (221, "Vineyards", "Areas planted with vines."),
    (
        222,
        "Fruit trees and berry plantations",
        "Parcels planted with fruit trees or shrubs: single or mixed fruit species, fruit "
        "trees associated with permanently grassed surfaces, including chestnut and walnut.",
    ),
    (
        223,
        "Olive groves",
        "Areas planted with olive trees, including mixed occurrence of olive trees and vines.",
    ),
    (
        231,
        "Pastures",
        "Dense, predominantly graminoid grass cover of floral composition, not under a "
        "rotation system, mainly for grazing; mechanical harvesting possible. Includes hedges.",
    ),
    (
        241,
        "Annual crops associated with permanent crops",
        "Non-permanent crops (arable land or pasture) associated with permanent crops on the "
        "same parcel.",
    ),
    (
        242,
        "Complex cultivation patterns",
        "Juxtaposition of small parcels of diverse annual crops, pasture and/or permanent "
        "crops with scattered houses/gardens (a mosaic too small to map separately).",
    ),
    (
        243,
        "Land principally occupied by agriculture, with significant natural vegetation",
        "Areas principally occupied by agriculture, interspersed with significant natural "
        "vegetation (semi-natural areas, forest, wetlands, water bodies).",
    ),
    (
        244,
        "Agro-forestry areas",
        "Annual crops or grazing under the wooded cover of forestry species (e.g. the Iberian "
        "dehesa/montado -- oak parkland grazing).",
    ),
    (
        311,
        "Broad-leaved forest",
        "Vegetation formation composed principally of trees, including shrub and bush "
        "understorey, where broad-leaved species predominate (>75% of tree cover).",
    ),
    (
        312,
        "Coniferous forest",
        "Vegetation formation composed principally of trees, including shrub and bush "
        "understorey, where coniferous species predominate (>75% of tree cover).",
    ),
    (
        313,
        "Mixed forest",
        "Vegetation formation composed principally of trees, including shrub and bush "
        "understorey, where neither broad-leaved nor coniferous species predominate.",
    ),
    (
        321,
        "Natural grasslands",
        "Low-productivity grassland, often in rough/uneven areas with rocky outcrops, frequently "
        "including areas of coarse grass, heath and scrub; not affected by human activity.",
    ),
    (
        322,
        "Moors and heathland",
        "Vegetation with low and closed cover, dominated by bushes, shrubs and herbaceous "
        "plants (heather, briars, broom, gorse, laburnum).",
    ),
    (
        323,
        "Sclerophyllous vegetation",
        "Bushy sclerophyllous (hard-leaved, drought-adapted) vegetation: maquis, matorral, "
        "garrigue -- dense or degraded Mediterranean shrubland.",
    ),
    (
        324,
        "Transitional woodland-shrub",
        "Bushy or herbaceous vegetation with scattered trees; either woodland degradation or "
        "forest regeneration/recolonisation.",
    ),
    (
        331,
        "Beaches, dunes, sands",
        "Beaches, dunes and expanses of sand or pebbles in coastal or continental locations, "
        "including river beds in dry-climate regimes.",
    ),
    (
        332,
        "Bare rocks",
        "Scree, cliffs, rock outcrops, including active erosion, rocks and reef flats "
        "situated above the high-water mark.",
    ),
    (
        333,
        "Sparsely vegetated areas",
        "Includes steppes, tundra and badlands; scattered high-altitude vegetation.",
    ),
    (
        334,
        "Burnt areas",
        "Areas affected by recent fires, still mainly black (charcoal remains).",
    ),
    (
        335,
        "Glaciers and perpetual snow",
        "Land covered by glaciers or permanent snowfields.",
    ),
    (
        411,
        "Inland marshes",
        "Low-lying land usually flooded in winter, more or less saturated by water all year "
        "round (freshwater).",
    ),
    (
        412,
        "Peat bogs",
        "Peatland consisting mainly of decomposed moss and vegetable matter, with or without "
        "extraction (exploited/unexploited bogs).",
    ),
    (
        421,
        "Salt marshes",
        "Vegetated low-lying areas above the high-tide line, susceptible to flooding by sea "
        "water, often in the process of filling in by coastal mud and sand deposits.",
    ),
    (
        422,
        "Salines",
        "Salt-pans / salt-works: active or in process of abandonment, where salt is produced "
        "by evaporation of sea water in embanked basins.",
    ),
    (
        423,
        "Intertidal flats",
        "Generally unvegetated expanses of mud, sand or rock between high- and low-water marks, "
        "exposed at low tide (tidal mudflats).",
    ),
    (
        511,
        "Water courses",
        "Natural or artificial water courses serving as drainage channels, including canals; "
        "minimum width ~100 m.",
    ),
    (
        512,
        "Water bodies",
        "Natural or artificial stretches of inland water: lakes, ponds, reservoirs.",
    ),
    (
        521,
        "Coastal lagoons",
        "Stretches of salt or brackish water separated from the sea by a land barrier, "
        "connected to the sea (lagoons).",
    ),
    (
        522,
        "Estuaries",
        "The mouth of a river within which the tide ebbs and flows: mixing of fresh and marine "
        "water in the tidal transition zone.",
    ),
    (
        523,
        "Sea and ocean",
        "Zones seaward of the lowest tide limit (open marine waters).",
    ),
]

CODE_TO_ID = {code: i for i, (code, _n, _d) in enumerate(CLC_CLASSES)}
VALID_CODES = np.array(sorted(CODE_TO_ID), dtype=np.int32)

# Curated European region blocks (name -> (lon, lat) of block centre). Each fetches a
# BLOCK_PX x BLOCK_PX native 100 m block (~150 km) snapped to the LAEA grid. Chosen to span
# every biogeographic region (Boreal, Alpine, Atlantic, Continental, Pannonian, Steppic,
# Mediterranean, Macaronesian, Black Sea) and to include the geographically-restricted CLC
# classes; a single block typically supplies many classes.
REGIONS: dict[str, tuple[float, float]] = {
    # --- Iberia: Mediterranean crops, dehesa/agro-forestry, olive, rice, sclerophyllous ---
    "andalusia_es": (-4.8, 37.4),
    "extremadura_dehesa_es": (-6.2, 39.2),
    "guadalquivir_donana_es": (-6.3, 37.0),
    "ebro_valley_es": (-0.6, 41.7),
    "valencia_rice_es": (-0.3, 39.3),
    "meseta_castilla_es": (-4.5, 41.3),
    "galicia_es": (-8.2, 42.8),
    "portugal_montado": (-8.2, 38.6),
    "lisbon_tagus_estuary_pt": (-9.2, 38.7),
    "pyrenees_es_fr": (0.6, 42.6),
    # --- France ---
    "paris_basin_fr": (2.4, 48.8),
    "camargue_rhone_fr": (4.6, 43.5),
    "aquitaine_landes_fr": (-0.9, 44.3),
    "gironde_estuary_fr": (-1.0, 45.3),
    "brittany_fr": (-3.0, 48.2),
    "alsace_rhine_fr": (7.6, 48.3),
    "provence_fr": (6.0, 43.9),
    # --- Alps / mountain (glaciers, bare rock, natural grassland, sparse veg) ---
    "mont_blanc_alps": (6.9, 45.85),
    "eastern_alps_at": (11.4, 47.1),
    "swiss_alps": (8.0, 46.5),
    "dolomites_it": (11.8, 46.5),
    # --- Italy ---
    "po_valley_rice_it": (8.8, 45.3),
    "venice_lagoon_it": (12.3, 45.4),
    "tuscany_it": (11.2, 43.4),
    "puglia_olive_it": (16.6, 40.9),
    "sicily_it": (14.3, 37.5),
    "sardinia_it": (9.0, 40.1),
    # --- Central Europe ---
    "germany_ruhr": (7.2, 51.4),
    "bavaria_de": (11.5, 48.7),
    "saxony_lignite_de": (12.4, 51.2),
    "bohemia_cz": (14.4, 49.9),
    "poland_central": (19.4, 52.0),
    "hungary_pannonian": (19.6, 47.1),
    "romania_danube_delta": (29.0, 45.0),
    "carpathians_ro": (25.3, 45.6),
    # --- British Isles (peat bogs, moors, pastures, estuaries) ---
    "ireland_bogs": (-8.0, 53.3),
    "scotland_highlands": (-4.5, 57.0),
    "wales_severn": (-3.3, 51.7),
    "england_east_anglia": (0.6, 52.4),
    # --- Low Countries / North Sea (intertidal flats, salt marshes, lagoons) ---
    "wadden_sea_nl_de": (8.3, 53.6),
    "netherlands_randstad": (4.8, 52.1),
    "denmark_jutland": (9.3, 56.2),
    # --- Scandinavia / Baltic (coniferous, peat bogs, water, glaciers, bare rock) ---
    "south_sweden": (14.5, 57.5),
    "north_sweden_lapland": (18.5, 66.5),
    "finland_lakes": (26.0, 62.0),
    "norway_fjords": (7.5, 61.2),
    "norway_jotunheimen": (8.3, 61.6),
    "estonia_bogs": (25.5, 58.6),
    "lithuania_curonian": (21.2, 55.4),
    # --- SE Europe / Balkans / Aegean (olive, sclerophyllous, sea) ---
    "greece_thessaloniki_rice": (22.8, 40.7),
    "greece_peloponnese": (22.2, 37.4),
    "croatia_dalmatia": (16.4, 43.5),
    "bulgaria_thrace": (25.3, 42.1),
    # --- Iceland (glaciers, bare rock, sparse veg, lava) ---
    "iceland_vatnajokull": (-17.5, 64.4),
    # --- Turkey (EEA39: Mediterranean + Anatolian steppe) ---
    "turkey_aegean": (27.5, 38.5),
    "turkey_central_anatolia": (33.0, 39.0),
}


def blocks_dir():
    return io.raw_dir(SLUG) / "blocks"


def block_path(name: str):
    return blocks_dir() / f"{name}.tif"


def block_origin(lon: float, lat: float) -> tuple[int, int]:
    """Return the LAEA-grid-snapped (tx0, ty0) top-left for a BLOCK_PX block centred on
    (lon, lat). Snapping to the native 100 m grid keeps the EE read aligned (no resampling).
    """
    from pyproj import Transformer

    tf = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    x, y = tf.transform(lon, lat)
    half = BLOCK_PX * 100 / 2.0
    tx0 = round((x - half - LAEA_ORIGIN_X) / 100.0) * 100 + LAEA_ORIGIN_X
    ty0 = round((y + half - LAEA_ORIGIN_Y) / 100.0) * 100 + LAEA_ORIGIN_Y
    return int(tx0), int(ty0)


# --- Earth Engine (per-process lazy init) ---------------------------------------------
_EE_READY = False


def _ensure_ee() -> None:
    global _EE_READY
    if _EE_READY:
        return
    import ee

    info = json.load(open(GEE_KEY))
    ee.Initialize(ee.ServiceAccountCredentials(info["client_email"], GEE_KEY))
    _EE_READY = True


def fetch_block(name: str) -> None:
    """Fetch one native-100 m EPSG:3035 CLC block via computePixels; cache as GeoTIFF."""
    dst = block_path(name)
    if dst.exists():
        return
    import ee

    _ensure_ee()
    lon, lat = REGIONS[name]
    tx0, ty0 = block_origin(lon, lat)
    img = ee.Image(EE_ASSET).select("landcover")
    req = {
        "expression": img,
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {
            "dimensions": {"width": BLOCK_PX, "height": BLOCK_PX},
            "affineTransform": {
                "scaleX": 100,
                "shearX": 0,
                "translateX": tx0,
                "shearY": 0,
                "scaleY": -100,
                "translateY": ty0,
            },
            "crsCode": "EPSG:3035",
        },
    }
    last = None
    for attempt in range(4):
        try:
            arr = ee.data.computePixels(req)
            break
        except Exception as e:  # noqa: BLE001 - retry transient EE errors
            last = e
            import time

            time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"computePixels failed for block {name}: {last}")

    a = arr[arr.dtype.names[0]] if arr.dtype.names else arr
    a = np.asarray(a)
    if a.ndim == 3:
        a = a[..., 0]
    a = a.astype(np.uint16)
    transform = Affine(100, 0, tx0, 0, -100, ty0)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / (dst.name + ".tmp")
    with rasterio.open(
        tmp.path,
        "w",
        driver="GTiff",
        height=BLOCK_PX,
        width=BLOCK_PX,
        count=1,
        dtype="uint16",
        crs="EPSG:3035",
        transform=transform,
        compress="deflate",
    ) as ds:
        ds.write(a, 1)
    tmp.rename(dst)


def _scan_block(name: str) -> list[dict[str, Any]]:
    """Find homogeneous single-dominant-class BLOCKxBLOCK windows in one region block."""
    with rasterio.open(block_path(name).path) as ds:
        arr = ds.read(1)
        st = ds.transform
    h, w = arr.shape
    nby, nbx = h // BLOCK, w // BLOCK
    if nby == 0 or nbx == 0:
        return []
    a = arr[: nby * BLOCK, : nbx * BLOCK].reshape(nby, BLOCK, nbx, BLOCK)
    denom = float(BLOCK * BLOCK)
    best = np.zeros((nby, nbx), np.float32)
    best_code = np.zeros((nby, nbx), np.int32)
    for code in VALID_CODES.tolist():
        cnt = (a == code).sum(axis=(1, 3)).astype(np.float32)
        m = cnt > best
        best[m] = cnt[m]
        best_code[m] = code
    dom_frac = best / denom
    qual = (dom_frac >= DOM_MIN) & (best_code > 0)
    brs, bcs = np.nonzero(qual)
    cx = bcs * BLOCK + BLOCK / 2.0
    cy = brs * BLOCK + BLOCK / 2.0
    xs = st.c + cx * st.a  # EPSG:3035 easting of window centre
    ys = st.f + cy * st.e  # EPSG:3035 northing of window centre

    # Convert block-centre 3035 coords -> lon/lat (batched) for UTM assignment later.
    from pyproj import Transformer

    tf = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    lons, lats = tf.transform(xs, ys)

    per_class: dict[int, list[dict[str, Any]]] = {}
    rng = random.Random(hash(name) & 0xFFFFFFFF)
    idx = list(range(len(brs)))
    rng.shuffle(idx)
    for i in idx:
        code = int(best_code[brs[i], bcs[i]])
        cid = CODE_TO_ID[code]
        bucket = per_class.setdefault(cid, [])
        if len(bucket) >= CAP_PER_BLOCK_CLASS:
            continue
        bucket.append(
            {
                "block": name,
                "lon": float(lons[i]),
                "lat": float(lats[i]),
                "label": cid,
                "dom_frac": float(dom_frac[brs[i], bcs[i]]),
                "source_id": f"{name}_r{int(brs[i])}_c{int(bcs[i])}",
            }
        )
    out: list[dict[str, Any]] = []
    for bucket in per_class.values():
        out.extend(bucket)
    return out


def _write_one(rec: dict[str, Any]) -> None:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return
    lon, lat = rec["lon"], rec["lat"]
    dst_proj, col, row = io.lonlat_to_utm_pixel(lon, lat)
    bounds = io.centered_bounds(col, row, TILE, TILE)
    dst_transform = get_transform_from_projection_and_bounds(dst_proj, bounds)

    # Geographic (3035) bbox of the UTM tile so we can window the source block read.
    xs = [bounds[0] * io.RESOLUTION, bounds[2] * io.RESOLUTION]
    ys = [bounds[1] * -io.RESOLUTION, bounds[3] * -io.RESOLUTION]
    left, right = min(xs), max(xs)
    bottom, top = min(ys), max(ys)
    l2, b2, r2, t2 = transform_bounds(
        dst_proj.crs, "EPSG:3035", left, bottom, right, top
    )

    with rasterio.open(block_path(rec["block"]).path) as ds:
        win = from_bounds(l2 - PAD_M, b2 - PAD_M, r2 + PAD_M, t2 + PAD_M, ds.transform)
        src = ds.read(1, window=win, boundless=True, fill_value=0)
        win_transform = ds.window_transform(win)
        src_crs = ds.crs

    dst_codes = np.zeros((TILE, TILE), np.uint16)
    reproject(
        source=src,
        destination=dst_codes,
        src_transform=win_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_proj.crs,
        resampling=Resampling.nearest,
        src_nodata=0,
        dst_nodata=0,
    )
    out = np.full((TILE, TILE), io.CLASS_NODATA, np.uint8)
    for code, cid in CODE_TO_ID.items():
        out[dst_codes == code] = cid

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
        change_time=None,
        source_id=rec["source_id"],
        classes_present=present,
    )


def _write_source_txt(used: list[str]) -> None:
    d = io.raw_dir(SLUG)
    d.mkdir(parents=True, exist_ok=True)
    (d / "SOURCE.txt").write_text(
        "CORINE Land Cover 2018 (raster 100 m), version V2020_20u1 "
        "(U2018_CLC2018_V2020_20u1).\n"
        "EEA / Copernicus Land Monitoring Service. EPSG:3035, 100 m, 25 ha MMU, 44 classes.\n"
        "DOI: 10.2909/960998c1-1870-4e82-8051-6485205ebbac  License: Copernicus open.\n"
        "Full-coverage download is gated behind a free EEA Land Portal (EU-Login) account\n"
        "not covered by the Copernicus Data Space creds in .env, and EEA discomap exposes\n"
        "only styled RGB MapServer renderings. The identical product is read from Google\n"
        "Earth Engine asset COPERNICUS/CORINE/V20/100m/2018 (band landcover = raw CLC grid\n"
        "code) via the authorized GEE service account (gcp_credentials.json).\n"
        f"{len(used)} native-100 m EPSG:3035 blocks ({BLOCK_PX}x{BLOCK_PX} px = 150 km) over\n"
        "curated European regions were fetched via ee.data.computePixels and cached under\n"
        "blocks/ for bounded-tile homogeneous-window sampling; the full mosaic is not pulled.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    names = sorted(REGIONS)
    _write_source_txt(names)

    print(f"Fetching {len(names)} native-100 m CLC blocks from Earth Engine...")
    # Modest fan-out for EE fetch (avoid hammering computePixels quotas).
    with multiprocessing.Pool(min(len(names), 12)) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, fetch_block, [dict(name=n) for n in names]),
            total=len(names),
        ):
            pass
    io.check_disk()

    print("Scanning blocks for homogeneous single-class windows...")
    all_recs: list[dict[str, Any]] = []
    with multiprocessing.Pool(min(len(names), 16)) as p:
        for recs in tqdm.tqdm(
            star_imap_unordered(p, _scan_block, [dict(name=n) for n in names]),
            total=len(names),
        ):
            all_recs.extend(recs)
    print(f"scanned {len(all_recs)} candidate homogeneous windows")
    cand_counts = Counter(r["label"] for r in all_recs)
    print("candidate class counts (by id):", dict(sorted(cand_counts.items())))

    selected = balance_by_class(all_recs, "label", per_class=PER_CLASS)
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(f"selected {len(selected)} windows (tiles-per-class balanced, 25k cap)")

    io.check_disk()
    with multiprocessing.Pool(args.workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            pass

    counts = Counter(r["label"] for r in selected)
    class_counts = {
        f"{code}:{name}": counts.get(i, 0)
        for i, (code, name, _d) in enumerate(CLC_CLASSES)
    }
    missing = [
        name for i, (_c, name, _d) in enumerate(CLC_CLASSES) if counts.get(i, 0) == 0
    ]
    print("class counts:", class_counts)
    if missing:
        print("classes with 0 samples:", missing)

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "EEA / Copernicus",
            "license": "Copernicus open (free access, full reuse with attribution)",
            "provenance": {
                "url": URL,
                "have_locally": False,
                "annotation_method": (
                    "photointerpretation (visual) of Sentinel-2 / Sentinel-1 imagery; "
                    "CORINE Land Cover 2018 V2020_20u1, 100 m raster, 25 ha MMU"
                ),
                "accessed_via": (
                    "Google Earth Engine asset COPERNICUS/CORINE/V20/100m/2018 "
                    "(band landcover = raw CLC grid code), authorized GEE service account"
                ),
                "doi": "10.2909/960998c1-1870-4e82-8051-6485205ebbac",
                "product_version": "V2020_20u1 (U2018_CLC2018_V2020_20u1)",
                "native_crs": "EPSG:3035",
                "native_resolution_m": 100,
                "year": YEAR,
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "clc_code": code, "description": desc}
                for i, (code, name, desc) in enumerate(CLC_CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": class_counts,
            "notes": (
                "Bounded-tile, homogeneous-window sampling of the CORINE Land Cover 2018 "
                "100 m derived product (spec sections 4 & 5). "
                f"{len(names)} native-100 m EPSG:3035 blocks ({BLOCK_PX}x{BLOCK_PX} px ~= "
                "150 km) over curated European biogeographic regions are fetched from GEE and "
                f"scanned on the native 100 m grid for {BLOCK}x{BLOCK} (~600 m) windows where a "
                f"single CLC class occupies >= {int(DOM_MIN * 100)}% of the window. Windows are "
                "balanced tiles-per-class by their dominant class (<= 1000/class, lowered to "
                "25000//44 = 568 by the 25k cap) and reprojected from EPSG:3035 100 m to local "
                "UTM at 10 m with NEAREST resampling. Output tiles keep the TRUE CLC class of "
                "every pixel (full multi-class segmentation), not only the dominant class; only "
                "genuine source nodata / unclassified codes become 255. Static 2018 label: "
                "change_time=null, 1-year time range on 2018. Native MMU is 25 ha (500 m), so a "
                "640 m tile carries only a handful of native pixels per side -- a deliberately "
                "coarse land-cover probe. Small artificial classes (port areas, airports, dump/"
                "construction sites, road/rail) and geographically-restricted classes rarely "
                "form homogeneous 600 m windows and are naturally sparse; per spec section 5 all "
                "classes are kept even where sparse (downstream assembly drops the too-small)."
            ),
        },
    )
    print(f"num_samples={len(selected)} task_type=classification")
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()

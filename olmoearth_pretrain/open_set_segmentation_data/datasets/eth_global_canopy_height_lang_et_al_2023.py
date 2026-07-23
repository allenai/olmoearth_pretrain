"""ETH Global Canopy Height (Lang et al. 2023) -> open-set-segmentation regression patches.

Source: ETH Zurich EcoVision Lab "A high-resolution canopy height model of the Earth"
(Lang, Jetz, Schindler & Wegner, Nature Ecology & Evolution 2023). A global, wall-to-wall
canopy *top* height map for the year 2020 at 10 m ground sampling distance, produced by a
probabilistic deep-learning ensemble that fuses NASA GEDI spaceborne lidar (RH98 canopy
height reference) with Sentinel-2 optical imagery, with a companion per-pixel predictive
standard-deviation (uncertainty) layer. Project page:
https://langnico.github.io/globalcanopyheight/ ; data DOI 10.3929/ethz-b-000609802;
CC-BY-4.0 (free of charge, no use restriction).

Distribution: the product is released as 3 deg x 3 deg tiles (ESA-WorldCover grid) as
Cloud-Optimized GeoTIFFs on ETH's public libdrive (no credentials). Each canopy-height
("_Map") tile is EPSG:4326, ~10 m (1/12000 deg), single-band **uint8** giving canopy top
height in **metres**, with **255 = no-data** (ocean, permanent snow/ice, and pixels the
model masks out). The tile browser (langnico.github.io/globalcanopyheight/assets/
tile_index.html) enumerates 2651 land tiles; each tile's download URL follows a fixed
template on the libdrive share.

This is a *regression* dataset (continuous per-pixel canopy top height in metres) and the
map is already 10 m, so it is a `dense_raster`. As a **global derived product** we do
BOUNDED-tile sampling (spec 5): we download a curated, cross-biome set of 35 tiles spanning
tropical/temperate/boreal forest, savanna/woodland, Mediterranean/shrubland, grassland/
steppe and tundra (see REGIONS), and draw up to 5000 64x64 windows from them. The raw
distribution of canopy height is heavily zero-inflated (deserts, grassland, water edges),
and tall canopy (>30 m) is globally rare (~5% of land), so we **bucket-balance across fixed
height buckets** to get an even spread from 0 m to the tallest canopies instead of a mostly
low/zero-height corpus. We deliberately do NOT filter by the uncertainty (SD) layer: model
uncertainty is strongly correlated with canopy height, so an SD threshold would
preferentially discard the tall-canopy windows we most want to represent.

Output: single-band **float32** GeoTIFFs, local UTM, 10 m/pixel, 64x64 (~640 m), nodata
**-99999** (io.REGRESSION_NODATA). The uint8 source is converted to float32 metres and the
source no-data value 255 is mapped to -99999 (keeping the standard regression sentinel
rather than the uint8-only 255). Each source window (EPSG:4326, ~10 m) is reprojected /
resampled to the local UTM tile at 10 m (bilinear over the continuous height field; a
nearest-resampled validity mask is warped alongside so no-data never blends into valid
pixels). The map is a 2020 annual product -> each tile gets a 1-year time window on 2020.
"""

import argparse
import multiprocessing
import random
import urllib.parse
from collections import Counter
from typing import Any

import numpy as np
import rasterio
import tqdm
from affine import Affine
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import download, io

SLUG = "eth_global_canopy_height_lang_et_al_2023"
NAME = "ETH Global Canopy Height (Lang et al. 2023)"
URL = "https://langnico.github.io/globalcanopyheight/"
DOI = "https://doi.org/10.3929/ethz-b-000609802"

# libdrive public share download template (canopy-height "_Map" tiles).
HREF_TMPL = (
    "https://libdrive.ethz.ch/index.php/s/cO8or7iOe5dT2Rt/download"
    "?path=%2F3deg_cogs&files=ETH_GlobalCanopyHeight_10m_2020_{tile}_Map.tif"
)

# Curated bounded, cross-biome set of 3x3 deg tiles (resolved from the official tile index
# by biome seed points). A global derived product -> we sample a representative set of tiles
# rather than the whole planet (spec 5). tile name -> region/biome description.
REGIONS = {
    "N00E021": "Congo Basin (DRC) tropical rainforest",
    "N00E114": "Borneo (Indonesia) tropical rainforest",
    "N06E018": "Central African savanna/woodland",
    "N12E075": "Western Ghats (India) moist forest",
    "N15E009": "Sahel (Niger) semi-arid grassland",
    "N18E096": "Myanmar mixed forest",
    "N18W102": "Central Mexico highland",
    "N21E078": "India dry deciduous/agriculture",
    "N24E108": "Southeast China subtropical forest",
    "N30W087": "Southeast US pine forest",
    "N36E138": "Japan temperate forest",
    "N36W084": "US Appalachian temperate broadleaf forest",
    "N36W120": "California Mediterranean shrubland/forest",
    "N39W006": "Iberia (Spain) Mediterranean woodland",
    "N39W102": "US Great Plains grassland",
    "N45E066": "Central Asian steppe",
    "N45W123": "US Pacific Northwest temperate conifer forest",
    "N48E009": "Central Europe (Germany) temperate forest",
    "N54W102": "Canada boreal forest",
    "N60E099": "Siberia boreal forest",
    "N63E024": "Fennoscandia (Finland) boreal forest",
    "N63W150": "Alaska boreal forest",
    "N66E090": "Northern Siberia tundra",
    "N66W111": "Northern Canada tundra",
    "S03E009": "Congo Basin (Gabon) tropical rainforest",
    "S03E102": "Sumatra (Indonesia) tropical rainforest",
    "S06E033": "East African (Tanzania) savanna/woodland",
    "S06E144": "Papua New Guinea tropical rainforest",
    "S06W063": "Amazon (Brazil) tropical rainforest",
    "S09W075": "Amazon (Peru) tropical rainforest",
    "S12W048": "Brazilian Cerrado savanna",
    "S15E132": "Northern Australia savanna",
    "S21E048": "Madagascar dry/spiny forest",
    "S36E147": "Southeast Australia temperate forest",
    "S42W072": "Chile Valdivian temperate rainforest",
}
TILES = sorted(REGIONS)

YEAR = 2020
TILE = 64  # output tile size (px); 10 m => ~640 m ground.
TOTAL = 5000  # regression per-dataset target (<= 25k cap).
SRC_NODATA = 255  # uint8 no-data (ocean / masked / snow-ice).
SRC_RES_DEG = 1.0 / 12000  # native pixel size in degrees (~10 m at equator).

# Candidate scan: exact 64x64 native-pixel blocks, valid-fraction gate, reservoir per chunk.
BLOCK = 64
MIN_VALID_FRAC = 0.60  # >=60% of the block must be observed (non-255) land.
EDGE_MARGIN_PX = 260  # keep block centers this far from tile edges so the reprojection
# window (HALF px) never runs off the downloaded tile.
CHUNK_ROWS = 3200  # native rows per parallel scan chunk (multiple of BLOCK).
CAP_PER_CHUNK = 400  # reservoir cap per scan chunk (bounds memory).
HALF = 220  # native-px half-window read around a center for reprojection
# (covers the 640 m UTM tile even at ~lat 70).

# Fixed height buckets (metres). The distribution is zero-inflated and tall canopy is
# globally rare, so balancing across these gives an even spread of heights (many
# low/zero-height windows AND scarce tall-forest windows). Right edge 300 catches any high
# value; realistic canopy tops are <~65 m.
BUCKET_EDGES = [0, 1, 3, 5, 10, 15, 20, 25, 30, 40, 300]
SEED = 42


def _tile_path(tile: str):
    return io.raw_dir(SLUG) / f"ETH_GlobalCanopyHeight_10m_2020_{tile}_Map.tif"


def _download_one(tile: str) -> str:
    """Download one canopy-height tile to raw/ (atomic, idempotent)."""
    dst = _tile_path(tile)
    url = HREF_TMPL.format(tile=urllib.parse.quote(tile))
    download.download_http(url, dst, headers={"User-Agent": "Mozilla/5.0"}, timeout=600)
    return dst.name


def download_tiles(workers: int = 12) -> None:
    """Download (idempotently) the curated set of canopy-height tiles to raw/."""
    io.raw_dir(SLUG).mkdir(parents=True, exist_ok=True)
    todo = [t for t in TILES if not _tile_path(t).exists()]
    if not todo:
        print(f"all {len(TILES)} tiles present")
        return
    print(f"downloading {len(todo)} / {len(TILES)} tiles")
    with multiprocessing.Pool(min(workers, len(todo))) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _download_one, [dict(tile=t) for t in todo]),
            total=len(todo),
            desc="download",
        ):
            pass
    io.check_disk()


def scan_chunk(tile: str, row0: int, nrows: int) -> list[dict[str, Any]]:
    """Scan a native row range of one tile in 64x64 blocks; return candidate windows.

    Each kept candidate is a block with >= MIN_VALID_FRAC observed pixels, recording its
    center (native col/row + lon/lat) and the mean canopy height over its valid pixels
    (used for height bucket-balancing). Reservoir-sampled to CAP_PER_CHUNK for memory.
    """
    path = _tile_path(tile).path
    rng = random.Random(f"{tile}:{row0}")
    kept: list[dict[str, Any]] = []
    n_seen = 0
    with rasterio.open(path) as ds:
        W, H = ds.width, ds.height
        nbx = W // BLOCK
        if nbx == 0:
            return []
        r_end = min(H, row0 + nrows)
        for r0 in range(row0, r_end - BLOCK + 1, BLOCK):
            row_c = r0 + BLOCK // 2
            if row_c < EDGE_MARGIN_PX or row_c > H - EDGE_MARGIN_PX:
                continue
            win = Window(0, r0, nbx * BLOCK, BLOCK)
            arr = ds.read(1, window=win)  # (BLOCK, nbx*BLOCK) uint8
            # (BLOCK, nbx, BLOCK) -> (nbx, BLOCK*BLOCK)
            blk = arr.reshape(BLOCK, nbx, BLOCK).transpose(1, 0, 2).reshape(nbx, -1)
            valid = blk != SRC_NODATA
            vcount = valid.sum(axis=1)
            npix = BLOCK * BLOCK
            sumh = np.where(valid, blk, 0).astype(np.int64).sum(axis=1)
            for j in range(nbx):
                col_c = j * BLOCK + BLOCK // 2
                if col_c < EDGE_MARGIN_PX or col_c > W - EDGE_MARGIN_PX:
                    continue
                vc = int(vcount[j])
                if vc < MIN_VALID_FRAC * npix:
                    continue
                mean_h = float(sumh[j]) / vc
                lon, lat = ds.xy(row_c, col_c)
                rec = {
                    "tile": tile,
                    "col": col_c,
                    "row": row_c,
                    "lon": float(lon),
                    "lat": float(lat),
                    "value": mean_h,
                    "valid_frac": vc / npix,
                }
                n_seen += 1
                if len(kept) < CAP_PER_CHUNK:
                    kept.append(rec)
                else:
                    k = rng.randint(0, n_seen - 1)
                    if k < CAP_PER_CHUNK:
                        kept[k] = rec
    return kept


def bucket_balance_fixed(
    records: list[dict[str, Any]], edges: list[int], total: int, seed: int = SEED
) -> list[dict[str, Any]]:
    """Balance across fixed [edge_i, edge_{i+1}) height buckets (zero-inflated data).

    Take up to total//n_buckets per bucket, then top up from leftovers until ``total``.
    (The shared quantile helper degenerates on zero-inflated height, like the RCMAP
    fractional-cover dataset, so we use fixed height buckets.)
    """
    n = len(edges) - 1
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(n)]
    for r in records:
        b = int(np.searchsorted(edges, r["value"], side="right")) - 1
        buckets[min(max(b, 0), n - 1)].append(r)
    rng = random.Random(seed)
    for b in buckets:
        rng.shuffle(b)
    per = max(1, total // n)
    selected: list[dict[str, Any]] = []
    leftovers: list[dict[str, Any]] = []
    for b in buckets:
        selected.extend(b[:per])
        leftovers.extend(b[per:])
    if len(selected) < total:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: total - len(selected)])
    rng.shuffle(selected)
    return selected[:total]


def _write_one(rec: dict[str, Any]) -> dict[str, Any] | None:
    sample_id = rec["sample_id"]
    tif_path = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif_path.exists():
        with rasterio.open(tif_path.path) as ds:
            ev = ds.read(1)
        good = ev[ev != io.REGRESSION_NODATA]
        if good.size == 0:
            return {"sample_id": sample_id, "n_valid": 0}
        return {
            "sample_id": sample_id,
            "n_valid": int(good.size),
            "mean": float(good.mean()),
            "min": float(good.min()),
            "max": float(good.max()),
        }

    proj, col, row = io.lonlat_to_utm_pixel(rec["lon"], rec["lat"])
    bounds = io.centered_bounds(col, row, TILE, TILE)
    dst_transform = Affine(
        proj.x_resolution,
        0,
        bounds[0] * proj.x_resolution,
        0,
        proj.y_resolution,
        bounds[1] * proj.y_resolution,
    )

    with rasterio.open(_tile_path(rec["tile"]).path) as ds:
        c0 = max(0, rec["col"] - HALF)
        r0 = max(0, rec["row"] - HALF)
        c1 = min(ds.width, rec["col"] + HALF)
        r1 = min(ds.height, rec["row"] + HALF)
        win = Window(c0, r0, c1 - c0, r1 - r0)
        src = ds.read(1, window=win)  # uint8 metres, 255=nodata
        src_transform = ds.window_transform(win)
        src_crs = ds.crs

    # Warp the continuous height field (bilinear) and a validity mask (bilinear, thresholded)
    # separately so the 255 no-data never blends into valid output pixels.
    h_src = np.where(src == SRC_NODATA, 0, src).astype(np.float32)
    m_src = (src != SRC_NODATA).astype(np.float32)
    dst_h = np.zeros((TILE, TILE), np.float32)
    dst_m = np.zeros((TILE, TILE), np.float32)
    reproject(
        h_src,
        dst_h,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=proj.crs,
        resampling=Resampling.bilinear,
    )
    reproject(
        m_src,
        dst_m,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=proj.crs,
        resampling=Resampling.bilinear,
    )
    valid = dst_m >= 0.5
    out = np.where(valid, dst_h, io.REGRESSION_NODATA).astype(np.float32)

    good = out[out != io.REGRESSION_NODATA]
    if good.size < 0.3 * TILE * TILE:
        # Window landed mostly on no-data (coast/edge) -> not a usable label; skip so the
        # sample id simply stays absent (keeps re-runs idempotent).
        return {"sample_id": sample_id, "n_valid": int(good.size)}

    io.write_label_geotiff(
        SLUG, sample_id, out, proj, bounds, nodata=io.REGRESSION_NODATA
    )
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(YEAR),
        source_id=f"{rec['tile']}:{rec['col']}_{rec['row']}",
    )
    return {
        "sample_id": sample_id,
        "n_valid": int(good.size),
        "mean": float(good.mean()),
        "min": float(good.min()),
        "max": float(good.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="cap #samples (0 = full)")
    args = parser.parse_args()

    io.check_disk()
    download_tiles()
    io.check_disk()

    # Build parallel scan tasks (row-chunk per tile).
    tasks: list[dict[str, Any]] = []
    for tile in TILES:
        with rasterio.open(_tile_path(tile).path) as ds:
            H = ds.height
        for r0 in range(0, H, CHUNK_ROWS):
            tasks.append({"tile": tile, "row0": r0, "nrows": CHUNK_ROWS})
    print(f"{len(TILES)} tiles, {len(tasks)} scan chunks")

    with multiprocessing.Pool(args.workers) as p:
        results = list(
            tqdm.tqdm(
                star_imap_unordered(p, scan_chunk, tasks),
                total=len(tasks),
                desc="scan",
            )
        )
    candidates = [r for sub in results for r in sub]
    print(f"gathered {len(candidates)} candidate windows")
    raw_bc = Counter(
        min(
            max(int(np.searchsorted(BUCKET_EDGES, r["value"], side="right")) - 1, 0),
            len(BUCKET_EDGES) - 2,
        )
        for r in candidates
    )
    print(f"candidate height-bucket counts {dict(sorted(raw_bc.items()))}")

    io.check_disk()
    selected = bucket_balance_fixed(candidates, BUCKET_EDGES, TOTAL, seed=SEED)
    if args.limit:
        selected = selected[: args.limit]
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    sel_bc = Counter(
        min(
            max(int(np.searchsorted(BUCKET_EDGES, r["value"], side="right")) - 1, 0),
            len(BUCKET_EDGES) - 2,
        )
        for r in selected
    )
    print(
        f"selected {len(selected)} windows; height-bucket counts {dict(sorted(sel_bc.items()))}"
    )

    io.locations_dir(SLUG).mkdir(parents=True, exist_ok=True)
    stats: list[dict[str, Any]] = []
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
            desc="write",
        ):
            if res is not None:
                stats.append(res)

    valid_stats = [s for s in stats if s.get("n_valid", 0) > 0]
    n_written = len(valid_stats)
    pix_min = min((s["min"] for s in valid_stats), default=0.0)
    pix_max = max((s["max"] for s in valid_stats), default=0.0)

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "regression",
            "source": "ETH Zurich (EcoVision Lab)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": URL,
                "doi": DOI,
                "have_locally": False,
                "annotation_method": (
                    "derived product: probabilistic CNN ensemble fusing GEDI lidar (RH98 "
                    "reference) + Sentinel-2, 2020"
                ),
                "access": (
                    "public ETH libdrive 3deg_cogs COGs (no credentials); curated bounded "
                    f"cross-biome set of {len(TILES)} tiles"
                ),
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "regression": {
                "name": "canopy_height",
                "description": (
                    "Top-of-canopy height (metres) for 2020 from the ETH Global Canopy "
                    "Height model (Lang et al. 2023), a probabilistic deep-learning ensemble "
                    "fusing GEDI spaceborne lidar (RH98) with Sentinel-2 optical imagery at "
                    "10 m. Source is uint8 metres with 255=no-data; converted to float32 "
                    "metres with no-data -99999. Distribution is zero-inflated and tall "
                    "canopy is globally rare, so windows were bucket-balanced across fixed "
                    "height buckets to span 0 m to the tallest canopies."
                ),
                "unit": "meters",
                "dtype": "float32",
                "value_range": [round(pix_min, 3), round(pix_max, 3)],
                "nodata_value": io.REGRESSION_NODATA,
                "source_nodata_value": SRC_NODATA,
                "buckets": BUCKET_EDGES,
            },
            "num_samples": n_written,
            "notes": (
                "Global 10 m derived product; bounded-tile dense_raster regression sampling "
                f"from {len(TILES)} curated cross-biome 3x3 deg tiles (tropical/temperate/"
                "boreal forest, savanna, Mediterranean, grassland/steppe, tundra). 64x64 "
                "windows reprojected from EPSG:4326 ~10 m to local UTM at 10 m (bilinear "
                "height + nearest/threshold validity mask so 255 no-data never blends into "
                "valid pixels). uint8 metres -> float32 metres; source no-data 255 -> "
                "-99999. Not filtered by the SD/uncertainty layer (uncertainty correlates "
                "with height; filtering would drop rare tall canopy). Annual 2020 product -> "
                "1-year time window on 2020."
            ),
        },
    )

    hist, _ = np.histogram([r["value"] for r in selected], bins=BUCKET_EDGES)
    print("selected-window mean-height histogram (m):")
    for lo, hi, c in zip(BUCKET_EDGES[:-1], BUCKET_EDGES[1:], hist):
        print(f"  [{lo:>3}, {hi:>3}) : {c}")
    print(f"per-pixel value range across tiles: [{pix_min:.3f}, {pix_max:.3f}] m")
    print(f"num_samples={n_written} task_type=regression")
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()

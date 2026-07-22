"""Process KelpWatch (Landsat/Sentinel-2 kelp canopy) into open-set-segmentation labels.

Source: SBC LTER / kelpwatch.org "Time series of quarterly NetCDF files of kelp biomass
in the canopy from Landsat 5, 7 and 8, since 1984 (ongoing)", EDI data package
``knb-lter-sbc.74`` (rev 33; CC-BY-4.0, openly downloadable, no login). The single NetCDF
(``LandsatKelpBiomass_*_withmetadata.nc``, ~2.6 GB) is a *point cloud* of 593,426 fixed
30 m Landsat pixels along the US West Coast + Baja California, each with a WGS84 lat/lon
and quarterly time series (169 quarters, 1984 Q1 - 2026 Q1) of:

  - ``area``   : surface kelp canopy area (m^2) within the 30x30 m (=900 m^2) pixel
  - ``biomass``: wet biomass (kg) of giant kelp within the pixel
  - ``passes`` : number of Landsat scenes averaged that quarter (0 / NaN area = unobserved)

Every station is a "kelp-capable" reef pixel (essentially all have kelp in *some* quarter),
so a given quarter partitions the observed stations into surface-canopy-present (area>0)
and bare-reef/water (area==0). Kelp canopy is highly seasonal (summer/autumn peak, winter
storm loss) and interannually dynamic (2014-16 heatwave collapse), so a label is valid
ONLY for its quarter -- we therefore give each tile a ~3-month time range matching its
labeled quarter (NOT a static year), with ``change_time=null`` (a seasonal state, not a
dated change event).

TASK: dense_raster **classification** (kelp presence/absence), matching the manifest
classes ["kelp canopy", "water"]:

    id 0 = water        (observed kelp-capable pixel with no surface canopy this quarter)
    id 1 = kelp canopy  (observed surface kelp canopy this quarter, area > 0)
    255  = nodata/ignore (unobserved this quarter, or non-reef pixel with no station)

We chose classification over canopy-fraction regression because presence/absence is robust
to the per-pixel area noise (esp. at low fractions), matches the manifest's two classes,
and yields interpretable dense kelp-forest masks; a regression (area/900 fraction) framing
is possible from the same file (see summary).

Because the source is a large derived product, we do BOUNDED-TILE dense_raster sampling
(spec section 5) with tiles-per-class balancing (spec section 4): snap every station to a
64 px (640 m) tile grid in its local UTM zone, and for each Sentinel-era quarter emit
candidate tiles that are either high-confidence kelp forests (>= MIN_KELP kelp pixels) or
well-observed bare-reef negatives (>= MIN_OBS observed pixels, zero kelp). Each 30 m
station is painted as a 3x3 block of 10 m pixels (nearest upsample; categorical). The
final selection is balanced across the two classes and capped at 25k.

Run (from repo root):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.kelpwatch_landsat_sentinel_2_kelp_canopy
"""

import argparse
import multiprocessing
from collections import Counter
from datetime import UTC, datetime
from typing import Any

import numpy as np
import tqdm
import xarray as xr
from pyproj import Transformer
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, sampling

SLUG = "kelpwatch_landsat_sentinel_2_kelp_canopy"
NAME = "Kelpwatch (Landsat/Sentinel-2 kelp canopy)"

DOWNLOAD_URL = (
    "https://pasta.lternet.edu/package/data/eml/knb-lter-sbc/74/33/"
    "c2bea785267fa434c40a22e2239bb337"
)
NC_NAME = "kelp_biomass_canopy_landsat.nc"

# Tile / reconstruction parameters.
TILE = 64  # output tile size in 10 m pixels (= 640 m).
BLOCK = 3  # a 30 m source pixel spans 3x3 output (10 m) pixels.
MIN_KELP = (
    15  # a "kelp" tile needs >= this many kelp (area>0) 30 m pixels (~13,500 m^2).
)
MIN_OBS = 150  # a "water" (bare-reef negative) tile needs >= this many observed pixels.
PER_CLASS = 1000
SEED = 42
FIRST_YEAR = 2016  # Sentinel era.

CLASSES = [
    (
        "water",
        "Observed kelp-capable coastal (reef) pixel with no surface kelp canopy detected "
        "in the labeled quarter (KelpWatch area == 0 m^2).",
    ),
    (
        "kelp canopy",
        "Surface canopy of giant kelp (Macrocystis pyrifera) or bull kelp (Nereocystis "
        "luetkeana) floating on the sea surface, detected from Landsat spectral unmixing "
        "in the labeled quarter (KelpWatch area > 0 m^2 within the 900 m^2 pixel).",
    ),
]


def _nc_path():
    return io.raw_dir(SLUG) / NC_NAME


def quarter_range(year: int, quarter: int) -> tuple[datetime, datetime]:
    """~3-month UTC window for a calendar quarter (Q1=Jan-Mar, ... Q4=Oct-Dec)."""
    start_month = (quarter - 1) * 3 + 1
    start = datetime(year, start_month, 1, tzinfo=UTC)
    if quarter == 4:
        end = datetime(year + 1, 1, 1, tzinfo=UTC)
    else:
        end = datetime(year, start_month + 3, 1, tzinfo=UTC)
    return start, end


def _paint_tile(cols: np.ndarray, rows: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """Reconstruct a TILE x TILE uint8 label from per-station local coords + values.

    Each station (a 30 m pixel) is painted as a BLOCK x BLOCK (3x3) square of 10 m
    output pixels centered on its pixel. Water (0) is painted first, then kelp (1) on top
    so kelp wins at any boundary overlap. Unpainted pixels stay nodata (255).
    """
    arr = np.full((TILE, TILE), io.CLASS_NODATA, dtype=np.uint8)
    h = BLOCK // 2
    for order_val in (0, 1):
        sel = vals == order_val
        for c, r in zip(cols[sel], rows[sel]):
            r0 = max(0, int(r) - h)
            r1 = min(TILE, int(r) + h + 1)
            c0 = max(0, int(c) - h)
            c1 = min(TILE, int(c) + h + 1)
            arr[r0:r1, c0:c1] = order_val
    return arr


def _write_one(payload: dict[str, Any]) -> None:
    sample_id = payload["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return
    proj = Projection(CRS.from_epsg(payload["epsg"]), io.RESOLUTION, -io.RESOLUTION)
    x_min, y_min = payload["x_min"], payload["y_min"]
    bounds = (x_min, y_min, x_min + TILE, y_min + TILE)
    arr = _paint_tile(
        np.frombuffer(payload["cols"], dtype=np.uint8),
        np.frombuffer(payload["rows"], dtype=np.uint8),
        np.frombuffer(payload["vals"], dtype=np.uint8),
    )
    present = sorted(int(v) for v in np.unique(arr) if v != io.CLASS_NODATA)
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        quarter_range(payload["year"], payload["quarter"]),
        change_time=None,
        source_id=payload["source_id"],
        classes_present=present,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()

    # SOURCE.txt describing the raw download (already fetched by the download step).
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    src_txt = raw / "SOURCE.txt"
    if not src_txt.exists():
        with src_txt.open("w") as f:
            f.write(
                f"{NAME}\nEDI knb-lter-sbc.74 (rev 33), CC-BY-4.0\n{DOWNLOAD_URL}\n"
                f"file: {NC_NAME}\n"
            )

    print("loading NetCDF ...")
    ds = xr.open_dataset(str(_nc_path()))
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    year = ds["year"].values.astype(int)
    quarter = ds["quarter"].values.astype(int)
    valid = np.isfinite(lat) & np.isfinite(lon)
    lat, lon = lat[valid], lon[valid]
    n = lat.shape[0]
    print(f"{n} valid stations, {len(year)} quarters")

    # Per-station UTM pixel coords (10 m, matches io.lonlat_to_utm_pixel) + tile indices.
    zone = np.floor((lon + 180) / 6).astype(np.int64) + 1
    ipx = np.zeros(n, dtype=np.int64)
    ipy = np.zeros(n, dtype=np.int64)
    for z in np.unique(zone):
        m = zone == z
        tr = Transformer.from_crs("EPSG:4326", f"EPSG:326{int(z)}", always_xy=True)
        E, N = tr.transform(lon[m], lat[m])
        ipx[m] = np.floor(np.asarray(E) / io.RESOLUTION).astype(np.int64)
        ipy[m] = np.floor(-np.asarray(N) / io.RESOLUTION).astype(np.int64)
    tcol = np.floor_divide(ipx, TILE)
    trow = np.floor_divide(ipy, TILE)
    lc = (ipx - tcol * TILE).astype(np.uint8)  # local col within tile (0..63)
    lr = (ipy - trow * TILE).astype(np.uint8)  # local row within tile (0..63)

    key = (zone * 4000 + tcol) * 20000 + (trow + 10000)
    uk, tid = np.unique(key, return_inverse=True)
    n_tiles = len(uk)
    print(f"{n_tiles} unique spatial tiles")

    # Decode each unique tile key back to (zone, tcol, trow) -> epsg, pixel-bounds origin.
    uk_trow = (uk % 20000) - 10000
    uk_rest = uk // 20000
    uk_tcol = uk_rest % 4000
    uk_zone = uk_rest // 4000
    tile_epsg = (32600 + uk_zone).astype(int)
    tile_xmin = (uk_tcol * TILE).astype(int)
    tile_ymin = (uk_trow * TILE).astype(int)

    # Group station indices by tile (for fast per-tile reconstruction later).
    order = np.argsort(tid, kind="stable")
    tid_sorted = tid[order]
    starts = np.searchsorted(tid_sorted, np.arange(n_tiles))
    ends = np.searchsorted(tid_sorted, np.arange(n_tiles), side="right")

    area = ds["area"].values[:, valid]  # (time, station), NaN = unobserved

    # SCAN: per Sentinel-era quarter, find candidate tiles (kelp forests + water negatives).
    print("scanning quarters ...")
    candidates: list[dict[str, Any]] = []
    rng = np.random.default_rng(SEED)
    for t in range(len(year)):
        if year[t] < FIRST_YEAR:
            continue
        a = area[t]
        obs = np.isfinite(a)
        kelp = obs & (a > 0)
        nobs = np.bincount(tid[obs], minlength=n_tiles)
        nkelp = np.bincount(tid[kelp], minlength=n_tiles)
        kelp_tiles = np.where(nkelp >= MIN_KELP)[0]
        water_tiles = np.where((nkelp == 0) & (nobs >= MIN_OBS))[0]
        for ti in kelp_tiles:
            candidates.append({"tid": int(ti), "t": int(t), "classes": [0, 1]})
        for ti in water_tiles:
            candidates.append({"tid": int(ti), "t": int(t), "classes": [0]})
    print(
        f"candidates: {len(candidates)} "
        f"(kelp-tiles={sum(1 for c in candidates if 1 in c['classes'])}, "
        f"water-tiles={sum(1 for c in candidates if c['classes'] == [0])})"
    )

    io.check_disk()

    # SELECT: tiles-per-class balanced, 25k cap (spec 4/5).
    selected = sampling.balance_tiles_by_class(
        candidates,
        classes_key="classes",
        per_class=PER_CLASS,
        seed=SEED,
        total_cap=sampling.MAX_SAMPLES_PER_DATASET,
    )
    # Deterministic id assignment.
    selected = sorted(selected, key=lambda c: (c["tid"], c["t"]))
    print(f"selected {len(selected)} tiles")

    # BUILD write payloads (small per-tile arrays; avoids sharing the 800 MB area cube).
    payloads: list[dict[str, Any]] = []
    for i, c in enumerate(selected):
        ti, t = c["tid"], c["t"]
        st = order[starts[ti] : ends[ti]]  # station indices in this tile
        a = area[t, st]
        obs = np.isfinite(a)
        st = st[obs]
        vals = (a[obs] > 0).astype(np.uint8)
        payloads.append(
            {
                "sample_id": f"{i:06d}",
                "epsg": int(tile_epsg[ti]),
                "x_min": int(tile_xmin[ti]),
                "y_min": int(tile_ymin[ti]),
                "cols": lc[st].tobytes(),
                "rows": lr[st].tobytes(),
                "vals": vals.tobytes(),
                "year": int(year[t]),
                "quarter": int(quarter[t]),
                "source_id": f"z{int(uk_zone[ti])}_c{int(uk_tcol[ti])}_r{int(uk_trow[ti])}"
                f"_{int(year[t])}Q{int(quarter[t])}",
            }
        )

    # WRITE (parallel; idempotent).
    print("writing tiles ...")
    with multiprocessing.Pool(args.workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(payload=pl) for pl in payloads]),
            total=len(payloads),
            desc="write",
        ):
            pass

    # Per-class tile counts (a tile counts toward every class present in it).
    kelp_ct = sum(1 for c in selected if 1 in c["classes"])
    water_ct = len(selected)  # every tile has water pixels (0) somewhere
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "SBC LTER / kelpwatch.org (EDI knb-lter-sbc.74)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://kelpwatch.org",
                "edi_package": "knb-lter-sbc.74.33",
                "download_url": DOWNLOAD_URL,
                "have_locally": False,
                "annotation_method": "derived-product (Landsat spectral-unmixing, validated)",
            },
            "sensors_relevant": ["sentinel2", "landsat", "sentinel1"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_tile_counts": {"water": water_ct, "kelp canopy": kelp_ct},
            "notes": (
                "Bounded-tile dense_raster classification from the KelpWatch quarterly "
                "Landsat kelp-canopy product (30 m native, reconstructed to 64x64 UTM 10 m "
                "tiles, nearest 3x3 upsample). Kelp canopy is highly seasonal, so each tile "
                "carries a ~3-month time range matching its labeled quarter (Sentinel era "
                "2016 Q1 - 2026 Q1); change_time=null (seasonal state, not a dated event). "
                "Class 1 (kelp) pixels have area>0; class 0 (water) are observed reef "
                "pixels with area==0; 255=nodata (unobserved / non-reef). Tiles-per-class "
                "balanced, <=1000/class, 25k cap."
            ),
        },
    )
    counts = Counter()
    for pl in payloads:
        counts[pl["year"]] += 1
    print("per-year tile counts:", dict(sorted(counts.items())))
    print("done; num_samples =", len(selected))


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()

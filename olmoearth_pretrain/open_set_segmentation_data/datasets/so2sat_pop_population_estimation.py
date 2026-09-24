"""Process So2Sat POP (Population Estimation) into open-set regression label patches.

Source: So2Sat POP Part1 (Doda et al., Sci Data 2022; doi:10.14459/2021mp1633792), TUM
mediaTUM, CC-BY-4.0. A benchmark for population estimation over 98 European cities. The
population reference is the EU GEOSTAT 1 km population grid (2011 census, EPSG:3035
ETRS89-LAEA Europe); each 1x1 km grid cell carries an absolute population count (persons
living in that km^2) and a log2-binned population class. The dataset also ships Sentinel-2
seasonal mosaics (2016), LCZ, land use, nightlights, DEM and OSM patches per cell, but
those are pretraining *inputs* we do not need -- we only need the population LABEL.

Task decision -- REGRESSION on population density (persons/km^2). The source provides both
the continuous population count and a discrete class bin per cell; population is naturally a
regression target and the continuous count is strictly more informative, so we regress it.
Because each cell is exactly 1 km^2, the per-cell count IS a population density in persons
per square kilometre -- a resolution-invariant intensity -- so it can be written to a tile
of any size without a count/area rescale (same unit/convention as
``worldpop_global_population_density``). We fill each output tile uniformly with the cell's
density (the product gives one value per 1 km cell).

Georeferencing (recoverable, label-only): each populated cell's ``GRD_ID`` is a GEOSTAT/
INSPIRE LAEA grid name ``1kmN{north_km}E{east_km}`` giving the lower-left corner in EPSG:
3035 (north_km/east_km are in km). Cell centre = (east_km*1000+500, north_km*1000+500);
reproject 3035 -> WGS84 to place the tile on a local UTM grid. Cells that are NOT on the
population grid (uninhabited filler patches) carry a plain numeric id and POP=0 instead of a
``1kmN...`` name -- they have no recoverable coordinates and no population signal, so they
are skipped (the assembly step supplies negatives from other datasets, spec 5).

Download strategy (label-only; NO imagery pulled): Part1 is distributed as a single ~103 GB
``So2Sat_POP_Part1.zip`` on the mediaTUM WebDAV/dataserv share (public creds m1633792 /
m1633792, also published for rsync). The per-city population CSVs (``{city}/{city}.csv``,
one per city, 98 total, a few KB-90 KB each) live inside that zip. We read the zip's central
directory over HTTP Range requests and extract ONLY those 98 CSVs -- never touching the ~96
GB of imagery/aux patches. (mediaTUM mishandles a degenerate ``bytes=0-0`` probe by
streaming the whole file; download.HttpRangeFile now streams the probe so this is safe.)

Time range: population is a persistent/slowly-varying attribute of built structure and the
dataset was assembled to pair with the 2016 Sentinel-2 mosaics, so we treat it as a static
label and anchor a 1-year window at 2016 (spec 5 static-label rule; Sentinel era). The
underlying census is 2011, which we document; EU urban population distribution is stable
across 2011->2016, and a post-2016 pairing window is usable, so this is not a pre-2016
rejection (the label is not a dated pre-Sentinel observation).

Run:  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.so2sat_pop_population_estimation
"""

import argparse
import csv
import math
import multiprocessing
import re
import zipfile
from typing import Any

import numpy as np
import tqdm
from pyproj import Transformer
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    bucket_balance_regression,
)

SLUG = "so2sat_pop_population_estimation"
NAME = "So2Sat POP (Population Estimation)"
URL = "https://mediatum.ub.tum.de/1633792"
ZIP_URL = (
    "https://dataserv.ub.tum.de/public.php/dav/files/m1633792/So2Sat_POP_Part1.zip"
)
ZIP_AUTH = ("m1633792", "m1633792")

TILE = 64  # 64x64 @ 10 m (~640 m), a centred sub-window of the 1 km cell (<=64 cap)
TOTAL = 5000  # regression sample cap (spec 5)
N_BUCKETS = 10
YEAR = 2016  # So2Sat S2 mosaics are 2016; persistent label -> 1-year window (spec 5)
EXPECTED_CITIES = 98

# GEOSTAT/INSPIRE LAEA 1 km grid id: 1kmN{north_km}E{east_km} (lower-left corner, in km).
GRID_RE = re.compile(r"1kmN(\d+)E(\d+)$")
_TF_3035_TO_WGS84 = Transformer.from_crs(3035, 4326, always_xy=True)


def _csv_dir() -> Any:
    return io.raw_dir(SLUG) / "city_csv"


def ensure_city_csvs() -> list[str]:
    """Extract the 98 per-city population CSVs from the remote zip (range reads only).

    Idempotent: skips the (37 s) central-directory read entirely once all 98 CSVs are on
    disk, and skips any individual CSV already extracted. Returns the list of local paths.
    """
    csv_dir = _csv_dir()
    csv_dir.mkdir(parents=True, exist_ok=True)
    have = sorted(p for p in csv_dir.iterdir() if p.name.endswith(".csv"))
    if len(have) >= EXPECTED_CITIES:
        return [str(p) for p in have]

    print(f"  opening remote zip central directory: {ZIP_URL}", flush=True)
    rf = download.HttpRangeFile(ZIP_URL, auth=ZIP_AUTH)
    zf = zipfile.ZipFile(rf)
    # Per-city population CSV: So2Sat_POP_Part1/{split}/{city}/{city}.csv
    city_infos = [
        info
        for info in zf.infolist()
        if (parts := info.filename.split("/"))
        and len(parts) == 4
        and parts[3] == parts[2] + ".csv"
    ]
    print(f"  {len(city_infos)} per-city population CSVs in archive", flush=True)
    for info in tqdm.tqdm(city_infos, desc="extract-csv"):
        out = csv_dir / info.filename.split("/")[-1]
        if out.exists():
            continue
        data = zf.read(info)
        tmp = csv_dir / (out.name + ".tmp")
        with tmp.open("wb") as f:
            f.write(data)
        tmp.rename(out)
    return [str(p) for p in sorted(csv_dir.iterdir()) if p.name.endswith(".csv")]


def _parse_csv(path: str) -> list[dict[str, Any]]:
    """Parse one city CSV -> records for populated, coordinate-bearing grid cells."""
    import os

    city = os.path.basename(path)[:-4]
    recs: list[dict[str, Any]] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            gid = row.get("GRD_ID", "")
            m = GRID_RE.match(gid)
            if not m:
                continue  # numeric-id filler patch: no coords, POP=0
            try:
                pop = float(row["POP"])
            except (KeyError, ValueError):
                continue
            if not math.isfinite(pop) or pop <= 0:
                continue
            north_km, east_km = int(m.group(1)), int(m.group(2))
            cx = east_km * 1000 + 500.0
            cy = north_km * 1000 + 500.0
            lon, lat = _TF_3035_TO_WGS84.transform(cx, cy)
            recs.append(
                {
                    "lon": float(lon),
                    "lat": float(lat),
                    "pop": pop,
                    "cls": int(float(row.get("Class", 0))),
                    "source_id": f"{city}/{gid}",
                }
            )
    return recs


def build_records(paths: list[str], workers: int) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []
    with multiprocessing.Pool(min(workers, max(1, len(paths)))) as p:
        for r in tqdm.tqdm(
            star_imap_unordered(p, _parse_csv, [dict(path=pp) for pp in paths]),
            total=len(paths),
            desc="parse-csv",
        ):
            recs.extend(r)
    return recs


def _write_one(rec: dict[str, Any]) -> dict[str, Any] | None:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return None
    proj, col, row = io.lonlat_to_utm_pixel(rec["lon"], rec["lat"])
    bounds = io.centered_bounds(col, row, TILE, TILE)
    arr = np.full((TILE, TILE), np.float32(rec["pop"]), dtype=np.float32)
    io.write_label_geotiff(
        SLUG, sample_id, arr, proj, bounds, nodata=io.REGRESSION_NODATA
    )
    io.write_sample_json(
        SLUG, sample_id, proj, bounds, io.year_range(YEAR), source_id=rec["source_id"]
    )
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "So2Sat POP Part1 (Doda et al. 2022), doi:10.14459/2021mp1633792, CC-BY-4.0.\n"
            f"Label-only: the 98 per-city population CSVs ({{city}}/{{city}}.csv, columns "
            "GRD_ID,Class,POP) were range-extracted from So2Sat_POP_Part1.zip "
            f"({ZIP_URL}, public creds m1633792/m1633792) into city_csv/; the ~96 GB of "
            "Sentinel-2/LCZ/LU/nightlights/DEM/OSM patches were never downloaded.\n"
            "Population reference: EU GEOSTAT 1 km grid, 2011 census (EPSG:3035 LAEA).\n"
        )

    print("Extracting per-city population CSVs...")
    paths = ensure_city_csvs()
    print(f"  {len(paths)} city CSVs on disk")
    io.check_disk()

    print("Parsing population grid cells...")
    recs = build_records(paths, args.workers)
    print(f"  {len(recs)} populated, coordinate-bearing grid cells")

    # Population is very right-skewed -> bucket-balance across log10(count) deciles.
    def log_pop(r: dict[str, Any]) -> float:
        return math.log10(r["pop"])

    selected, log_edges = bucket_balance_regression(
        recs, log_pop, total=TOTAL, n_buckets=N_BUCKETS
    )
    pop_edges = [round(10.0**e, 2) for e in log_edges]
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(
        f"  selected {len(selected)} tiles (<= {TOTAL}); pop bucket edges {pop_edges}"
    )

    io.locations_dir(SLUG).mkdir(parents=True, exist_ok=True)
    io.check_disk()
    print(f"Writing {len(selected)} uniform-density {TILE}x{TILE} tiles...")
    with multiprocessing.Pool(args.workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
            desc="write",
        ):
            pass

    vals = np.array([r["pop"] for r in selected], dtype=np.float64)
    all_vals = np.array([r["pop"] for r in recs], dtype=np.float64)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "regression",
            "source": "TU Munich / So2Sat POP (Sci Data 2022)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": URL,
                "doi": "10.14459/2021mp1633792",
                "have_locally": False,
                "annotation_method": (
                    "census-derived: EU GEOSTAT 1 km population grid (2011 census, EPSG:3035 "
                    "LAEA); per-cell absolute population count binned to a log2 class."
                ),
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "regression": {
                "name": "population_density",
                "description": (
                    "Human population per 1 km^2 grid cell from the EU GEOSTAT 2011 census "
                    "grid. Each cell is exactly 1 km^2, so the per-cell count equals a "
                    "population density in persons per square kilometre (resolution-invariant); "
                    "tiles are filled uniformly with that value. The source also ships a "
                    "discrete log2 population class (Class 0 = 0 persons; Class c>=1 = "
                    "2^(c-1) <= pop < 2^c); we regress the continuous count instead."
                ),
                "unit": "persons per square kilometre",
                "dtype": "float32",
                "value_range": [
                    round(float(vals.min()), 2),
                    round(float(vals.max()), 2),
                ],
                "nodata_value": io.REGRESSION_NODATA,
                "buckets": pop_edges,
            },
            "num_samples": len(selected),
            "notes": (
                f"Regression on population density (persons/km^2) over 98 EU cities. "
                f"{len(recs)} populated grid cells carry recoverable EPSG:3035 LAEA "
                f"coordinates (GRD_ID '1kmN..E..'); uninhabited filler patches (numeric id, "
                f"POP=0) have no coordinates and are skipped. Bucket-balanced across "
                f"log10(pop) deciles to <= {TOTAL}; {TILE}x{TILE} tiles @ 10 m in local UTM "
                f"(centred sub-window of each 1 km cell, filled with the cell density). "
                f"Time range: 1-year window at {YEAR} (S2 mosaic year; population treated as "
                f"persistent, underlying census 2011). Label-only extraction of per-city CSVs "
                f"from So2Sat_POP_Part1.zip via HTTP Range; no imagery downloaded. "
                f"All-cell pop percentiles: p50={np.percentile(all_vals, 50):.0f}, "
                f"p90={np.percentile(all_vals, 90):.0f}, p99={np.percentile(all_vals, 99):.0f}, "
                f"max={all_vals.max():.0f}."
            ),
        },
    )

    hist_edges = [1, 10, 50, 100, 500, 1000, 5000, 10000, 50000, np.inf]
    hist, _ = np.histogram(vals, bins=hist_edges)
    print("selected-tile population histogram (persons/km^2):")
    for lo, hi, c in zip(hist_edges[:-1], hist_edges[1:], hist):
        print(f"  [{lo:>8}, {hi:>8}) : {c}")
    print(f"per-cell value range: [{vals.min():.1f}, {vals.max():.1f}] persons/km^2")
    print(f"num_samples={len(selected)} task_type=regression")

    manifest.write_registry_entry(
        SLUG, "completed", task_type="regression", num_samples=len(selected)
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()

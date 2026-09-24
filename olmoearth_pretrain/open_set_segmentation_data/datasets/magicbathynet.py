"""Process MagicBathyNet shallow-water bathymetry into open-set regression patches.

Source: "MagicBathyNet: A Multimodal Remote Sensing Dataset for Bathymetry Prediction
and Pixel-based Classification in Shallow Waters" (Agrafiotis et al., IGARSS 2024;
Zenodo record 16753753, https://zenodo.org/records/16753753, CC-BY-NC-4.0). Two coastal
areas: Agia Napa (Cyprus, Mediterranean) and Puck Lagoon (Poland, Baltic). The dataset
ships co-registered 180x180 m image patches for Sentinel-2 (18x18 px @ 10 m), SPOT-6 and
aerial, plus per-patch DSM (depth) rasters and seabed-class annotations.

TASK: REGRESSION of shallow-water bathymetry (depth). We use the **Sentinel-2 depth
patches** (``{area}/depth/s2/depth_{id}.tif``) because they are already single-band
GeoTIFFs in a local UTM projection at 10 m/pixel (18x18) -- exactly our target grid, so
no resampling is needed. Depth values are the SfM-MVS + reference-LiDAR/echosounder DSM,
in metres, referenced to the sea surface: **negative = below the water surface (deeper);
small positive values at Puck Lagoon = emergent/near-shore land in the DSM.** The fill
value 0.0 (no reference / masked) is mapped to REGRESSION_NODATA (-99999).

We take the **annotated bathymetry patches** listed in ``{area}/s2_split_bathymetry.txt``
(train + test union; all splits are fair game as pretraining labels, spec 5): 35 for Agia
Napa + 2822 for Puck Lagoon = 2857 total, comfortably under the 5000-sample regression cap,
so ALL are used (no sub-sampling, no bucket balancing).

Depth is a quasi-static seabed quantity, not a dated change event, so change_time is null
and each patch gets a 1-year window anchored on its S2 acquisition year (spec 5, static /
annual labels): Agia Napa S2 = 2016-01-10 -> 2016; Puck Lagoon S2 = 2021-04-20 -> 2021.
(Agia Napa's aerial/LiDAR reference is 2015, but the co-registered S2 image -- what
pretraining pairs against -- is Jan 2016, i.e. Sentinel-era, and the seabed is static.)

Georeferencing: source patches are already UTM 10 m (Agia Napa EPSG:32636 = WGS84 UTM 36N;
Puck Lagoon EPSG:25834 = ETRS89 UTM 34N, <1 m from WGS84). We REUSE the source CRS (spec
2 allows this) and snap the origin to the integer 10 m pixel grid (<= half-pixel, <=5 m
shift), writing values as-is with no resampling.

Output: single-band float32 GeoTIFFs, local UTM, 10 m/pixel, 18x18, nodata -99999.

Reproduce:
    python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.magicbathynet
"""

import argparse
import multiprocessing
import re
from collections import Counter
from typing import Any

import numpy as np
import rasterio
import tqdm
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.download import download_http

SLUG = "magicbathynet"
NAME = "MagicBathyNet"
URL = "https://zenodo.org/records/16753753"
ZENODO_RECORD = "16753753"
ZIP_NAME = "MagicBathyNet.zip"
ZIP_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{ZIP_NAME}/content"

# Per-area S2 acquisition year (anchors the 1-year static-label window).
AREA_YEAR = {"agia_napa": 2016, "puck_lagoon": 2021}

MAX_SAMPLES = 5000  # regression cap (spec 5)


# ---------------------------------------------------------------------------
# Download + selective extraction
# ---------------------------------------------------------------------------
def raw_root():
    return io.raw_dir(SLUG)


def extracted_dir():
    return raw_root() / "extracted" / "MagicBathyNet"


def download_and_extract() -> None:
    """Download the Zenodo zip (5.9 GB) and extract only the S2 depth patches + splits."""
    import zipfile

    io.check_disk()
    raw = raw_root()
    raw.mkdir(parents=True, exist_ok=True)
    zip_path = raw / ZIP_NAME
    if not zip_path.exists():
        print(f"downloading {ZIP_URL}")
        download_http(ZIP_URL, zip_path)
    io.check_disk()

    ext = extracted_dir()
    # Idempotent: skip if a marker area folder already extracted.
    if (ext / "agia_napa" / "s2_split_bathymetry.txt").exists():
        print("already extracted")
        return
    print("extracting S2 depth patches + split files ...")
    with zipfile.ZipFile(zip_path.path) as z:
        members = [
            n
            for n in z.namelist()
            if not n.endswith("/")
            and (("/depth/s2/" in n) or n.endswith("_split_bathymetry.txt"))
        ]
        z.extractall(path=(raw / "extracted").path, members=members)
    print(f"extracted {len(members)} files")


# ---------------------------------------------------------------------------
# Scan: build one record per annotated bathymetry patch
# ---------------------------------------------------------------------------
def _parse_annotated_ids(area: str) -> list[str]:
    """Return the patch ids in the 'annotated sample' list of s2_split_bathymetry.txt."""
    p = extracted_dir() / area / "s2_split_bathymetry.txt"
    txt = p.read_text()
    # The file has several labelled Python-list blocks; the first bracket after
    # 'annotated sample:' is the full annotated set (train + test union).
    m = re.search(r"annotated sample:\s*\[([^\]]*)\]", txt)
    if not m:
        raise RuntimeError(f"could not parse annotated list in {p}")
    return re.findall(r"\d+", m.group(1))


def scan_records() -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []
    for area in AREA_YEAR:
        ids = _parse_annotated_ids(area)
        for pid in ids:
            tif = extracted_dir() / area / "depth" / "s2" / f"depth_{pid}.tif"
            recs.append(
                {
                    "area": area,
                    "pid": pid,
                    "path": tif.path,
                    "year": AREA_YEAR[area],
                    "source_id": f"{area}/depth_{pid}",
                }
            )
    return recs


# ---------------------------------------------------------------------------
# Write one label patch (reuse source UTM CRS, snap to integer pixel grid)
# ---------------------------------------------------------------------------
def _write_one(rec: dict[str, Any]) -> dict[str, Any] | None:
    sample_id = rec["sample_id"]
    out_tif = io.locations_dir(SLUG) / f"{sample_id}.tif"

    # Always read the source and compute stats (so metadata is correct even on an
    # idempotent re-run); only skip the actual encode/write when the output exists.
    with rasterio.open(rec["path"]) as ds:
        arr = ds.read(1).astype(np.float32)  # (H, W)
        t = ds.transform
        crs = ds.crs
        h, w = arr.shape

    if h > io.MAX_TILE or w > io.MAX_TILE:
        return None  # should not happen (18x18), guard anyway

    # 0.0 is the no-reference / masked fill value -> nodata sentinel.
    depth = arr.copy()
    depth[arr == 0.0] = io.REGRESSION_NODATA

    valid = depth[depth != io.REGRESSION_NODATA]
    if valid.size == 0:
        return {"sample_id": sample_id, "n_valid": 0, "area": rec["area"]}

    if not out_tif.exists():
        proj = Projection(crs, io.RESOLUTION, -io.RESOLUTION)
        col0 = round(t.c / io.RESOLUTION)
        row0 = round(t.f / -io.RESOLUTION)
        bounds = (col0, row0, col0 + w, row0 + h)
        io.write_label_geotiff(
            SLUG, sample_id, depth, proj, bounds, nodata=io.REGRESSION_NODATA
        )
        io.write_sample_json(
            SLUG,
            sample_id,
            proj,
            bounds,
            io.year_range(rec["year"]),
            source_id=rec["source_id"],
        )
    return {
        "sample_id": sample_id,
        "area": rec["area"],
        "n_valid": int(valid.size),
        "min": float(valid.min()),
        "max": float(valid.max()),
        "mean": float(valid.mean()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    if not args.skip_download:
        download_and_extract()

    recs = scan_records()
    print(f"scanned {len(recs)} annotated bathymetry patches")
    # 2857 << 5000 cap -> use all; assign running ids in a stable order.
    recs.sort(key=lambda r: (r["area"], int(r["pid"])))
    recs = recs[:MAX_SAMPLES]
    for i, r in enumerate(recs):
        r["sample_id"] = f"{i:06d}"

    io.locations_dir(SLUG).mkdir(parents=True, exist_ok=True)
    io.check_disk()

    stats: list[dict[str, Any]] = []
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in recs]),
            total=len(recs),
            desc="write",
        ):
            if res is not None:
                stats.append(res)

    written = [s for s in stats if not s.get("skipped") and s.get("n_valid", 0) > 0]
    empty = [s for s in stats if s.get("n_valid", 0) == 0 and not s.get("skipped")]
    num_samples = sum(
        1 for r in recs if (io.locations_dir(SLUG) / f"{r['sample_id']}.tif").exists()
    )
    area_counts = Counter(r["area"] for r in recs)

    pix_min = min((s["min"] for s in written), default=0.0)
    pix_max = max((s["max"] for s in written), default=0.0)
    all_means = np.array([s["mean"] for s in written], dtype=np.float64)

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "regression",
            "source": "Zenodo / IGARSS (record 16753753)",
            "license": "CC-BY-NC-4.0",
            "provenance": {
                "url": URL,
                "have_locally": False,
                "annotation_method": (
                    "SfM-MVS DSM from aerial imagery refraction-corrected and validated "
                    "against reference LiDAR (Agia Napa) / multibeam+LiDAR (Puck Lagoon); "
                    "co-registered to Sentinel-2 patches"
                ),
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "regression": {
                "name": "water_depth",
                "description": (
                    "Shallow-water bathymetry: per-pixel DSM elevation relative to the sea "
                    "surface (metres). Negative = below the water surface (deeper); small "
                    "positive values (Puck Lagoon) are emergent/near-shore land in the DSM. "
                    "Derived from refraction-corrected SfM-MVS aerial DSMs validated against "
                    "reference LiDAR / multibeam echosounder, co-registered to the Sentinel-2 "
                    "10 m grid. Only clear, optically-shallow water is observable, so depths "
                    "span roughly 0 to -30 m."
                ),
                "unit": "meters",
                "dtype": "float32",
                "value_range": [round(pix_min, 2), round(pix_max, 2)],
                "nodata_value": io.REGRESSION_NODATA,
            },
            "num_samples": num_samples,
            "area_counts": dict(area_counts),
            "notes": (
                "S2 depth patches (18x18 @ 10 m) used directly (already local UTM 10 m); "
                "source CRS reused (Agia Napa EPSG:32636, Puck Lagoon EPSG:25834), origin "
                "snapped to the integer 10 m grid (<=5 m), no resampling. Fill value 0.0 -> "
                "nodata (-99999). Annotated bathymetry split (train+test) used: "
                f"{dict(area_counts)}; 2857 total < 5000 cap so all kept (no bucketing). "
                "Time range = 1-year window on S2 acquisition year (Agia Napa 2016, Puck "
                "Lagoon 2021); depth is static so change_time is null. "
                f"per-pixel depth range [{pix_min:.2f}, {pix_max:.2f}] m; "
                f"patch-mean depth p5/p50/p95 = "
                f"{np.percentile(all_means, 5):.2f}/{np.percentile(all_means, 50):.2f}/"
                f"{np.percentile(all_means, 95):.2f} m. "
                f"{len(empty)} patches had no valid (non-zero) pixels and were skipped."
            ),
        },
    )

    print(
        f"wrote {num_samples} label patches; per-pixel depth range "
        f"[{pix_min:.2f}, {pix_max:.2f}] m; area counts {dict(area_counts)}; "
        f"{len(empty)} empty skipped"
    )

    # Verification histogram over patch-mean depths.
    edges = [-30, -25, -20, -15, -10, -5, 0, 5, 15]
    hist, _ = np.histogram(all_means, bins=edges)
    print("patch-mean depth histogram (m):")
    for lo, hi, c in zip(edges[:-1], edges[1:], hist):
        print(f"  [{lo:>4}, {hi:>4}) : {c}")

    manifest.write_registry_entry(
        SLUG, "completed", task_type="regression", num_samples=num_samples
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()

"""Download stratification source data for ERA5 climate-stratified sampling.

Downloads:
  1. Köppen-Geiger 0.1° GeoTIFF (Beck et al. 2023) from figshare/gloh2o
  2. Natural Earth 10m land polygons (shapefile)
  3. ETOPO2022 60-second global surface elevation GeoTIFF from NOAA

All downloads are idempotent — existing files are skipped.

Usage:
    python -m olmoearth_pretrain.dataset_creation.era5_sample.fetch_stratification_sources \
        --output /weka/dfive-default/helios/dataset/era5enc_pretrain/metadata
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Source URLs ---
# Beck et al. (2023) Köppen-Geiger maps - GeoTIFF archive from gloh2o.org
# Contains koppen_geiger_0p1.tif for the 1991-2020 period at 0.1° resolution.
KOPPEN_ZIP_URL = "https://ndownloader.figshare.com/files/42602809"
KOPPEN_ZIP_FNAME = "koppen_geiger_tif.zip"
# The 0.1° map for 1991-2020 inside the archive:
KOPPEN_TIF_RELATIVE = "1991_2020/koppen_geiger_0p1.tif"
KOPPEN_LEGEND_RELATIVE = "legend.txt"

# Natural Earth 10m land polygons
NATURAL_EARTH_URL = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip"
NATURAL_EARTH_ZIP_FNAME = "ne_10m_land.zip"

# ETOPO2022 v1 60-second ice-surface elevation (single global GeoTIFF, ~350MB)
ETOPO_URL = (
    "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/data/"
    "60s/60s_surface_elev_gtif/ETOPO_2022_v1_60s_N90W180_surface.tif"
)
ETOPO_FNAME = "ETOPO_2022_v1_60s_N90W180_surface.tif"


def _download_file(url: str, dest: Path, desc: str) -> None:
    """Download a file with progress bar. Skips if dest already exists."""
    if dest.exists():
        logger.info("Already exists, skipping: %s", dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    logger.info("Downloading %s -> %s", desc, dest)
    resp = requests.get(url, stream=True, timeout=600, allow_redirects=True)
    resp.raise_for_status()
    content_type = resp.headers.get("content-type", "")
    if "text/html" in content_type:
        raise RuntimeError(
            f"Download returned HTML instead of binary data for {desc}. "
            f"URL may require manual download: {url}"
        )
    total = int(resp.headers.get("content-length", 0))
    written = 0
    with (
        open(tmp, "wb") as f,
        tqdm(total=total or None, unit="B", unit_scale=True, desc=desc) as bar,
    ):
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            bar.update(len(chunk))
            written += len(chunk)
    if written == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded 0 bytes for {desc}. URL may be invalid: {url}")
    tmp.rename(dest)
    logger.info("Downloaded %s (%.1f MB)", dest.name, written / 1e6)


def fetch_koppen(output_dir: Path) -> Path:
    """Download and extract the Köppen-Geiger 0.1° GeoTIFF.

    Returns path to the extracted .tif file.
    """
    tif_path = output_dir / "koppen_geiger_0p1.tif"
    if tif_path.exists():
        logger.info("Köppen GeoTIFF already present: %s", tif_path)
        return tif_path

    zip_path = output_dir / KOPPEN_ZIP_FNAME
    _download_file(KOPPEN_ZIP_URL, zip_path, "Köppen-Geiger GeoTIFFs")

    logger.info("Extracting Köppen data from %s", zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        logger.info("Archive contains %d files", len(members))

        # Find the 0.1° GeoTIFF for 1991-2020
        tif_member = None
        legend_member = None
        for m in members:
            if "1991_2020" in m and "0p1" in m and m.endswith(".tif"):
                tif_member = m
            elif m.endswith("legend.txt"):
                legend_member = m

        if tif_member is None:
            # Broader fallback: any 0p1.tif
            for m in members:
                if "0p1" in m and m.endswith(".tif"):
                    tif_member = m
                    break

        if tif_member is None:
            raise FileNotFoundError(
                f"Could not find a 0.1° GeoTIFF in {zip_path}. "
                f"Archive contents: {members[:20]}"
            )

        logger.info("Extracting %s", tif_member)
        with zf.open(tif_member) as src, open(tif_path, "wb") as dst:
            shutil.copyfileobj(src, dst)

        legend_dest = output_dir / "koppen_legend.txt"
        if legend_member and not legend_dest.exists():
            logger.info("Extracting %s", legend_member)
            with zf.open(legend_member) as src, open(legend_dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
    logger.info("Köppen GeoTIFF extracted: %s", tif_path)
    return tif_path


def fetch_land_polygons(output_dir: Path) -> Path:
    """Download Natural Earth 10m land polygons.

    Returns path to the extracted .shp file.
    """
    shp_path = output_dir / "ne_10m_land.shp"
    if shp_path.exists():
        logger.info("Land polygons already present: %s", shp_path)
        return shp_path

    zip_path = output_dir / NATURAL_EARTH_ZIP_FNAME
    _download_file(NATURAL_EARTH_URL, zip_path, "Natural Earth 10m land")

    logger.info("Extracting land polygons from %s", zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)

    if not shp_path.exists():
        raise FileNotFoundError(f"Expected {shp_path} after extraction")
    logger.info("Land polygons extracted: %s", shp_path)
    return shp_path


def fetch_dem(output_dir: Path) -> Path:
    """Download ETOPO2022 60-second global elevation GeoTIFF.

    Returns path to the .tif file.
    """
    tif_path = output_dir / ETOPO_FNAME
    if tif_path.exists():
        logger.info("DEM already present: %s", tif_path)
        return tif_path

    _download_file(ETOPO_URL, tif_path, "ETOPO2022 60s elevation")
    logger.info("DEM downloaded: %s", tif_path)
    return tif_path


def fetch_all(output_dir: str | Path) -> dict[str, Path]:
    """Download all stratification sources. Returns dict of paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "koppen": fetch_koppen(output_dir),
        "land": fetch_land_polygons(output_dir),
        "dem": fetch_dem(output_dir),
    }

    # Write a manifest for provenance
    manifest = output_dir / "SOURCES.txt"
    if not manifest.exists():
        with open(manifest, "w") as f:
            f.write(
                "# Stratification source data for ERA5 climate-stratified sampling\n"
            )
            f.write(f"# Downloaded by {os.path.basename(__file__)}\n\n")
            f.write(f"koppen_url: {KOPPEN_ZIP_URL}\n")
            f.write("koppen_ref: Beck et al. (2023) Sci Data 10, 724\n")
            f.write(f"koppen_file: {paths['koppen'].name}\n\n")
            f.write(f"land_url: {NATURAL_EARTH_URL}\n")
            f.write(f"land_file: {paths['land'].name}\n\n")
            f.write(f"dem_url: {ETOPO_URL}\n")
            f.write(f"dem_file: {paths['dem'].name}\n")

    logger.info("All stratification sources ready in %s", output_dir)
    return paths


def main() -> None:
    """Run the stratification source download CLI."""
    parser = argparse.ArgumentParser(
        description="Download stratification sources for ERA5 climate-stratified sampling."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory to store downloaded source data.",
    )
    args = parser.parse_args()
    fetch_all(args.output)


if __name__ == "__main__":
    main()

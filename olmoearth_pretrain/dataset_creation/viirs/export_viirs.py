"""Export the annual VIIRS night lights composites (VNL v2) from Earth Engine.

Each year's average_masked and cf_cvg bands are exported once as global images at
their native 15 arc-second grid to GCS, then the export pieces are stitched into one
tiled, compressed GeoTIFF per band at {out_dir}/{year}/{band}.tif for
olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth.viirs. Masked pixels are
written as NODATA_VALUE and tagged as nodata.

Years come from NOAA/VIIRS/DNB/ANNUAL_V22 when it has them and ANNUAL_V21 otherwise
(the same products EOG distributes as VNL v2.2 and v2.1).

The export runs as the Earth Engine service account, and the pieces are read back from
GCS with the default application credentials.
"""

import argparse
import os
import tempfile
import time

import ee
import rasterio
import rasterio.windows
from google.cloud import storage

COLLECTIONS = ["NOAA/VIIRS/DNB/ANNUAL_V22", "NOAA/VIIRS/DNB/ANNUAL_V21"]
BANDS = ["average_masked", "cf_cvg"]
NODATA_VALUE = -99999
# EOG's annual composites cover 180W-180E, 65S-75N.
REGION = [-180, -65, 180, 75]
# Rows copied per write when stitching pieces.
ROWS_PER_CHUNK = 1024


def get_image(year: int) -> ee.Image:
    """Get the composite for this year from the first collection that has it."""
    for collection_name in COLLECTIONS:
        collection = ee.ImageCollection(collection_name).filterDate(
            f"{year}-01-01", f"{year + 1}-01-01"
        )
        if collection.size().getInfo() > 0:
            print(f"{year}: using {collection_name}")
            return collection.first()
    raise ValueError(f"no VIIRS annual composite for {year}")


def start_export(image: ee.Image, band: str, bucket: str, prefix: str) -> ee.batch.Task:
    """Start a global export of one band at its native grid."""
    band_image = image.select(band)
    projection = band_image.projection().getInfo()
    task = ee.batch.Export.image.toCloudStorage(
        image=band_image.toFloat().unmask(NODATA_VALUE, False),
        description=prefix.replace("/", "_"),
        bucket=bucket,
        fileNamePrefix=prefix,
        region=ee.Geometry.Rectangle(REGION, "EPSG:4326", False),
        crs=projection["crs"],
        crsTransform=projection["transform"],
        maxPixels=10_000_000_000,
        fileFormat="GeoTIFF",
    )
    task.start()
    return task


def wait_for(tasks: dict[str, ee.batch.Task]) -> None:
    """Wait until all export tasks complete, raising if any fails."""
    pending = dict(tasks)
    while pending:
        time.sleep(60)
        for name, task in list(pending.items()):
            state = task.status()["state"]
            if state in ["UNSUBMITTED", "READY", "RUNNING"]:
                continue
            if state != "COMPLETED":
                raise ValueError(
                    f"export {name} ended in state {state}: {task.status()}"
                )
            print(f"export {name} completed")
            del pending[name]


def stitch(blobs: list[storage.Blob], dst_fname: str, tmp_dir: str) -> None:
    """Stitch the GeoTIFF pieces of one export into a single tiled GeoTIFF."""
    local_fnames = []
    for blob in blobs:
        local_fname = os.path.join(tmp_dir, blob.name.split("/")[-1])
        blob.download_to_filename(local_fname)
        local_fnames.append(local_fname)

    # The pieces share one grid, so each is placed by its pixel offset.
    sources = [rasterio.open(fname) for fname in local_fnames]
    res_x, res_y = sources[0].transform.a, sources[0].transform.e
    left = min(src.bounds.left for src in sources)
    top = max(src.bounds.top for src in sources)
    width = round((max(src.bounds.right for src in sources) - left) / res_x)
    height = round((min(src.bounds.bottom for src in sources) - top) / res_y)
    transform = rasterio.Affine(res_x, 0, left, 0, res_y, top)

    tmp_dst_fname = dst_fname + ".tmp"
    with rasterio.open(
        tmp_dst_fname,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float32",
        crs=sources[0].crs,
        transform=transform,
        nodata=NODATA_VALUE,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        compress="deflate",
        predictor=3,
        BIGTIFF="YES",
    ) as dst:
        for src in sources:
            col_off = round((src.bounds.left - left) / res_x)
            row_off = round((src.bounds.top - top) / res_y)
            for row in range(0, src.height, ROWS_PER_CHUNK):
                rows = min(ROWS_PER_CHUNK, src.height - row)
                data = src.read(
                    1, window=rasterio.windows.Window(0, row, src.width, rows)
                )
                dst.write(
                    data,
                    1,
                    window=rasterio.windows.Window(
                        col_off, row_off + row, src.width, rows
                    ),
                )
    for src in sources:
        src.close()
    os.rename(tmp_dst_fname, dst_fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export annual VIIRS VNL v2 from GEE")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--gcs_bucket", type=str, required=True)
    parser.add_argument("--gcs_prefix", type=str, default="viirs_annual")
    parser.add_argument(
        "--service_account_credentials",
        type=str,
        default="/etc/credentials/gee_credentials.json",
    )
    parser.add_argument("--start_year", type=int, default=2016)
    parser.add_argument("--end_year", type=int, default=2024)
    args = parser.parse_args()

    ee.Initialize(ee.ServiceAccountCredentials(None, args.service_account_credentials))
    bucket = storage.Client().bucket(args.gcs_bucket)

    # Start every export first so Earth Engine runs them concurrently. Exports whose
    # pieces are already on GCS (e.g. from a preempted run) are not restarted.
    tasks: dict[str, ee.batch.Task] = {}
    todo: list[tuple[int, str, str]] = []
    for year in range(args.start_year, args.end_year + 1):
        image = None
        for band in BANDS:
            dst_fname = os.path.join(args.out_dir, str(year), f"{band}.tif")
            if os.path.exists(dst_fname):
                continue
            prefix = f"{args.gcs_prefix}/{year}/{band}"
            todo.append((year, band, prefix))
            if any(bucket.list_blobs(prefix=prefix, max_results=1)):
                continue
            if image is None:
                image = get_image(year)
            tasks[prefix] = start_export(image, band, args.gcs_bucket, prefix)
    print(f"started {len(tasks)} exports, {len(todo)} bands to stitch")
    wait_for(tasks)

    for year, band, prefix in todo:
        year_dir = os.path.join(args.out_dir, str(year))
        os.makedirs(year_dir, exist_ok=True)
        blobs = list(bucket.list_blobs(prefix=prefix))
        print(f"{year} {band}: stitching {len(blobs)} pieces")
        with tempfile.TemporaryDirectory(dir=args.out_dir) as tmp_dir:
            stitch(blobs, os.path.join(year_dir, f"{band}.tif"), tmp_dir)

"""Download Meta's tree canopy height model."""

import multiprocessing
import os
import sys
import tempfile

import boto3
import botocore
import botocore.config
import rasterio
import rasterio.vrt
import tqdm
from rasterio.enums import Resampling
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import (
    GeotiffRasterFormat,
    get_raster_projection_and_bounds,
)
from upath import UPath

BUCKET = "dataforgood-fb-data"
PREFIX = "forests/v1/alsgedi_global_v6_float/chm/"


def process_chm_tif(bucket: str, key: str, out_fname: str) -> None:
    """Download and downsample one CHM GeoTIFF."""
    s3 = boto3.client(
        "s3", config=botocore.config.Config(signature_version=botocore.UNSIGNED)
    )

    # Download to temporary file.
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_fname = UPath(tmp_dir) / "geotiff.tif"
        print(f"downloading from {key} to {local_fname}")
        s3.download_file(Bucket=bucket, Key=key, Filename=local_fname.path)

        # Get the projection and bounds.
        with rasterio.open(local_fname.path) as raster:
            projection, bounds = get_raster_projection_and_bounds(raster)

        if projection.crs.to_epsg() != 3857:
            raise ValueError(f"file at {key} has unexpected projection {projection}")

        # Now convert it to 10 m/pixel.
        wanted_projection = Projection(projection.crs, 10, -10)
        x_factor = projection.x_resolution / wanted_projection.x_resolution
        y_factor = projection.y_resolution / wanted_projection.y_resolution
        wanted_bounds = (
            bounds[0] * x_factor,
            bounds[1] * y_factor,
            bounds[2] * x_factor,
            bounds[3] * y_factor,
        )

        array = GeotiffRasterFormat().decode_raster(
            local_fname.parent,
            wanted_projection,
            wanted_bounds,
            resampling=Resampling.average,
            fname=local_fname.name,
        )
        out_upath = UPath(out_fname)
        print(f"writing {array.shape} to {out_upath}")
        GeotiffRasterFormat().encode_raster(
            out_upath.parent,
            wanted_projection,
            wanted_bounds,
            array,
            fname=(out_upath.name + ".tmp"),
        )
        os.rename(out_upath.path + ".tmp", out_upath.path)


if __name__ == "__main__":
    dst_dir = sys.argv[1]

    # List all the GeoTIFFs we need.
    s3 = boto3.client(
        "s3", config=botocore.config.Config(signature_version=botocore.UNSIGNED)
    )
    paginator = s3.get_paginator("list_objects_v2")
    jobs = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".tif"):
                continue

            dst_fname = os.path.join(dst_dir, key.split("/")[-1])
            if os.path.exists(dst_fname):
                continue

            jobs.append(
                dict(
                    bucket=BUCKET,
                    key=key,
                    out_fname=dst_fname,
                )
            )

    p = multiprocessing.Pool(64)
    outputs = star_imap_unordered(p, process_chm_tif, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()

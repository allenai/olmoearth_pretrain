"""Helpers for rslearn RasterArray-based raster IO."""

import numpy.typing as npt
from rslearn.utils.geometry import PixelBounds, Projection
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import RasterFormat
from upath import UPath


def decode_chw_raster(
    raster_format: RasterFormat,
    path: UPath,
    projection: Projection,
    bounds: PixelBounds,
) -> npt.NDArray:
    """Decode a single-timestep raster as a CHW NumPy array."""
    return raster_format.decode_raster(path, projection, bounds).get_chw_array()


def encode_chw_raster(
    raster_format: RasterFormat,
    path: UPath,
    projection: Projection,
    bounds: PixelBounds,
    array: npt.NDArray,
    fname: str,
) -> None:
    """Encode a CHW NumPy array with rslearn's RasterArray API."""
    raster_format.encode_raster(
        path=path,
        projection=projection,
        bounds=bounds,
        raster=RasterArray(chw_array=array),
        fname=fname,
    )

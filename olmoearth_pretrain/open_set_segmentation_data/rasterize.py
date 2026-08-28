"""Rasterize vector labels (polygons/lines/boxes) into UTM 10 m label patches.

Work entirely in the target projection's *pixel* coordinates: reproject each geometry
with ``geom_to_pixels`` (an rslearn Projection includes resolution, so to_projection
yields pixel coords), then rasterize onto the tile's pixel grid with an offset-identity
transform. This avoids CRS-metre / negative-resolution bookkeeping.
"""

from typing import Any

import numpy as np
from affine import Affine
from rasterio.features import rasterize
from rslearn.utils.geometry import Projection, STGeometry


def geom_to_pixels(
    geom: Any, src_projection: Projection, dst_projection: Projection
) -> Any:
    """Reproject a shapely geometry into dst_projection pixel coordinates."""
    return STGeometry(src_projection, geom, None).to_projection(dst_projection).shp


def rasterize_shapes(
    shapes: list[tuple[Any, int]],
    bounds: tuple[int, int, int, int],
    fill: int,
    dtype: str = "uint8",
    all_touched: bool = False,
) -> np.ndarray:
    """Rasterize (geometry_in_pixel_coords, value) pairs into a (1, H, W) array.

    ``bounds`` are integer pixel bounds (x_min, y_min, x_max, y_max) in the target
    projection (pixel space; y increases downward / southward as usual for these tiles).
    Geometries must already be in that same pixel space (see ``geom_to_pixels``).
    """
    x_min, y_min, x_max, y_max = bounds
    width, height = x_max - x_min, y_max - y_min
    # Map array (col, row) -> pixel (x, y): x = col + x_min, y = row + y_min.
    transform = Affine(1, 0, x_min, 0, 1, y_min)
    arr = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=fill,
        dtype=dtype,
        all_touched=all_touched,
    )
    return arr[np.newaxis, :, :]

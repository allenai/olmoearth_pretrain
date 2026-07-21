"""Tests for the PASTIS processor's embedding-grid helper."""

from rasterio.crs import CRS
from rasterio.warp import transform as warp_transform

from olmoearth_pretrain.evals.datasets.pastis_processor import (
    patch_grid_from_geometry,
)

# A PASTIS-like 1280m x 1280m footprint on the 10m UTM 31N grid.
UTM_CRS = CRS.from_epsg(32631)
X0, Y1 = 500_000.0, 6_700_000.0  # min x, max y
X1, Y0 = X0 + 1280.0, Y1 - 1280.0


def _square_geometry(xs: list[float], ys: list[float]) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [list(zip(xs, ys, strict=True))],
    }


def test_patch_grid_native_utm() -> None:
    """A UTM-native footprint snaps exactly onto its own 10m grid."""
    geometry = _square_geometry([X0, X1, X1, X0, X0], [Y1, Y1, Y0, Y0, Y1])
    bounds, projection = patch_grid_from_geometry(geometry, UTM_CRS)

    assert projection.crs == UTM_CRS
    assert projection.x_resolution == 10.0
    assert projection.y_resolution == -10.0
    assert bounds == (50_000, -670_000, 50_128, -669_872)


def test_patch_grid_from_lonlat_footprint() -> None:
    """A footprint given in EPSG:4326 recovers the same native UTM grid."""
    lons, lats = warp_transform(
        UTM_CRS,
        CRS.from_epsg(4326),
        [X0, X1, X1, X0, X0],
        [Y1, Y1, Y0, Y0, Y1],
    )
    geometry = _square_geometry(lons, lats)
    bounds, projection = patch_grid_from_geometry(geometry, CRS.from_epsg(4326))

    assert projection.crs == UTM_CRS
    assert bounds == (50_000, -670_000, 50_128, -669_872)

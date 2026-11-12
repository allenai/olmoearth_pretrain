"""Tests for the pastis_rslearn export's patch grid and split replay."""

from rasterio.crs import CRS
from rasterio.warp import transform as warp_transform

from olmoearth_pretrain.evals.datasets.pastis_rslearn_export import (
    patch_grid_from_geometry,
    replay_split_patches,
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


def test_replay_split_patches() -> None:
    """Folds 1-3 form train in feature order; short series are dropped."""
    full_year = {str(m): 20180915 + 100 * m for m in range(4)} | {
        str(4 + m): 20190115 + 100 * m for m in range(8)
    }
    short = {"0": 20180915}
    features = [
        {
            "properties": {"ID_PATCH": patch_id, "Fold": fold, "dates-S2": dates},
            "geometry": None,
        }
        for patch_id, fold, dates in [
            (1, 2, full_year),
            (2, 1, full_year),
            (3, 4, full_year),
            (4, 5, full_year),
            (5, 1, short),
        ]
    ]
    splits = replay_split_patches({"features": features})

    assert [p["patch_id"] for p in splits["train"]] == [2, 1]
    assert [p["patch_id"] for p in splits["valid"]] == [3]
    assert [p["patch_id"] for p in splits["test"]] == [4]
    assert splits["train"][0]["months"][:2] == [201809, 201810]
    assert len(splits["train"][0]["months"]) == 12

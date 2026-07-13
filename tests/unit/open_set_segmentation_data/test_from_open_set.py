"""Unit tests for open-set window/label construction helpers."""

from pathlib import Path

import numpy as np
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from rslearn.utils.raster_array import RasterArray
from upath import UPath

from olmoearth_pretrain.dataset_creation.create_windows.from_open_set import (
    LABEL_RASTER_FORMAT,
    ClassLookup,
    _build_open_set_label,
    _build_regression_label,
    _centered_window_bounds,
    _window_geometry_for_sample,
    load_exclusion_index,
    normalize_regression,
)
from olmoearth_pretrain.open_set_segmentation_data.pretrain_constants import (
    OPEN_SET_NODATA,
    REGRESSION_VALUE_MAX_OUT,
    REGRESSION_VALUE_MIN_OUT,
)

CRS_STR = "EPSG:32610"


def test_centered_window_bounds() -> None:
    """Centered window bounds are computed from center pixel and size."""
    assert _centered_window_bounds(100, 200, 128) == (36, 136, 164, 264)


def test_normalize_regression_endpoints() -> None:
    """Regression normalization maps value-range endpoints to output min/max."""
    out = normalize_regression(np.array([0.0, 50.0, 100.0]), [0.0, 100.0])
    assert out[0] == REGRESSION_VALUE_MIN_OUT
    assert out[2] == REGRESSION_VALUE_MAX_OUT
    assert REGRESSION_VALUE_MIN_OUT < out[1] < REGRESSION_VALUE_MAX_OUT


def test_sparse_open_set_label() -> None:
    """A sparse point label is placed at the window center and remapped to global."""
    lookup = ClassLookup(local_to_global={"ds": {5: 42}}, regression={})
    sample = {"kind": "sparse", "slug": "ds", "sample_id": "p1", "label": 5}
    projection = Projection(CRS.from_string(CRS_STR), 10, -10)
    bounds = (-64, -64, 64, 64)
    out = _build_open_set_label(sample, lookup, UPath("."), projection, bounds)
    assert out.shape == (1, 128, 128)
    assert out[0, 64, 64] == 42
    # Everything else is nodata.
    others = out[0].copy()
    others[64, 64] = OPEN_SET_NODATA
    assert np.all(others == OPEN_SET_NODATA)


def test_dense_open_set_label_remaps_and_masks(tmp_path: Path) -> None:
    """A dense source tif is cropped into the window and remapped local->global."""
    datasets_root = UPath(tmp_path)
    slug = "ds"
    locations = datasets_root / slug / "locations"
    locations.mkdir(parents=True)

    projection = Projection(CRS.from_string(CRS_STR), 10, -10)
    # 4x4 source patch with local classes 0,1 and nodata 255.
    src = np.array(
        [
            [0, 1, 1, 255],
            [0, 1, 1, 255],
            [0, 0, 1, 1],
            [255, 255, 1, 1],
        ],
        dtype=np.uint8,
    )
    src_bounds = (1000, 2000, 1004, 2004)
    LABEL_RASTER_FORMAT.encode_raster(
        locations,
        projection,
        src_bounds,
        RasterArray(chw_array=src[np.newaxis, :, :]),
        fname="s1.tif",
    )

    lookup = ClassLookup(local_to_global={slug: {0: 10, 1: 11}}, regression={})
    sample = {
        "kind": "dense",
        "slug": slug,
        "sample_id": "s1",
        "crs": CRS_STR,
        "pixel_bounds": list(src_bounds),
    }
    projection2, bounds, cc, cr = _window_geometry_for_sample(sample)
    out = _build_open_set_label(sample, lookup, datasets_root, projection2, bounds)[0]

    # Window is centered on the patch center (1002, 2002) -> patch occupies the
    # middle of the 128x128 window. Recover the patch region and check the remap.
    off_col = src_bounds[0] - bounds[0]
    off_row = src_bounds[1] - bounds[1]
    patch = out[off_row : off_row + 4, off_col : off_col + 4]
    expected = np.where(src == 255, OPEN_SET_NODATA, np.where(src == 0, 10, 11))
    assert np.array_equal(patch, expected)
    # Outside the patch is nodata.
    assert out[0, 0] == OPEN_SET_NODATA
    assert cc == 1002 and cr == 2002


def test_dense_regression_label(tmp_path: Path) -> None:
    """Dense regression labels emit a dataset-id band and a normalized value band."""
    datasets_root = UPath(tmp_path)
    slug = "canopy"
    locations = datasets_root / slug / "locations"
    locations.mkdir(parents=True)
    projection = Projection(CRS.from_string(CRS_STR), 10, -10)
    src = np.array([[0.0, 50.0], [100.0, -99999.0]], dtype=np.float32)
    src_bounds = (0, 0, 2, 2)
    LABEL_RASTER_FORMAT.encode_raster(
        locations,
        projection,
        src_bounds,
        RasterArray(
            chw_array=src[np.newaxis, :, :],
        ),
        fname="r1.tif",
    )
    lookup = ClassLookup(
        local_to_global={},
        regression={slug: {"dataset_id": 3, "value_range": [0.0, 100.0]}},
    )
    sample = {
        "kind": "dense",
        "slug": slug,
        "sample_id": "r1",
        "crs": CRS_STR,
        "pixel_bounds": list(src_bounds),
    }
    _, bounds, _, _ = _window_geometry_for_sample(sample)
    out = _build_regression_label(sample, lookup, datasets_root, projection, bounds)
    assert out.shape == (2, 128, 128)
    off_col = src_bounds[0] - bounds[0]
    off_row = src_bounds[1] - bounds[1]
    band0 = out[0, off_row : off_row + 2, off_col : off_col + 2]
    band1 = out[1, off_row : off_row + 2, off_col : off_col + 2]
    # Valid pixels get dataset_id 3; nodata pixel gets 0.
    assert band0[0, 0] == 3 and band0[1, 0] == 3
    assert band0[1, 1] == 0  # -99999 source nodata
    assert band1[0, 0] == REGRESSION_VALUE_MIN_OUT  # value 0 -> min
    assert band1[1, 0] == REGRESSION_VALUE_MAX_OUT  # value 100 -> max
    assert band1[1, 1] == 0  # nodata


def test_load_exclusion_index(tmp_path: Path) -> None:
    """Exclusion index loads geometries from a GeoJSON file into an STRtree."""
    import json

    import shapely

    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": shapely.geometry.mapping(shapely.box(0, 0, 1, 1)),
                "properties": {},
            }
        ],
    }
    p = tmp_path / "excl.geojson"
    with open(p, "w") as f:
        json.dump(geojson, f)
    tree = load_exclusion_index(str(p))
    assert tree is not None
    assert len(tree.query(shapely.box(0.5, 0.5, 2, 2), predicate="intersects")) == 1
    assert len(tree.query(shapely.box(5, 5, 6, 6), predicate="intersects")) == 0
    assert load_exclusion_index(None) is None

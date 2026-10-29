"""Unit tests for scripts/tools/build_us_socioeconomic_eval.py."""

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import geopandas as gpd
import numpy as np
import pytest
import shapely
from upath import UPath

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "tools" / "build_us_socioeconomic_eval.py"


@pytest.fixture(scope="module")
def mod() -> ModuleType:
    """Import the script as a module."""
    spec = importlib.util.spec_from_file_location(
        "build_us_socioeconomic_eval", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tile_in_albers(mod: ModuleType, tile: tuple[str, int, int]) -> Any:
    to_albers = mod._transformer(tile[0], mod.EQUAL_AREA_CRS)
    return shapely.transform(
        mod.tile_polygon(*tile),
        lambda xy: np.column_stack(to_albers.transform(xy[:, 0], xy[:, 1])),
    )


def test_tile_for_point_contains_point(mod: ModuleType) -> None:
    """The returned tile's footprint contains the point it was built from."""
    for lon, lat in [(-98.3, 38.1), (-73.95, 40.78), (-122.42, 37.77)]:
        crs, col, row = mod.tile_for_point(lon, lat)
        x, y = mod._transformer("EPSG:4326", crs).transform(lon, lat)
        assert mod.tile_polygon(crs, col, row).contains(shapely.Point(x, y))


def test_tile_labels_people_weighted(mod: ModuleType) -> None:
    """Labels are people-weighted means; suppressed block groups drop out."""
    tile = mod.tile_for_point(-98.3, 38.1)
    footprint = _tile_in_albers(mod, tile)
    minx, miny, maxx, maxy = footprint.bounds
    midx = (minx + maxx) / 2
    # West block group lies fully inside the west half of the tile; the east one
    # is twice the tile's east half, so only half its people count.
    west = shapely.box(minx - 1, miny - 1, midx, maxy + 1).intersection(footprint)
    east_full = shapely.box(midx, miny - 1, maxx + (maxx - midx), maxy + 1)
    bgs = gpd.GeoDataFrame(
        {
            "income": [50_000.0, 100_000.0],
            "home_value": [np.nan, 300_000.0],
            "pop": [400.0, 1200.0],
        },
        geometry=[west, east_full],
        crs=mod.EQUAL_AREA_CRS,
    )
    bgs["area_m2"] = bgs.geometry.area
    labels = mod.tile_labels(bgs, [tile]).iloc[0]

    east_frac = east_full.intersection(footprint).area / east_full.area
    people_east = 1200.0 * east_frac
    people = 400.0 + people_east
    assert labels["people"] == pytest.approx(people, rel=1e-6)
    assert labels["log_pop_density"] == pytest.approx(
        math.log1p(people / mod.TILE_AREA_KM2), rel=1e-6
    )
    expected_income = (400.0 * 50_000 + people_east * 100_000) / people
    assert labels["income"] == pytest.approx(expected_income, rel=1e-6)
    assert labels["log_income"] == pytest.approx(math.log(expected_income))
    # The west block group's home value is suppressed: east alone supplies
    # 600 / 1000 people >= MIN_VALID_FRAC, so the label is the east value.
    assert labels["home_value_valid_frac"] == pytest.approx(people_east / people)
    assert labels["home_value"] == pytest.approx(300_000.0)


def test_label_dropped_below_min_valid_frac(mod: ModuleType) -> None:
    """A tile whose people are mostly in suppressed block groups gets no label."""
    tile = mod.tile_for_point(-98.3, 38.1)
    footprint = _tile_in_albers(mod, tile)
    bgs = gpd.GeoDataFrame(
        {"income": [np.nan], "home_value": [np.nan], "pop": [500.0]},
        geometry=[footprint.buffer(10)],
        crs=mod.EQUAL_AREA_CRS,
    )
    bgs["area_m2"] = bgs.geometry.area
    labels = mod.tile_labels(bgs, [tile]).iloc[0]
    assert "log_income" not in labels or np.isnan(labels["log_income"])
    assert labels["log_pop_density"] > 0


def test_split_is_blocked_and_deterministic(mod: ModuleType) -> None:
    """Points in the same block share a split; the mapping is stable."""
    assert mod.split_for(-98.31, 38.12) == mod.split_for(-98.01, 38.49)
    splits = {
        mod.split_for(lon, lat) for lon in range(-120, -70) for lat in range(26, 48)
    }
    assert splits == {"train", "val", "test"}


def _source_config(tmp_path: Path, months: int = 12) -> UPath:
    """A year-aligned-style config.json with the monthly layers plus extras."""
    layers: dict[str, Any] = {
        "label_raster": {"type": "raster"},
        "gse": {"type": "raster"},
    }
    for prefix, cls in [
        ("sentinel2_l2a_mo", "rslearn.data_sources.planetary_computer.Sentinel2"),
        ("sentinel1_mo", "rslearn.data_sources.planetary_computer.Sentinel1"),
        ("landsat_mo", "rslearn.data_sources.aws_landsat.LandsatOliTirs"),
    ]:
        for m in range(1, months + 1):
            layers[f"{prefix}{m:02d}"] = {
                "type": "raster",
                "band_sets": [{"bands": ["b1"], "dtype": "uint16"}],
                "data_source": {"class_path": cls, "time_offset": f"{(m - 1) * 30}d"},
            }
    path = tmp_path / "source_config.json"
    path.write_text(json.dumps({"layers": layers}))
    return UPath(path)


def test_write_config_copies_only_imagery_layers(
    mod: ModuleType, tmp_path: Path
) -> None:
    """The 36 monthly layers are copied verbatim; the source's other layers are not."""
    source = _source_config(tmp_path)
    ds_path = UPath(tmp_path / "ds")
    mod.write_config(ds_path, source)
    layers = json.loads((ds_path / "config.json").read_text())["layers"]
    expected = json.loads(source.read_text())["layers"]
    assert len(layers) == 37
    assert layers[mod.LABEL_LAYER] == {"type": "vector"}
    assert "gse" not in layers and "label_raster" not in layers
    for name in ("sentinel2_l2a_mo01", "sentinel1_mo06", "landsat_mo12"):
        assert layers[name] == expected[name]


def test_write_config_rejects_incomplete_source(
    mod: ModuleType, tmp_path: Path
) -> None:
    """A source config missing monthly layers is refused rather than half-copied."""
    with pytest.raises(ValueError, match="expected 36"):
        mod.write_config(UPath(tmp_path / "ds"), _source_config(tmp_path, months=11))


def test_write_windows_round_trip(mod: ModuleType, tmp_path: Path) -> None:
    """Windows land on the 128 px grid with their labels as vector properties."""
    ds_path = UPath(tmp_path / "ds")
    mod.write_config(ds_path, _source_config(tmp_path))

    crs, col, row = mod.tile_for_point(-98.3, 38.1)
    labels = mod.pd.DataFrame(
        [
            {
                "crs": crs,
                "col": col,
                "row": row,
                "people": 900.0,
                "log_pop_density": 6.3,
                "income": 60_000.0,
                "log_income": math.log(60_000.0),
                "home_value": np.nan,
                "log_home_value": np.nan,
            }
        ]
    )
    splits = mod.write_windows(ds_path, labels)
    assert sum(splits.values()) == 1

    (window,) = mod.Dataset(ds_path).load_windows()
    assert window.bounds == (col * 128, row * 128, (col + 1) * 128, (row + 1) * 128)
    assert window.options["split"] in {"train", "val", "test"}
    (feature,) = mod.GeojsonVectorFormat().decode_vector(
        window.get_layer_dir(mod.LABEL_LAYER), window.projection, window.bounds
    )
    assert feature.properties["log_income"] == pytest.approx(math.log(60_000.0))
    assert "log_home_value" not in feature.properties

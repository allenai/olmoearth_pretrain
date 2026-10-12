"""Tests for the open-set high-quality subset index selection (label-layer scan)."""

import importlib.util
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from upath import UPath

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "select_h5_indices",
    ROOT / "scripts" / "official" / "v1_3" / "open_set_hq" / "select_h5_indices.py",
)
assert _spec is not None and _spec.loader is not None
select = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(select)

CLASS_MAPPING = {
    "open_set": {
        "classes": [
            {"global_id": 0, "local_id": 0, "name": "cloud", "slug": "cloudsen12"},
            {"global_id": 1, "local_id": 1, "name": "clear", "slug": "cloudsen12"},
            {"global_id": 2, "local_id": 0, "name": "water", "slug": "coast_train"},
            {"global_id": 3, "local_id": 1, "name": "sand", "slug": "coast_train"},
            # A merged concept shared by two datasets (slug null, members listed).
            {
                "global_id": 4,
                "local_id": None,
                "name": "wind_turbine",
                "slug": None,
                "members": [
                    {"local_id": 0, "name": "turbine", "slug": "uswtdb"},
                    {"local_id": 1, "name": "offshore", "slug": "deepowt"},
                ],
            },
            {"global_id": 5, "local_id": 1, "name": "background", "slug": "uswtdb"},
        ]
    },
    "open_set_regression": {
        "datasets": [
            {"dataset_id": 1, "slug": "ai4arctic_asip_sea_ice_dataset"},
            {"dataset_id": 2, "slug": "magicbathynet"},
        ]
    },
}
NODATA = select.OPEN_SET_NODATA


def _write_h5(
    path: Path,
    open_set: np.ndarray | None = None,
    regression_ids: np.ndarray | None = None,
) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("timestamps", data=np.zeros((12, 3), dtype=np.int32))
        if open_set is not None:
            f.create_dataset("open_set", data=open_set[..., None, None])
        if regression_ids is not None:
            regression = np.stack([regression_ids, regression_ids], axis=-1)
            f.create_dataset("open_set_regression", data=regression[..., None, :])


NUM_SAMPLES = 7


def _make_h5_dir(tmp_path: Path) -> UPath:
    """A 7-sample build.

    cloudsen12, coast_train, sea ice (regression), mixed datasets, no labels, a uswtdb
    window (merged wind_turbine concept + uswtdb background) and a window labeled only
    with the merged concept.
    """
    h5_dir = tmp_path / "h5py_data" / "open_set_open_set_regression" / str(NUM_SAMPLES)
    h5_dir.mkdir(parents=True)
    h, w = 8, 8
    nodata = np.full((h, w), NODATA, dtype=np.uint16)

    cloud = nodata.copy()
    cloud[:4] = 0
    cloud[4:] = 1
    _write_h5(h5_dir / "sample_0.h5", open_set=cloud)

    coast = nodata.copy()
    coast[0, 0] = 2  # sparse label
    _write_h5(h5_dir / "sample_1.h5", open_set=coast)

    ice = np.zeros((h, w), dtype=np.uint16)
    ice[2:6, 2:6] = 1
    _write_h5(h5_dir / "sample_2.h5", regression_ids=ice)

    mixed = nodata.copy()
    mixed[:6] = 3  # mostly coast_train ...
    mixed[6:] = 0  # ... with a few cloudsen12 pixels
    _write_h5(h5_dir / "sample_3.h5", open_set=mixed)

    _write_h5(h5_dir / "sample_4.h5", open_set=nodata)

    turbine = np.full((h, w), 5, dtype=np.uint16)  # uswtdb background ...
    turbine[3:5, 3:5] = 4  # ... around a merged wind_turbine concept blob
    _write_h5(h5_dir / "sample_5.h5", open_set=turbine)

    concept_only = nodata.copy()
    concept_only[0, :3] = 4  # only the shared concept: cannot be attributed
    _write_h5(h5_dir / "sample_6.h5", open_set=concept_only)

    with open(h5_dir / "sample_metadata.csv", "w") as f:
        f.write("sample_index,open_set,open_set_regression\n")
        for i in range(NUM_SAMPLES):
            f.write(f"{i},1,0\n")
    return UPath(h5_dir)


def test_slug_lookups_from_mapping() -> None:
    """Class ids and regression dataset ids both resolve to dataset slugs."""
    lookups = select.slug_lookups_from_mapping(CLASS_MAPPING)
    assert lookups.class_to_slug == {
        0: "cloudsen12",
        1: "cloudsen12",
        2: "coast_train",
        3: "coast_train",
        4: "concept:wind_turbine",
        5: "uswtdb",
    }
    assert lookups.regression_to_slug == {
        1: "ai4arctic_asip_sea_ice_dataset",
        2: "magicbathynet",
    }
    assert lookups.concept_members == {
        "concept:wind_turbine": frozenset({"uswtdb", "deepowt"})
    }


def test_resolve_slug() -> None:
    """The most-labeled slug wins; the others are recorded; empty is unknown."""
    assert select.resolve_slug(Counter()) == ("", 0, "")
    assert select.resolve_slug(Counter({"a": 5})) == ("a", 5, "")
    assert select.resolve_slug(Counter({"a": 5, "b": 9, "c": 1})) == ("b", 9, "a|c")


def test_resolve_slug_merged_concepts() -> None:
    """Concept pixels are credited to a co-occurring member dataset, else kept as is."""
    members = {"concept:wind_turbine": frozenset({"uswtdb", "deepowt"})}
    # Concept pixels outnumber the dataset's own, but the dataset is a member.
    assert select.resolve_slug(
        Counter({"concept:wind_turbine": 60, "uswtdb": 4}), members
    ) == ("uswtdb", 64, "")
    # A non-member dataset does not absorb the concept.
    assert select.resolve_slug(
        Counter({"concept:wind_turbine": 60, "cloudsen12": 4}), members
    ) == ("concept:wind_turbine", 60, "cloudsen12")
    # Only the concept: unattributable.
    assert select.resolve_slug(Counter({"concept:wind_turbine": 3}), members) == (
        "concept:wind_turbine",
        3,
        "",
    )


def test_scan_and_select(tmp_path: Path) -> None:
    """Slugs are read off the label layers and the kept indices follow them."""
    h5_dir = _make_h5_dir(tmp_path)
    lookups = select.slug_lookups_from_mapping(CLASS_MAPPING)
    slug_counts = select.scan_h5_dir(h5_dir, range(NUM_SAMPLES), lookups, workers=1)
    assert slug_counts[0] == Counter({"cloudsen12": 64})
    assert slug_counts[1] == Counter({"coast_train": 1})
    assert slug_counts[2] == Counter({"ai4arctic_asip_sea_ice_dataset": 16})
    assert slug_counts[3] == Counter({"coast_train": 48, "cloudsen12": 16})
    assert slug_counts[4] == Counter()
    assert slug_counts[5] == Counter({"uswtdb": 60, "concept:wind_turbine": 4})
    assert slug_counts[6] == Counter({"concept:wind_turbine": 3})

    indices, rows = select.select_indices(
        slug_counts,
        keep_slugs={"coast_train", "ai4arctic_asip_sea_ice_dataset", "uswtdb"},
        concept_members=lookups.concept_members,
    )
    np.testing.assert_array_equal(indices, [1, 2, 3, 5])
    assert indices.dtype == np.int64
    assert rows == [
        (0, "cloudsen12", 64, ""),
        (1, "coast_train", 1, ""),
        (2, "ai4arctic_asip_sea_ice_dataset", 16, ""),
        (3, "coast_train", 48, "cloudsen12"),
        (4, "", 0, ""),
        (5, "uswtdb", 64, ""),
        (6, "concept:wind_turbine", 3, ""),
    ]
    assert select.count_metadata_rows(h5_dir) == NUM_SAMPLES


def test_unknown_ids_are_reported_not_dropped(tmp_path: Path) -> None:
    """Ids missing from the class mapping surface as unknown_* slugs."""
    h5_dir = _make_h5_dir(tmp_path)
    bad = np.full((8, 8), 999, dtype=np.uint16)
    _write_h5(Path(str(h5_dir)) / "sample_0.h5", open_set=bad)
    lookups = select.slug_lookups_from_mapping(CLASS_MAPPING)
    counts = select.slug_counts_of_h5(h5_dir / "sample_0.h5", lookups)
    assert counts == Counter({"unknown_class_999": 64})

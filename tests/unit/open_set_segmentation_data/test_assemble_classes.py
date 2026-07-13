"""Unit tests for open-set class assembly."""

import json
from pathlib import Path

from upath import UPath

from olmoearth_pretrain.open_set_segmentation_data.assemble_classes import (
    assemble_classes,
)
from olmoearth_pretrain.open_set_segmentation_data.pretrain_constants import (
    PRESENCE_ONLY_GROUP,
)


def _write_dataset(root: Path, slug: str, metadata: dict) -> None:
    d = root / slug
    d.mkdir(parents=True, exist_ok=True)
    with (d / "metadata.json").open("w") as f:
        json.dump(metadata, f)


def _registry(*slugs: str) -> dict:
    return {
        "datasets": [
            {"slug": s, "status": "completed", "task_type": "classification"}
            for s in slugs
        ]
    }


def test_contiguous_disjoint_global_ids(tmp_path: Path) -> None:
    """Global ids are contiguous and disjoint, assigned in slug order."""
    root = tmp_path / "datasets"
    _write_dataset(
        root,
        "b_dataset",
        {
            "task_type": "classification",
            "classes": [
                {"id": 0, "name": "background"},
                {"id": 1, "name": "water"},
            ],
        },
    )
    _write_dataset(
        root,
        "a_dataset",
        {
            "task_type": "classification",
            "classes": [
                {"id": 0, "name": "background"},
                {"id": 1, "name": "coffee"},
                {"id": 2, "name": "maize"},
            ],
        },
    )

    result = assemble_classes(
        datasets_root=UPath(root),
        summaries_dir=None,
        registry=_registry("b_dataset", "a_dataset"),
    )
    open_set = result.mapping["open_set"]

    # a_dataset sorts first -> global ids 0,1,2; b_dataset -> 3,4.
    ids = [c["global_id"] for c in open_set["classes"]]
    assert ids == [0, 1, 2, 3, 4]
    assert open_set["num_classes"] == 5

    a_group = next(d for d in open_set["training_datasets"] if d["name"] == "a_dataset")
    b_group = next(d for d in open_set["training_datasets"] if d["name"] == "b_dataset")
    assert a_group["global_ids"] == [0, 1, 2]
    assert b_group["global_ids"] == [3, 4]
    assert set(a_group["global_ids"]).isdisjoint(b_group["global_ids"])


def test_presence_only_merged_into_one_group(tmp_path: Path) -> None:
    """Datasets with no background class are merged into the presence-only group."""
    root = tmp_path / "datasets"
    # Two presence-only datasets (no background/negative class).
    _write_dataset(
        root,
        "hillforts",
        {"task_type": "classification", "classes": [{"id": 0, "name": "hillfort"}]},
    )
    _write_dataset(
        root,
        "species",
        {"task_type": "classification", "classes": [{"id": 0, "name": "oak"}]},
    )
    # A normal dataset with a background class.
    _write_dataset(
        root,
        "landcover",
        {
            "task_type": "classification",
            "classes": [
                {"id": 0, "name": "background"},
                {"id": 1, "name": "forest"},
            ],
        },
    )

    result = assemble_classes(
        datasets_root=UPath(root),
        summaries_dir=None,
        registry=_registry("hillforts", "species", "landcover"),
    )
    open_set = result.mapping["open_set"]

    presence_groups = [
        d for d in open_set["training_datasets"] if d["name"] == PRESENCE_ONLY_GROUP
    ]
    assert len(presence_groups) == 1
    # hillforts (global 0) + species (global 2) are the presence-only classes.
    presence = presence_groups[0]
    assert presence["presence_only"] is True
    names = {
        c["name"]
        for c in open_set["classes"]
        if c["global_id"] in presence["global_ids"]
    }
    assert names == {"hillfort", "oak"}

    # landcover stays its own group.
    assert any(d["name"] == "landcover" for d in open_set["training_datasets"])


def test_regression_registry_and_exclusions(tmp_path: Path) -> None:
    """Regression datasets get 1-based ids; excluded slugs are dropped."""
    root = tmp_path / "datasets"
    _write_dataset(
        root,
        "canopy_height",
        {
            "task_type": "regression",
            "regression": {
                "name": "canopy_height",
                "unit": "meters",
                "dtype": "float32",
                "value_range": [0.0, 61.2],
                "nodata_value": -99999,
            },
        },
    )
    _write_dataset(
        root,
        "eurosat",
        {"task_type": "classification", "classes": [{"id": 0, "name": "crop"}]},
    )

    registry = {
        "datasets": [
            {"slug": "canopy_height", "status": "completed", "task_type": "regression"},
            {"slug": "eurosat", "status": "completed", "task_type": "classification"},
        ]
    }
    result = assemble_classes(
        datasets_root=UPath(root), summaries_dir=None, registry=registry
    )

    regression = result.mapping["open_set_regression"]
    assert len(regression["datasets"]) == 1
    entry = regression["datasets"][0]
    assert entry["dataset_id"] == 1  # 1-based; 0 is nodata
    assert entry["value_range"] == [0.0, 61.2]

    # eurosat is excluded -> no classes.
    assert result.mapping["open_set"]["num_classes"] == 0
    assert "eurosat" in result.mapping["excluded_slugs"]


def test_ambiguous_presence_only_flagged(tmp_path: Path) -> None:
    """A dataset called presence-only in its summary but with a background class is flagged."""
    root = tmp_path / "datasets"
    summaries = tmp_path / "summaries"
    summaries.mkdir()
    _write_dataset(
        root,
        "weird",
        {
            "task_type": "classification",
            "classes": [
                {"id": 0, "name": "background"},
                {"id": 1, "name": "kelp"},
            ],
        },
    )
    (summaries / "weird.md").write_text("This is a presence-only dataset of kelp.")

    result = assemble_classes(
        datasets_root=UPath(root),
        summaries_dir=summaries,
        registry=_registry("weird"),
    )
    assert result.ambiguous_presence_only == ["weird"]

"""Integration tests for studio ingest registry I/O."""

from pathlib import Path

from olmoearth_pretrain.evals.studio_ingest.registry import REGISTRY_PATH, Registry
from olmoearth_pretrain.evals.studio_ingest.schema import EvalDatasetEntry


def test_registry_roundtrip_from_repo_registry(tmp_path: Path) -> None:
    """Load repo registry, save with new entry, then reload."""
    assert REGISTRY_PATH.exists()

    registry = Registry.load()
    existing_names = set(registry.list_names())

    new_name = "integration_registry_roundtrip_test_dataset"
    if new_name in existing_names:
        new_name = f"{new_name}_copy"

    registry.add(
        EvalDatasetEntry(
            name=new_name,
            source_path="/tmp/source",
            weka_path="/tmp/weka",
            task_type="classification",
            num_classes=2,
            modalities=["sentinel2_l2a"],
            imputes=[],
        )
    )

    out_path = tmp_path / "registry_roundtrip.json"
    registry.save(path=str(out_path))

    reloaded = Registry.load(path=str(out_path))
    reloaded_names = set(reloaded.list_names())

    assert existing_names.issubset(reloaded_names)
    assert new_name in reloaded_names

    new_entry = reloaded.get(new_name)
    assert new_entry.task_type == "classification"
    assert new_entry.num_classes == 2

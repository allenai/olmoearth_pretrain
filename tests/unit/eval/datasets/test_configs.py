"""Test configs are properly constructed."""

from olmoearth_pretrain.evals.datasets.configs import DATASET_TO_CONFIG, TaskType, dataset_to_config
from olmoearth_pretrain.evals.studio_ingest.schema import EvalDatasetEntry


def test_gb2_dataset_configs() -> None:
    """Every ``gb2-*`` entry resolves and satisfies task-type constraints."""
    gb2_names = sorted(k for k in DATASET_TO_CONFIG if k.startswith("gb2-"))
    assert gb2_names
    for name in gb2_names:
        cfg = dataset_to_config(name)
        assert cfg.supported_modalities
        assert cfg.num_classes >= 1
        if cfg.task_type == TaskType.SEGMENTATION:
            assert cfg.height_width is not None
        elif cfg.task_type == TaskType.REGRESSION:
            assert cfg.height_width is None
        else:
            assert cfg.task_type == TaskType.CLASSIFICATION
            assert cfg.height_width is None


def test_segmentation_tasks_have_hw() -> None:
    """Segmentation tasks require a defined h/w."""
    for dataset, config in DATASET_TO_CONFIG.items():
        if config.task_type == TaskType.SEGMENTATION:
            assert config.height_width is not None, (
                f"No height width for segmentation task {dataset}"
            )


def test_eval_dataset_entry_to_eval_config() -> None:
    """Test converting EvalDatasetEntry to EvalDatasetConfig."""
    entry = EvalDatasetEntry(
        name="tolbi_crops",
        source_path="",
        weka_path="",
        task_type=TaskType.SEGMENTATION.value,
        num_classes=9,
        modalities=["SENTINEL2_L2A", "SENTINEL1"],
        window_size=64,
        is_multilabel=False,
        timeseries=True,
        imputes=[],
    )

    config = entry.to_eval_config()

    assert config.task_type == TaskType.SEGMENTATION
    assert config.num_classes == 9
    assert config.is_multilabel is False
    assert config.supported_modalities == ["sentinel2_l2a", "sentinel1"]
    assert config.height_width == 64  # segmentation uses window_size
    assert config.timeseries is True
    assert config.imputes == []


def test_eval_dataset_entry_classification_no_height_width() -> None:
    """Test that classification tasks don't get height_width."""
    entry = EvalDatasetEntry(
        name="test_classification",
        source_path="",
        weka_path="",
        task_type=TaskType.CLASSIFICATION.value,
        num_classes=10,
        modalities=["SENTINEL2_L2A"],
        window_size=64,
    )

    config = entry.to_eval_config()

    assert config.task_type == TaskType.CLASSIFICATION
    assert config.height_width is None  # classification doesn't use height_width

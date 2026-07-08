"""Test configs are properly constructed."""

import pytest

from olmoearth_pretrain.evals.datasets.configs import (
    DATASET_TO_CONFIG,
    EvalDatasetConfig,
    TaskType,
    dataset_to_config,
)
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
        elif cfg.task_type == TaskType.CLASSIFICATION:
            assert cfg.height_width is None
        # regression: height_width is optional (spatial tasks like biomassters set it)


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


def test_eval_dataset_entry_scalar_regression_no_height_width() -> None:
    """Scalar regression (per-sample) does not get a height_width."""
    entry = EvalDatasetEntry(
        name="test_scalar_regression",
        source_path="",
        weka_path="",
        task_type=TaskType.WINDOW_REGRESSION.value,
        num_classes=1,
        modalities=["SENTINEL1"],
        window_size=64,
    )

    config = entry.to_eval_config()

    assert config.task_type == TaskType.WINDOW_REGRESSION
    assert config.height_width is None  # scalar regression pools to (B, D)


def test_eval_dataset_entry_dense_regression_has_height_width() -> None:
    """Dense (per-pixel) regression still uses window_size as height_width."""
    entry = EvalDatasetEntry(
        name="test_dense_regression",
        source_path="",
        weka_path="",
        task_type=TaskType.PER_PIXEL_REGRESSION.value,
        num_classes=1,
        modalities=["SENTINEL2_L2A"],
        window_size=32,
    )

    config = entry.to_eval_config()

    assert config.task_type == TaskType.PER_PIXEL_REGRESSION
    assert config.height_width == 32


@pytest.mark.parametrize(
    "task_type",
    [TaskType.PER_PIXEL_REGRESSION, TaskType.WINDOW_REGRESSION],
)
def test_regression_requires_single_class(task_type: TaskType) -> None:
    """Regression task types reject num_classes != 1."""
    with pytest.raises(ValueError, match="num_classes=1"):
        EvalDatasetConfig(
            task_type=task_type,
            imputes=[],
            num_classes=2,
            is_multilabel=False,
            supported_modalities=[],
        )


@pytest.mark.parametrize(
    "task_type",
    [TaskType.PER_PIXEL_REGRESSION, TaskType.WINDOW_REGRESSION],
)
def test_regression_allows_single_class(task_type: TaskType) -> None:
    """Regression task types accept num_classes == 1."""
    config = EvalDatasetConfig(
        task_type=task_type,
        imputes=[],
        num_classes=1,
        is_multilabel=False,
        supported_modalities=[],
    )
    assert config.num_classes == 1

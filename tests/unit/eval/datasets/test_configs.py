"""Test configs are properly constructed."""

from olmoearth_pretrain.evals.datasets.configs import DATASET_TO_CONFIG, TaskType


def test_segmentation_tasks_have_hw() -> None:
    """Segmentation tasks require a defined h/w."""
    for dataset, config in DATASET_TO_CONFIG.items():
        if config.task_type == TaskType.SEGMENTATION:
            assert config.height_width is not None, (
                f"No height width for segmentation task {dataset}"
            )

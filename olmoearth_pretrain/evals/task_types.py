"""Task type enum shared across eval modules.

This module exists to avoid circular imports between:
- olmoearth_pretrain.evals.datasets.configs
- olmoearth_pretrain.evals.studio_ingest.schema
"""

from enum import Enum


class TaskType(Enum):
    """Possible task types."""

    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"


def get_eval_mode(task_type: TaskType) -> str:
    """Get the eval mode for a given task type."""
    if task_type == TaskType.CLASSIFICATION:
        return "knn"
    else:
        return "linear_probe"

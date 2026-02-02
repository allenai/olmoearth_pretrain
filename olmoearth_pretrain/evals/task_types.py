"""Task type utilities for eval modules."""

from enum import Enum


class TaskType(Enum):
    """Possible task types."""

    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"

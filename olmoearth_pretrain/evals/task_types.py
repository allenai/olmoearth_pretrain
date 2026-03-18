"""Task type and split enums for eval modules."""

from enum import StrEnum


class TaskType(StrEnum):
    """Possible task types."""

    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"


class SplitName(StrEnum):
    """Standard split names."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"

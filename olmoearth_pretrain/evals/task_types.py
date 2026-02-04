"""Task type and split enums for eval modules."""

from enum import StrEnum


class TaskType(StrEnum):
    """Possible task types."""

    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    REGRESSION = "regression"


class SplitType(StrEnum):
    """How splits are defined in the dataset."""

    GROUPS = "groups"  # Splits are rslearn groups
    TAGS = "tags"      # Splits are rslearn window tags


class SplitName(StrEnum):
    """Standard split names."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"

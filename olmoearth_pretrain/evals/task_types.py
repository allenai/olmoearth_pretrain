"""Task type and split enums for eval modules."""

from enum import StrEnum


class TaskType(StrEnum):
    """Possible task types.

    ``REGRESSION`` is dense (per-pixel) regression; ``SCALAR_REGRESSION`` is
    per-sample regression (one target value per window, e.g. vessel length).
    """

    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    REGRESSION = "regression"
    SCALAR_REGRESSION = "scalar_regression"


class SplitName(StrEnum):
    """Standard split names."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"

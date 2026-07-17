"""Task type and split enums for eval modules."""

from enum import StrEnum


class TaskType(StrEnum):
    """Possible task types.

    ``PER_PIXEL_REGRESSION`` is dense (per-pixel) regression; ``WINDOW_REGRESSION``
    is per-sample regression (one target value per window, e.g. vessel length).
    """

    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    DIAGNOSTIC = "diagnostic"
    PER_PIXEL_REGRESSION = "regression"
    WINDOW_REGRESSION = "scalar_regression"


class SplitName(StrEnum):
    """Standard split names."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"

"""A common home for all eval dataset configs."""

from enum import Enum
from typing import NamedTuple


class TaskType(Enum):
    """Possible task types."""

    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"


class EvalDatasetConfig(NamedTuple):
    """EvalDatasetConfig configs."""

    task_type: TaskType
    imputes: list[tuple[str, str]]
    num_classes: int
    is_multilabel: bool
    # this is only necessary for segmentation tasks,
    # and defines the input / output height width.
    height_width: int | None = None


DATASET_TO_CONFIG = {
    "m-eurosat": EvalDatasetConfig(
        task_type=TaskType.CLASSIFICATION,
        imputes=[],
        num_classes=10,
        is_multilabel=False,
    ),
    "mados": EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[
            ("05 - Vegetation Red Edge", "06 - Vegetation Red Edge"),
            ("08A - Vegetation Red Edge", "09 - Water vapour"),
            ("11 - SWIR", "10 - SWIR - Cirrus"),
        ],
        num_classes=15,
        is_multilabel=False,
        height_width=80,
    ),
    "sen1floods11": EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[],
        num_classes=2,
        is_multilabel=False,
        height_width=64,
    ),
    "pastis": EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[],
        num_classes=19,
        is_multilabel=False,
        height_width=64,
    ),
    "pastis-r": EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[],
        num_classes=19,
        is_multilabel=False,
        height_width=64,
    ),
    "breizhcrops": EvalDatasetConfig(
        task_type=TaskType.CLASSIFICATION,
        imputes=[],
        num_classes=9,
        is_multilabel=False,
        height_width=1,
    ),
}
ALL_DATASETS = list(DATASET_TO_CONFIG.keys())

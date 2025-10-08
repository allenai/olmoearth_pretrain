"""A common home for all eval dataset configs."""

from enum import Enum
from typing import NamedTuple

from olmoearth_pretrain.data.constants import Modality


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


class EvalDatasetConfig(NamedTuple):
    """EvalDatasetConfig configs."""

    task_type: TaskType
    imputes: list[tuple[str, str]]
    num_classes: int
    is_multilabel: bool
    supported_modalities: list[str]
    # this is only necessary for segmentation tasks,
    # and defines the input / output height width.
    height_width: int | None = None
    timeseries: bool = False


DATASET_TO_CONFIG = {
    "m-eurosat": EvalDatasetConfig(
        task_type=TaskType.CLASSIFICATION,
        imputes=[],
        num_classes=10,
        is_multilabel=False,
        supported_modalities=[Modality.SENTINEL2_L2A.name],
    ),
    "m-bigearthnet": EvalDatasetConfig(
        task_type=TaskType.CLASSIFICATION,
        imputes=[("11 - SWIR", "10 - SWIR - Cirrus")],
        num_classes=43,
        is_multilabel=True,
        supported_modalities=[Modality.SENTINEL2_L2A.name],
    ),
    "m-so2sat": EvalDatasetConfig(
        task_type=TaskType.CLASSIFICATION,
        imputes=[
            ("02 - Blue", "01 - Coastal aerosol"),
            ("08A - Vegetation Red Edge", "09 - Water vapour"),
            ("11 - SWIR", "10 - SWIR - Cirrus"),
        ],
        num_classes=17,
        is_multilabel=False,
        supported_modalities=[Modality.SENTINEL2_L2A.name],
    ),
    "m-brick-kiln": EvalDatasetConfig(
        task_type=TaskType.CLASSIFICATION,
        imputes=[],
        num_classes=2,
        is_multilabel=False,
        supported_modalities=[Modality.SENTINEL2_L2A.name],
    ),
    "m-sa-crop-type": EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[("11 - SWIR", "10 - SWIR - Cirrus")],
        num_classes=10,
        is_multilabel=False,
        height_width=256,
        supported_modalities=[Modality.SENTINEL2_L2A.name],
    ),
    "m-cashew-plant": EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[("11 - SWIR", "10 - SWIR - Cirrus")],
        num_classes=7,
        is_multilabel=False,
        height_width=256,
        supported_modalities=[Modality.SENTINEL2_L2A.name],
    ),
    "m-forestnet": EvalDatasetConfig(
        task_type=TaskType.CLASSIFICATION,
        imputes=[
            # src (we have), tgt (we want), using the geobench L8 names
            # we don't need to impute B8 since our
            # _landsathelios2geobench_name implicitly does it for us,
            ("02 - Blue", "01 - Coastal aerosol"),
            ("07 - SWIR2", "09 - Cirrus"),
            ("07 - SWIR2", "10 - Tirs1"),
        ],
        num_classes=12,
        is_multilabel=False,
        supported_modalities=[Modality.LANDSAT.name],
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
        supported_modalities=[Modality.SENTINEL2_L2A.name],
    ),
    "sen1floods11": EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[],
        num_classes=2,
        is_multilabel=False,
        height_width=64,
        supported_modalities=[Modality.SENTINEL1.name],
    ),
    "pastis": EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[],
        num_classes=19,
        is_multilabel=False,
        height_width=64,
        supported_modalities=[Modality.SENTINEL2_L2A.name, Modality.SENTINEL1.name],
        timeseries=True,
    ),
    "pastis128": EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[],
        num_classes=19,
        is_multilabel=False,
        height_width=128,
        supported_modalities=[Modality.SENTINEL2_L2A.name, Modality.SENTINEL1.name],
        timeseries=True,
    ),
    "breizhcrops": EvalDatasetConfig(
        task_type=TaskType.CLASSIFICATION,
        imputes=[],
        num_classes=9,
        is_multilabel=False,
        height_width=1,
        supported_modalities=[Modality.SENTINEL2_L2A.name],
        timeseries=True,
    ),
    "sickle": EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[],
        num_classes=2,
        is_multilabel=False,
        height_width=32,
        supported_modalities=[
            Modality.LANDSAT.name,
            Modality.SENTINEL1.name,
            Modality.SENTINEL2_L2A.name,
        ],
        timeseries=True,
    ),
    "cropharvest": EvalDatasetConfig(
        task_type=TaskType.CLASSIFICATION,
        imputes=[
            ("02 - Blue", "01 - Coastal aerosol"),
            ("11 - SWIR", "10 - SWIR - Cirrus"),
        ],
        num_classes=2,
        is_multilabel=False,
        supported_modalities=[
            Modality.SENTINEL2_L2A.name,
            Modality.SENTINEL1.name,
            Modality.SRTM.name,
        ],
        timeseries=True,
    ),
    "nandi": EvalDatasetConfig(
        task_type=TaskType.CLASSIFICATION,
        imputes=[],
        num_classes=6,
        is_multilabel=False,
        supported_modalities=[
            Modality.SENTINEL2_L2A.name,
            Modality.SENTINEL1.name,
            Modality.LANDSAT.name,
        ],
        timeseries=True,
    ),
    "awf": EvalDatasetConfig(
        task_type=TaskType.CLASSIFICATION,
        imputes=[],
        num_classes=9,
        is_multilabel=False,
        supported_modalities=[
            Modality.SENTINEL2_L2A.name,
            Modality.SENTINEL1.name,
            Modality.LANDSAT.name,
        ],
        timeseries=True,
    ),
}


def dataset_to_config(dataset: str) -> EvalDatasetConfig:
    """Retrieve the correct config for a given dataset."""
    key = "cropharvest" if dataset.startswith("cropharvest") else dataset
    try:
        return DATASET_TO_CONFIG[key]
    except KeyError:
        raise ValueError(f"Unrecognized dataset: {dataset}")

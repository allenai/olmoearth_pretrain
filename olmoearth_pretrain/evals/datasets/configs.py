"""A common home for all eval dataset configs."""

from dataclasses import asdict, dataclass
from typing import Any

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.evals.studio_ingest import get_dataset_entry
from olmoearth_pretrain.evals.task_types import TaskType


def get_eval_mode(task_type: TaskType) -> str:
    """Get the eval mode for a given task type."""
    if task_type == TaskType.CLASSIFICATION:
        return "knn"
    return "linear_probe"


__all__ = ["TaskType", "get_eval_mode", "EvalDatasetConfig"]


@dataclass
class EvalDatasetConfig:
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

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        d = asdict(self)
        d["task_type"] = self.task_type.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvalDatasetConfig":
        """Deserialize from dict."""
        d = d.copy()
        d["task_type"] = TaskType(d["task_type"])
        d["imputes"] = [tuple(x) for x in d["imputes"]]
        return cls(**d)


DATASET_TO_CONFIG = {
    # Dummy config — only used for embedding diagnostics, not actual classification.
    "pretrain_subset": EvalDatasetConfig(
        task_type=TaskType.CLASSIFICATION,
        imputes=[],
        num_classes=1,
        is_multilabel=False,
        supported_modalities=[
            Modality.SENTINEL2_L2A.name,
            Modality.SENTINEL1.name,
            Modality.LANDSAT.name,
        ],
    ),
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
            # we don't need to impute B8 since our band name conversion does it for us
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

# "gb2-" prefix denotes geobench v2 datasets
_GB2_DATASET_TO_CONFIG: dict[str, EvalDatasetConfig] = {
    "gb2-benv2": EvalDatasetConfig(
        task_type=TaskType.CLASSIFICATION,
        imputes=[],
        num_classes=19,
        is_multilabel=True,
        supported_modalities=[Modality.SENTINEL2_L2A.name],
    ),
    "gb2-biomassters": EvalDatasetConfig(
        task_type=TaskType.REGRESSION,
        imputes=[],
        num_classes=1,
        is_multilabel=False,
        supported_modalities=[Modality.SENTINEL1.name, Modality.SENTINEL2_L2A.name],
        timeseries=True,
    ),
    "gb2-burn_scars": EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[],
        num_classes=2,
        is_multilabel=False,
        height_width=512,
        supported_modalities=[Modality.SENTINEL2_L2A.name],
    ),
    "gb2-caffe": EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[],
        num_classes=4,
        is_multilabel=False,
        height_width=512,
        supported_modalities=[Modality.SENTINEL2_L2A.name],
    ),
    "gb2-cloudsen12": EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[],
        num_classes=4,
        is_multilabel=False,
        height_width=512,
        supported_modalities=[Modality.SENTINEL2_L2A.name],
    ),
    "gb2-dynamic_earthnet": EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[],
        num_classes=7,
        is_multilabel=False,
        height_width=512,
        supported_modalities=[Modality.NAIP.name, Modality.SENTINEL2_L2A.name],
    ),
    "gb2-everwatch": EvalDatasetConfig(
        task_type=TaskType.CLASSIFICATION,
        imputes=[],
        num_classes=8,
        is_multilabel=True,
        supported_modalities=[Modality.NAIP.name],
    ),
    "gb2-flair2": EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[],
        num_classes=13,
        is_multilabel=False,
        height_width=512,
        supported_modalities=[Modality.NAIP.name, Modality.SRTM.name],
    ),
    "gb2-forestnet": EvalDatasetConfig(
        task_type=TaskType.CLASSIFICATION,
        imputes=[],
        num_classes=12,
        is_multilabel=False,
        supported_modalities=[Modality.LANDSAT.name],
    ),
    "gb2-fotw": EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[],
        num_classes=3,
        is_multilabel=False,
        height_width=256,
        supported_modalities=[Modality.NAIP.name],
    ),
    "gb2-kuro_siwo": EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[],
        num_classes=4,
        is_multilabel=False,
        height_width=224,
        supported_modalities=[Modality.SENTINEL1.name, Modality.SRTM.name],
    ),
    "gb2-nzcattle": EvalDatasetConfig(
        task_type=TaskType.CLASSIFICATION,
        imputes=[],
        num_classes=2,
        is_multilabel=False,
        supported_modalities=[Modality.NAIP.name],
    ),
    "gb2-pastis": EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[],
        num_classes=20,
        is_multilabel=False,
        height_width=128,
        supported_modalities=[Modality.SENTINEL2_L2A.name, Modality.SENTINEL1.name],
        timeseries=True,
    ),
    "gb2-so2sat": EvalDatasetConfig(
        task_type=TaskType.CLASSIFICATION,
        imputes=[],
        num_classes=17,
        is_multilabel=False,
        supported_modalities=[Modality.SENTINEL2_L2A.name, Modality.SENTINEL1.name],
    ),
    "gb2-spacenet2": EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[],
        num_classes=3,
        is_multilabel=False,
        height_width=512,
        supported_modalities=[Modality.SENTINEL2_L2A.name],
    ),
    "gb2-spacenet7": EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[],
        num_classes=3,
        is_multilabel=False,
        height_width=512,
        supported_modalities=[Modality.SENTINEL2_L2A.name],
    ),
    "gb2-substation": EvalDatasetConfig(
        task_type=TaskType.CLASSIFICATION,
        imputes=[],
        num_classes=2,
        is_multilabel=False,
        supported_modalities=[Modality.SENTINEL2_L2A.name],
    ),
    "gb2-treesatai": EvalDatasetConfig(
        task_type=TaskType.CLASSIFICATION,
        imputes=[],
        num_classes=15,
        is_multilabel=True,
        supported_modalities=[Modality.SENTINEL2_L2A.name],
    ),
}
DATASET_TO_CONFIG.update(_GB2_DATASET_TO_CONFIG)


def dataset_to_config(dataset: str) -> EvalDatasetConfig:
    """Get EvalDatasetConfig by name, checking both hardcoded and registry.

    First checks DATASET_TO_CONFIG dict, then falls back to registry.

    Args:
        dataset: Dataset name to look up.

    Returns:
        EvalDatasetConfig for the dataset.

    Raises:
        ValueError: If dataset not found in either location.
    """
    if dataset in DATASET_TO_CONFIG:
        return DATASET_TO_CONFIG[dataset]

    entry = get_dataset_entry(dataset)
    return entry.to_eval_config()

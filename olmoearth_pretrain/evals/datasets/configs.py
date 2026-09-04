"""A common home for all eval dataset configs."""

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.evals.studio_ingest import get_dataset_entry
from olmoearth_pretrain.evals.task_types import TaskType


def get_eval_mode(task_type: TaskType) -> str:
    """Get the eval mode for a given task type."""
    if task_type == TaskType.CLASSIFICATION:
        return "knn"
    if task_type in (
        TaskType.SEGMENTATION,
        TaskType.PER_PIXEL_REGRESSION,
        TaskType.WINDOW_REGRESSION,
    ):
        return "linear_probe"
    raise ValueError(f"Unsupported task type: {task_type}")


__all__ = ["TaskType", "get_eval_mode", "EvalDatasetConfig", "SourceImagery"]


class SourceImagery(StrEnum):
    """A source sensor/imagery type an eval dataset is collected from.

    ``EvalDatasetConfig.source_imagery`` is a list of these (a dataset may fuse
    several sensors). This is distinct from ``supported_modalities``, which
    records the model modality slot(s) the data is fed through: several datasets
    are not collected from a sensor OlmoEarth trains against, and their imagery
    is rescaled/mapped into the closest training modality (usually
    ``SENTINEL2_L2A``) before being passed to the model. This records what the
    data actually is so that masquerading is explicit in the config rather than
    buried in dataloader comments.
    """

    # Native training modalities.
    SENTINEL2 = "sentinel2"
    SENTINEL1 = "sentinel1"
    SRTM = "srtm"
    # Sources not trained against, rescaled/mapped into a training modality.
    # Harmonized Landsat Sentinel-2 surface reflectance.
    HLS = "hls"
    # PlanetScope (Dove) optical imagery.
    PLANETSCOPE = "planetscope"
    # Maxar WorldView-2 multispectral (+ panchromatic).
    WORLDVIEW2 = "worldview2"
    # Single-channel (grayscale) aerial/amplitude imagery.
    GRAYSCALE_AERIAL = "grayscale_aerial"
    # 5-band RGB + NIR + elevation aerial imagery.
    RGBNE_AERIAL = "rgbne_aerial"


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
    # Source sensor(s) the dataset is collected from (see SourceImagery). A list
    # because a dataset may fuse several sensors. Defaults to empty only for
    # datasets whose source we haven't recorded (e.g. the dynamic registry).
    source_imagery: list[SourceImagery] = field(default_factory=list)
    # Optional z-score normalization for REGRESSION targets, matching GeoBench-2's
    # protocol (stats estimated on the train split). When both are set, labels are
    # normalized as (y - mean) / std, so reported RMSE is in standardized units
    # (and thus comparable to other models / GeoBench-2). None = raw targets.
    target_mean: float | None = None
    target_std: float | None = None
    # Corrections for datasets whose upstream band NAMES are misassigned to the
    # data. Distinct from `imputes`, which fills in bands that are genuinely
    # absent: here every band is present, but the source labelled them wrongly.
    # Maps the name the source claims -> the band the data actually is. Applied
    # to the sample band names and to the dataset's own normalization stats, so
    # that everything downstream stays name-driven and needs no special-casing.
    band_name_corrections: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate task-type-specific invariants."""
        # Regression heads and metrics only support a single output channel, so
        # num_classes must be 1 for both dense (per-pixel) and per-sample
        # (window) regression tasks.
        if (
            self.task_type
            in (TaskType.PER_PIXEL_REGRESSION, TaskType.WINDOW_REGRESSION)
            and self.num_classes != 1
        ):
            raise ValueError(
                f"{self.task_type} only supports num_classes=1 "
                f"(single regression target), got num_classes={self.num_classes}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        d = asdict(self)
        d["task_type"] = self.task_type.value
        d["source_imagery"] = [s.value for s in self.source_imagery]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvalDatasetConfig":
        """Deserialize from dict."""
        d = d.copy()
        d["task_type"] = TaskType(d["task_type"])
        d["imputes"] = [tuple(x) for x in d["imputes"]]
        if "source_imagery" in d:
            d["source_imagery"] = [SourceImagery(s) for s in d["source_imagery"]]
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
            Modality.GSE.name,
        ],
    ),
    "pretrain_subset_worldcover": EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[],
        num_classes=11,
        is_multilabel=False,
        height_width=32,
        supported_modalities=[
            Modality.SENTINEL2_L2A.name,
            Modality.SENTINEL1.name,
            Modality.LANDSAT.name,
            Modality.GSE.name,
        ],
    ),
    "pretrain_subset_osm": EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[],
        num_classes=30,
        is_multilabel=False,
        height_width=32,
        supported_modalities=[
            Modality.SENTINEL2_L2A.name,
            Modality.SENTINEL1.name,
            Modality.LANDSAT.name,
            Modality.GSE.name,
        ],
    ),
    "pretrain_subset_srtm": EvalDatasetConfig(
        task_type=TaskType.PER_PIXEL_REGRESSION,
        imputes=[],
        num_classes=1,
        is_multilabel=False,
        height_width=32,
        supported_modalities=[
            Modality.SENTINEL2_L2A.name,
            Modality.SENTINEL1.name,
            Modality.LANDSAT.name,
            Modality.GSE.name,
        ],
    ),
    "pretrain_subset_canopy": EvalDatasetConfig(
        task_type=TaskType.PER_PIXEL_REGRESSION,
        imputes=[],
        num_classes=1,
        is_multilabel=False,
        height_width=32,
        supported_modalities=[
            Modality.SENTINEL2_L2A.name,
            Modality.SENTINEL1.name,
            Modality.LANDSAT.name,
            Modality.GSE.name,
        ],
    ),
    "pretrain_subset_meta_canopy": EvalDatasetConfig(
        task_type=TaskType.PER_PIXEL_REGRESSION,
        imputes=[],
        num_classes=1,
        is_multilabel=False,
        height_width=32,
        supported_modalities=[
            Modality.SENTINEL2_L2A.name,
            Modality.SENTINEL1.name,
            Modality.LANDSAT.name,
        ],
    ),
    "pretrain_subset_glo30_elevation": EvalDatasetConfig(
        task_type=TaskType.PER_PIXEL_REGRESSION,
        imputes=[],
        num_classes=1,
        is_multilabel=False,
        height_width=32,
        supported_modalities=[
            Modality.SENTINEL2_L2A.name,
            Modality.SENTINEL1.name,
            Modality.LANDSAT.name,
        ],
    ),
    "pretrain_subset_glo30_slope": EvalDatasetConfig(
        task_type=TaskType.PER_PIXEL_REGRESSION,
        imputes=[],
        num_classes=1,
        is_multilabel=False,
        height_width=32,
        supported_modalities=[
            Modality.SENTINEL2_L2A.name,
            Modality.SENTINEL1.name,
            Modality.LANDSAT.name,
        ],
    ),
    # Aspect is a compass bearing, so it is probed as sin/cos rather than raw
    # degrees: the 0/360 seam makes degree-space MSE ill-posed. See
    # pretrain_subset.GLO30_LABEL_ASPECT_SIN.
    "pretrain_subset_glo30_aspect_sin": EvalDatasetConfig(
        task_type=TaskType.PER_PIXEL_REGRESSION,
        imputes=[],
        num_classes=1,
        is_multilabel=False,
        height_width=32,
        supported_modalities=[
            Modality.SENTINEL2_L2A.name,
            Modality.SENTINEL1.name,
            Modality.LANDSAT.name,
        ],
    ),
    "pretrain_subset_glo30_aspect_cos": EvalDatasetConfig(
        task_type=TaskType.PER_PIXEL_REGRESSION,
        imputes=[],
        num_classes=1,
        is_multilabel=False,
        height_width=32,
        supported_modalities=[
            Modality.SENTINEL2_L2A.name,
            Modality.SENTINEL1.name,
            Modality.LANDSAT.name,
        ],
    ),
    "pretrain_subset_cdl": EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[],
        # CDL stores uint8 class codes; many codes are unused but bincount/probe
        # sizes are still tractable at this width.
        num_classes=256,
        is_multilabel=False,
        height_width=32,
        supported_modalities=[
            Modality.SENTINEL2_L2A.name,
            Modality.SENTINEL1.name,
            Modality.LANDSAT.name,
            Modality.GSE.name,
        ],
    ),
    "pretrain_subset_worldcereal": EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[],
        # Binary segmentation on the WorldCereal temporary-crops channel.
        num_classes=2,
        is_multilabel=False,
        height_width=32,
        supported_modalities=[
            Modality.SENTINEL2_L2A.name,
            Modality.SENTINEL1.name,
            Modality.LANDSAT.name,
            Modality.GSE.name,
        ],
    ),
    "m-eurosat": EvalDatasetConfig(
        task_type=TaskType.CLASSIFICATION,
        imputes=[],
        num_classes=10,
        is_multilabel=False,
        supported_modalities=[Modality.SENTINEL2_L2A.name],
        # GeoBench names EuroSAT's 13 tif channels positionally from a list with
        # B8A ninth (geo-bench make_benchmark/dataset_converters/eurosat.py, via
        # `gb.sentinel2_13_bands[band_idx]`), but the tifs put B8A LAST:
        # B01..B08, B09, B10, B11, B12, B8A. Channels 1-8 are unaffected; the
        # last five names are rotated by one, so four of the bands we feed the
        # model are the wrong band and true B11 never reaches it at all (it sits
        # under the cirrus name, which SENTINEL2_L2A.band_order drops).
        # Verified against the original EuroSAT tifs (EuroSAT100, 100 files):
        # channel 13 is the NIR plateau (corr 0.98 with B08) and channel 10 is
        # cirrus (mean ~12). Upstream fix for reference: torchgeo PR #2480.
        band_name_corrections={
            "08A - Vegetation Red Edge": "09 - Water vapour",
            "09 - Water vapour": "10 - SWIR - Cirrus",
            "10 - SWIR - Cirrus": "11 - SWIR",
            "11 - SWIR": "12 - SWIR",
            "12 - SWIR": "08A - Vegetation Red Edge",
        },
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
        # B06 and B09 were never exported, and B10 is only needed to build the
        # 13-slot stack (SENTINEL2_L2A.band_order drops it). Sources are the
        # nearest present neighbour; the last two match m-so2sat's imputes.
        imputes=[
            ("07 - Vegetation Red Edge", "06 - Vegetation Red Edge"),
            ("08A - Vegetation Red Edge", "09 - Water vapour"),
            ("11 - SWIR", "10 - SWIR - Cirrus"),
        ],
        num_classes=2,
        is_multilabel=False,
        supported_modalities=[Modality.SENTINEL2_L2A.name],
        # Same GeoBench converter bug as m-eurosat (brick_kiln.py:105, again
        # `gb.sentinel2_13_bands[i]` positional) but a worse fit: this source
        # never held the 13 MSI bands. SustainBench's docs claim "13 bands, B1
        # through B12", which is both impossible and false -- the export
        # (mliu356/kiln-scaling, notebooks/get_images.ipynb) is
        # ['B1','B2','B3','B4','B5','B7','B8A','B8','B11','B12','TCI_R','TCI_G',
        # 'TCI_B']. So B06/B09/B10 are absent, three channels are 8-bit
        # true-colour renders rather than spectral bands (measured: R/G/B / 9.9
        # clipped at 255, R2 0.998), and `08 - NIR` is right only by luck --
        # the export lists B8A before B8 and the two offsets cancel.
        band_name_corrections={
            "06 - Vegetation Red Edge": "07 - Vegetation Red Edge",
            "07 - Vegetation Red Edge": "08A - Vegetation Red Edge",
            "08A - Vegetation Red Edge": "11 - SWIR",
            "09 - Water vapour": "12 - SWIR",
            "10 - SWIR - Cirrus": "TCI_R (not a spectral band)",
            "11 - SWIR": "TCI_G (not a spectral band)",
            "12 - SWIR": "TCI_B (not a spectral band)",
        },
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
    # 50Cities: single-timestep S2+S1 land-cover segmentation, 64x64 tiles.
    # 13 semantic classes (the two nodata/background colors are mapped to the
    # segmentation ignore label by FiftyCitiesProcessor; see colormap.json).
    # One config per split mode (random / by_city / by_continent); the split
    # mode is encoded in the dataset name and resolved in get_eval_dataset.
    # supported_modalities lists what the data provides; the actual S2-only vs
    # S1+S2 runs are separate EVAL_TASKS in all_evals.py.
    # when we don't add a suffix "fifty_cities" its random.
    "fifty_cities": EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[],
        num_classes=13,
        is_multilabel=False,
        height_width=64,
        supported_modalities=[Modality.SENTINEL2_L2A.name, Modality.SENTINEL1.name],
        timeseries=False,
    ),
    "fifty_cities_by_city": EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[],
        num_classes=13,
        is_multilabel=False,
        height_width=64,
        supported_modalities=[Modality.SENTINEL2_L2A.name, Modality.SENTINEL1.name],
        timeseries=False,
    ),
    "fifty_cities_by_continent": EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[],
        num_classes=13,
        is_multilabel=False,
        height_width=64,
        supported_modalities=[Modality.SENTINEL2_L2A.name, Modality.SENTINEL1.name],
        timeseries=False,
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
        supported_modalities=[Modality.SENTINEL2_L2A.name, Modality.SENTINEL1.name],
        source_imagery=[SourceImagery.SENTINEL2, SourceImagery.SENTINEL1],
    ),
    "gb2-biomassters": EvalDatasetConfig(
        task_type=TaskType.PER_PIXEL_REGRESSION,
        imputes=[],
        num_classes=1,
        is_multilabel=False,
        height_width=256,
        supported_modalities=[Modality.SENTINEL1.name, Modality.SENTINEL2_L2A.name],
        source_imagery=[SourceImagery.SENTINEL1, SourceImagery.SENTINEL2],
        timeseries=True,
        # AGB targets in tons/ha; full train-split stats so RMSE matches
        # GeoBench-2's z-scored convention. To get their leaderboard score:
        # 1 - rmse * 0.2538.
        target_mean=63.96,
        target_std=72.52,
    ),
    "gb2-burn_scars": EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[],
        num_classes=2,
        is_multilabel=False,
        height_width=512,
        supported_modalities=[Modality.SENTINEL2_L2A.name],
        source_imagery=[SourceImagery.HLS],
    ),
    "gb2-caffe": EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[],
        num_classes=3,
        is_multilabel=False,
        height_width=512,
        # CaFFe is SAR (primarily Sentinel-1) glacier imagery, not optical.
        supported_modalities=[Modality.SENTINEL1.name],
        source_imagery=[SourceImagery.SENTINEL1],
    ),
    "gb2-cloudsen12": EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[],
        num_classes=4,
        is_multilabel=False,
        height_width=512,
        supported_modalities=[Modality.SENTINEL2_L2A.name],
        source_imagery=[SourceImagery.SENTINEL2],
    ),
    "gb2-kuro_siwo": EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[],
        num_classes=3,
        is_multilabel=False,
        height_width=224,
        supported_modalities=[Modality.SENTINEL1.name, Modality.SRTM.name],
        source_imagery=[SourceImagery.SENTINEL1, SourceImagery.SRTM],
    ),
    "gb2-spacenet2": EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[],
        num_classes=3,
        is_multilabel=False,
        height_width=512,
        supported_modalities=[Modality.SENTINEL2_L2A.name],
        source_imagery=[SourceImagery.WORLDVIEW2],
    ),
    "gb2-spacenet7": EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[],
        num_classes=3,
        is_multilabel=False,
        height_width=512,
        supported_modalities=[Modality.SENTINEL2_L2A.name],
        source_imagery=[SourceImagery.PLANETSCOPE],
    ),
    "gb2-treesatai": EvalDatasetConfig(
        task_type=TaskType.CLASSIFICATION,
        imputes=[],
        num_classes=15,
        is_multilabel=True,
        supported_modalities=[Modality.SENTINEL2_L2A.name],
        source_imagery=[SourceImagery.SENTINEL2],
    ),
    "gb2-flair2": EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[],
        num_classes=13,
        is_multilabel=False,
        height_width=512,
        supported_modalities=[Modality.SENTINEL2_L2A.name],
        source_imagery=[SourceImagery.RGBNE_AERIAL],
    ),
    "gb2-fotw": EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[],
        num_classes=3,
        is_multilabel=False,
        height_width=256,
        supported_modalities=[Modality.SENTINEL2_L2A.name],
        source_imagery=[SourceImagery.SENTINEL2],
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

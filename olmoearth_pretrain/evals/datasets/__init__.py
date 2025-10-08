"""OlmoEarth Pretrain eval datasets."""

import logging

from olmo_core.config import StrEnum
from torch.utils.data import Dataset

from .breizhcrops import BREIZHCROPS_DIR, BreizhCropsDataset
from .cropharvest import CROPHARVEST_DIR, CropHarvestDataset
from .floods_dataset import FLOODS_DIR, Sen1Floods11Dataset
from .geobench_dataset import GEOBENCH_DIR, GeobenchDataset
from .mados_dataset import MADOS_DIR, MADOSDataset
from .normalize import NormMethod
from .pastis_dataset import PASTIS_DIR, PASTIS_DIR_ORIG, PASTISRDataset
from .rslearn_dataset import RslearnToOlmoEarthDataset
from .sickle_dataset import SICKLE_DIR, SICKLEDataset

logger = logging.getLogger(__name__)


class EvalDatasetPartition(StrEnum):
    """Enum for different dataset partitions."""

    TRAIN1X = "default"
    TRAIN_001X = "0.01x_train"  # Not valid for non train split
    TRAIN_002X = "0.02x_train"
    TRAIN_005X = "0.05x_train"
    TRAIN_010X = "0.10x_train"
    TRAIN_020X = "0.20x_train"
    TRAIN_050X = "0.50x_train"


def get_eval_dataset(
    eval_dataset: str,
    split: str,
    norm_stats_from_pretrained: bool = False,
    input_modalities: list[str] = [],
    input_layers: list[str] = [],
    partition: str = EvalDatasetPartition.TRAIN1X,
    norm_method: str = NormMethod.NORM_NO_CLIP,
) -> Dataset:
    """Retrieve an eval dataset from the dataset name."""
    if input_modalities:
        if not (
            eval_dataset.startswith("cropharvest")
            or (eval_dataset in ["pastis", "pastis128", "sickle", "nandi", "awf"])
        ):
            raise ValueError(
                f"input_modalities is only supported for multimodal tasks, got {eval_dataset}"
            )

    if input_layers:
        if eval_dataset not in ["nandi", "awf"]:
            raise ValueError(
                f"input_layers is only supported for rslearn tasks, got {eval_dataset}"
            )

    if eval_dataset.startswith("m-"):
        # m- == "modified for geobench"
        return GeobenchDataset(
            geobench_dir=GEOBENCH_DIR,
            dataset=eval_dataset,
            split=split,
            partition=partition,
            norm_stats_from_pretrained=norm_stats_from_pretrained,
            norm_method=norm_method,
        )
    elif eval_dataset == "mados":
        if norm_stats_from_pretrained:
            logger.warning(
                "MADOS has very different norm stats than our pretraining dataset"
            )
        return MADOSDataset(
            path_to_splits=MADOS_DIR,
            split=split,
            partition=partition,
            norm_stats_from_pretrained=norm_stats_from_pretrained,
            norm_method=norm_method,
        )
    elif eval_dataset == "sen1floods11":
        return Sen1Floods11Dataset(
            path_to_splits=FLOODS_DIR,
            split=split,
            partition=partition,
            norm_stats_from_pretrained=norm_stats_from_pretrained,
            norm_method=norm_method,
        )
    elif eval_dataset.startswith("pastis"):
        kwargs = {
            "split": split,
            "partition": partition,
            "norm_stats_from_pretrained": norm_stats_from_pretrained,
            "input_modalities": input_modalities,
            "norm_method": norm_method,
        }
        if "128" in eval_dataset:
            # "pastis128"
            kwargs["path_to_splits"] = PASTIS_DIR_ORIG
        else:
            kwargs["path_to_splits"] = PASTIS_DIR
        return PASTISRDataset(**kwargs)  # type: ignore
    elif eval_dataset == "breizhcrops":
        return BreizhCropsDataset(
            path_to_splits=BREIZHCROPS_DIR,
            split=split,
            partition=partition,
            norm_stats_from_pretrained=norm_stats_from_pretrained,
            norm_method=norm_method,
        )
    elif eval_dataset == "sickle":
        return SICKLEDataset(
            path_to_splits=SICKLE_DIR,
            split=split,
            partition=partition,
            norm_stats_from_pretrained=norm_stats_from_pretrained,
            input_modalities=input_modalities,
            norm_method=norm_method,
        )
    elif eval_dataset.startswith("cropharvest"):
        # e.g. "cropharvest_Togo_12"
        try:
            _, country, timesteps = eval_dataset.split("_")
        except ValueError:
            raise ValueError(
                "CropHarvest tasks should have the following naming format: cropharvest_<country>_<timesteps> (e.g. 'cropharvest_Togo_12')"
            )
        return CropHarvestDataset(
            cropharvest_dir=CROPHARVEST_DIR,
            country=country,
            split=split,
            partition=partition,
            norm_stats_from_pretrained=norm_stats_from_pretrained,
            timesteps=int(timesteps),
            input_modalities=input_modalities,
            norm_method=norm_method,
        )
    elif eval_dataset == "nandi":
        return RslearnToOlmoEarthDataset(
            ds_path="/weka/dfive-default/rslearn-eai/datasets/crop/kenya_nandi/20250625",
            ds_groups=["groundtruth_polygon_split_window_32"],
            layers=input_layers,
            input_size=4,
            split=split,
            property_name="category",
            classes=["Coffee", "Trees", "Grassland", "Maize", "Sugarcane", "Tea"],
            partition=partition,
            norm_stats_from_pretrained=norm_stats_from_pretrained,
            norm_method=norm_method,
            input_modalities=input_modalities,
            start_time="2022-09-01",
            end_time="2023-09-01",
            ds_norm_stats_json="nandi_band_stats.json",
        )
    elif eval_dataset == "awf":
        return RslearnToOlmoEarthDataset(
            ds_path="/weka/dfive-default/rslearn-eai/datasets/crop/awf_2023",
            ds_groups=["20250822"],
            layers=input_layers,
            input_size=32,
            split=split,
            property_name="lulc",
            classes=[
                "Agriculture/Settlement",
                "Grassland/barren",
                "Herbaceous wetland",
                "Lava forest",
                "Montane forest",
                "Open water",
                "Shrubland/Savanna",
                "Urban/dense development",
                "Woodland forest (>40% canopy)",
            ],
            partition=partition,
            norm_stats_from_pretrained=norm_stats_from_pretrained,
            norm_method=norm_method,
            input_modalities=input_modalities,
            start_time="2023-01-01",
            end_time="2023-12-31",
            ds_norm_stats_json="awf_band_stats.json",
        )
    else:
        raise ValueError(f"Unrecognized eval_dataset {eval_dataset}")

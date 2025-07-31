"""Helios eval datasets."""

import logging

from olmo_core.config import StrEnum
from torch.utils.data import Dataset

from .breizhcrops import BREIZHCROPS_DIR, BreizhCropsDataset
from .cropharvest import CROPHARVEST_DIR, CropHarvestDataset
from .floods_dataset import FLOODS_DIR, Sen1Floods11Dataset
from .geobench_dataset import GEOBENCH_DIR, GeobenchDataset
from .mados_dataset import MADOS_DIR, MADOSDataset
from .pastis_dataset import PASTIS_DIR, PASTISRDataset
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
    partition: str = EvalDatasetPartition.TRAIN1X,
) -> Dataset:
    """Retrieve an eval dataset from the dataset name."""
    if input_modalities:
        if not (
            eval_dataset.startswith("cropharvest")
            or (eval_dataset in ["pastis", "sickle"])
        ):
            raise ValueError(
                f"input_modalities is only supported for multimodal tasks, got {eval_dataset}"
            )

    if eval_dataset.startswith("m-"):
        # m- == "modified for geobench"
        return GeobenchDataset(
            geobench_dir=GEOBENCH_DIR,
            dataset=eval_dataset,
            split=split,
            partition=partition,
            norm_stats_from_pretrained=norm_stats_from_pretrained,
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
        )
    elif eval_dataset == "sen1floods11":
        return Sen1Floods11Dataset(
            path_to_splits=FLOODS_DIR,
            split=split,
            partition=partition,
            norm_stats_from_pretrained=norm_stats_from_pretrained,
        )
    elif eval_dataset == "pastis":
        return PASTISRDataset(
            path_to_splits=PASTIS_DIR,
            split=split,
            partition=partition,
            norm_stats_from_pretrained=norm_stats_from_pretrained,
            input_modalities=input_modalities,
        )
    elif eval_dataset == "breizhcrops":
        return BreizhCropsDataset(
            path_to_splits=BREIZHCROPS_DIR,
            split=split,
            partition=partition,
            norm_stats_from_pretrained=norm_stats_from_pretrained,
        )
    elif eval_dataset == "sickle":
        return SICKLEDataset(
            path_to_splits=SICKLE_DIR,
            split=split,
            partition=partition,
            norm_stats_from_pretrained=norm_stats_from_pretrained,
            input_modalities=input_modalities,
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
        )
    else:
        raise ValueError(f"Unrecognized eval_dataset {eval_dataset}")

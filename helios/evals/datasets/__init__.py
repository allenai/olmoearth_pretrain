"""Helios eval datasets."""

import logging

from torch.utils.data import Dataset

from .breizhcrops import BREIZHCROPS_DIR, BreizhCropsDataset
from .configs import ALL_DATASETS
from .floods_dataset import FLOODS_DIR, Sen1Floods11Dataset
from .geobench_dataset import GEOBENCH_DIR, GeobenchDataset
from .mados_dataset import MADOS_DIR, MADOSDataset
from .pastis_dataset import PASTIS_DIR, PASTISRDataset
from .sickle_dataset import SICKLE_DIR, SICKLEDataset

logger = logging.getLogger(__name__)


def get_eval_dataset(
    eval_dataset: str,
    split: str,
    norm_stats_from_pretrained: bool = False,
    input_modalities: list[str] = [],
    partition: str = "default",
) -> Dataset:
    """Retrieve an eval dataset from the dataset name."""
    if eval_dataset not in ALL_DATASETS:
        raise ValueError(f"Unrecognized dataset {eval_dataset}")

    if input_modalities:
        if eval_dataset not in ["pastis", "sickle"]:
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
    else:
        raise ValueError(f"Unrecognized eval_dataset {eval_dataset}")

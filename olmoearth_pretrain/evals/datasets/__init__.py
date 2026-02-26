"""OlmoEarth Pretrain eval datasets."""

import logging

from olmo_core.config import StrEnum
from torch.utils.data import Dataset

import olmoearth_pretrain.evals.datasets.paths as paths
from olmoearth_pretrain.evals.studio_ingest.registry import get_dataset_entry

from .breizhcrops import BreizhCropsDataset
from .floods_dataset import Sen1Floods11Dataset
from .geobench_dataset import GeobenchDataset
from .mados_dataset import MADOSDataset
from .normalize import NormMethod
from .pastis_dataset import PASTISRDataset
from .rslearn_dataset import from_registry_entry

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
    # Default to 2std no clip - this matches what our model sees in pretraining,
    # so when using dataset stats (e.g. for MADOS) consistency is important.
    norm_method: str = NormMethod.NORM_NO_CLIP_2_STD,
) -> Dataset:
    """Retrieve an eval dataset from the dataset name."""
    if eval_dataset.startswith("m-"):
        # m- == "modified for geobench"
        return GeobenchDataset(
            geobench_dir=paths.GEOBENCH_DIR,
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
            path_to_splits=paths.MADOS_DIR,
            split=split,
            partition=partition,
            norm_stats_from_pretrained=norm_stats_from_pretrained,
            norm_method=norm_method,
        )
    elif eval_dataset == "sen1floods11":
        return Sen1Floods11Dataset(
            path_to_splits=paths.FLOODS_DIR,
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
            "dir_partition": paths.PASTIS_DIR_PARTITION,
        }
        if "128" in eval_dataset:
            # "pastis128"
            kwargs["path_to_splits"] = paths.PASTIS_DIR_ORIG
        else:
            kwargs["path_to_splits"] = paths.PASTIS_DIR
        return PASTISRDataset(**kwargs)  # type: ignore
    elif eval_dataset == "breizhcrops":
        return BreizhCropsDataset(
            path_to_splits=paths.BREIZHCROPS_DIR,
            split=split,
            partition=partition,
            norm_stats_from_pretrained=norm_stats_from_pretrained,
            norm_method=norm_method,
        )
    else:
        eval_dataset_entry = get_dataset_entry(eval_dataset)
        return from_registry_entry(
            entry=eval_dataset_entry,
            split=split,
            norm_stats_from_pretrained=norm_stats_from_pretrained,
            norm_method=norm_method,
            input_modalities_override=input_modalities if input_modalities else None,
        )

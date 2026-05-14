"""OlmoEarth Pretrain eval datasets."""

import logging
from typing import Any

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
from .pretrain_subset import PretrainSubsetDataset
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


LABEL_FRACTION_TO_PARTITION: dict[float, str] = {
    0.01: EvalDatasetPartition.TRAIN_001X,
    0.02: EvalDatasetPartition.TRAIN_002X,
    0.05: EvalDatasetPartition.TRAIN_005X,
    0.10: EvalDatasetPartition.TRAIN_010X,
    0.20: EvalDatasetPartition.TRAIN_020X,
    0.50: EvalDatasetPartition.TRAIN_050X,
    1.00: EvalDatasetPartition.TRAIN1X,
}


def fraction_to_partition(label_fraction: float) -> str:
    """Map a supported train-label fraction to an existing partition name.

    Standard eval datasets use precomputed partition files or GeoBench
    partition names. Unsupported fractions fail fast so low-label sweeps do not
    silently fall back to full data.
    """
    try:
        return LABEL_FRACTION_TO_PARTITION[label_fraction]
    except KeyError:
        valid = ", ".join(f"{value:g}" for value in sorted(LABEL_FRACTION_TO_PARTITION))
        raise ValueError(
            f"Unsupported label_fraction {label_fraction}. Supported values are: {valid}"
        )


def scale_train_samples(train_samples: int, label_fraction: float) -> int:
    """Scale pretrain-probe train samples for low-label runs."""
    if not 0 < label_fraction <= 1:
        raise ValueError("label_fraction must be in (0, 1].")
    return max(1, int(train_samples * label_fraction))


def get_eval_dataset(
    eval_dataset: str,
    split: str,
    norm_stats_from_pretrained: bool = False,
    input_modalities: list[str] = [],
    label_fraction: float = 1.0,
    # Default to 2std no clip - this matches what our model sees in pretraining,
    # so when using dataset stats (e.g. for MADOS) consistency is important.
    norm_method: str = NormMethod.NORM_NO_CLIP_2_STD,
    **kwargs: Any,
) -> Dataset:
    """Build the dataset wrapper for a downstream evaluation task.

    Args:
        eval_dataset: Registry name or built-in dataset key.
        split: Split to load: ``train``, ``valid``/``val``, or ``test``.
        norm_stats_from_pretrained: Whether to use pretraining normalization stats.
        input_modalities: Optional modality override for multimodal datasets.
        label_fraction: Fraction of training labels to use for low-label evals.
        norm_method: Dataset normalization strategy when not using pretrain stats.
        **kwargs: Dataset-family specific options, including pretrain probe target
            modality and split sizing.

    Returns:
        A PyTorch dataset that yields eval samples and labels.
    """
    if eval_dataset.startswith("pretrain_subset"):
        return PretrainSubsetDataset(
            h5py_dir=kwargs["h5py_dir"],
            training_modalities=kwargs.get("training_modalities", input_modalities),
            max_samples=kwargs.get("max_samples", 512),
            patch_size=kwargs.get("pretrain_patch_size", 4),
            hw_p=kwargs.get("pretrain_hw_p", 8),
            seed=kwargs.get("pretrain_seed", 42),
            split=kwargs.get("pretrain_split", split),
            target_modality=kwargs.get("target_modality"),
            label_seed=kwargs.get("pretrain_label_seed", 42),
            train_samples=scale_train_samples(
                kwargs.get("pretrain_train_samples", 512), label_fraction
            ),
            valid_samples=kwargs.get("pretrain_valid_samples", 512),
            test_samples=kwargs.get("pretrain_test_samples", 512),
            split_strategy=kwargs.get("pretrain_split_strategy", "random"),
            geographic_bin_size_deg=kwargs.get("pretrain_geographic_bin_size_deg", 5.0),
        )
    elif eval_dataset.startswith("m-"):
        # m- == "modified for geobench"
        partition = fraction_to_partition(label_fraction)
        return GeobenchDataset(
            geobench_dir=paths.GEOBENCH_DIR,
            dataset=eval_dataset,
            split=split,
            partition=partition,
            norm_stats_from_pretrained=norm_stats_from_pretrained,
            norm_method=norm_method,
        )
    elif eval_dataset == "mados":
        partition = fraction_to_partition(label_fraction)
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
        partition = fraction_to_partition(label_fraction)
        return Sen1Floods11Dataset(
            path_to_splits=paths.FLOODS_DIR,
            split=split,
            partition=partition,
            norm_stats_from_pretrained=norm_stats_from_pretrained,
            norm_method=norm_method,
        )
    elif eval_dataset.startswith("pastis"):
        partition = fraction_to_partition(label_fraction)
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
        partition = fraction_to_partition(label_fraction)
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
            label_fraction=label_fraction,
        )

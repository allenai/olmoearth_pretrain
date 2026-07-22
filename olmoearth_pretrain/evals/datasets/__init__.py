"""OlmoEarth Pretrain eval datasets."""

import logging
from typing import Any

from torch.utils.data import Dataset

import olmoearth_pretrain.evals.datasets.paths as paths
from olmoearth_pretrain.evals.studio_ingest.registry import get_dataset_entry

from .breizhcrops import BreizhCropsDataset
from .fifty_cities_dataset import FiftyCitiesDataset
from .floods_dataset import Sen1Floods11Dataset
from .geobench_dataset import GeobenchDataset
from .geobench_v2_dataset import GeobenchV2Dataset
from .mados_dataset import MADOSDataset
from .normalize import NormMethod
from .pastis_dataset import PASTISRDataset
from .pretrain_subset import PretrainSubsetDataset
from .rslearn_dataset import from_registry_entry

logger = logging.getLogger(__name__)


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
    elif eval_dataset.startswith("gb2-"):
        return GeobenchV2Dataset(
            dataset=eval_dataset,
            split=split,
            partition=kwargs.get("partition", ""),
            norm_stats_from_pretrained=norm_stats_from_pretrained,
            norm_method=norm_method,
        )
    elif eval_dataset.startswith("m-"):
        # m- == "modified for geobench"
        return GeobenchDataset(
            geobench_dir=paths.GEOBENCH_DIR,
            dataset=eval_dataset,
            split=split,
            label_fraction=label_fraction,
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
            label_fraction=label_fraction,
            norm_stats_from_pretrained=norm_stats_from_pretrained,
            norm_method=norm_method,
        )
    elif eval_dataset == "sen1floods11":
        return Sen1Floods11Dataset(
            path_to_splits=paths.FLOODS_DIR,
            split=split,
            label_fraction=label_fraction,
            norm_stats_from_pretrained=norm_stats_from_pretrained,
            norm_method=norm_method,
        )
    elif eval_dataset in ("pastis", "pastis128"):
        pastis_kwargs = {
            "split": split,
            "label_fraction": label_fraction,
            "norm_stats_from_pretrained": norm_stats_from_pretrained,
            "input_modalities": input_modalities,
            "norm_method": norm_method,
            "dir_partition": paths.PASTIS_DIR_PARTITION,
            "window_size": kwargs.get("window_size"),
        }
        if "128" in eval_dataset:
            # "pastis128"
            pastis_kwargs["path_to_splits"] = paths.PASTIS_DIR_ORIG
        else:
            pastis_kwargs["path_to_splits"] = paths.PASTIS_DIR
        return PASTISRDataset(**pastis_kwargs)  # type: ignore
    elif eval_dataset.startswith("fifty_cities"):
        # Split mode is encoded in the dataset-name suffix; "fifty_cities" alone
        # is the random split.
        if eval_dataset.endswith("by_continent"):
            split_mode = "by_continent"
        elif eval_dataset.endswith("by_city"):
            split_mode = "by_city"
        else:
            split_mode = "random"
        return FiftyCitiesDataset(
            path_to_splits=paths.FIFTY_CITIES_DIR,
            split=split,
            split_mode=split_mode,
            input_modalities=input_modalities,
            norm_stats_from_pretrained=norm_stats_from_pretrained,
            norm_method=norm_method,
            label_fraction=label_fraction,
        )
    elif eval_dataset == "breizhcrops":
        return BreizhCropsDataset(
            path_to_splits=paths.BREIZHCROPS_DIR,
            split=split,
            label_fraction=label_fraction,
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
            window_size=kwargs.get("window_size"),
            label_at_center_pixel=kwargs.get("label_at_center_pixel", False),
            tile_samples=kwargs.get("tile_samples", False),
        )

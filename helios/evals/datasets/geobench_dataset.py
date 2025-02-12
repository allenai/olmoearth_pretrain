"""GeoBench datasets, returning data in the Helios format."""

from collections.abc import Sequence
from pathlib import Path
from types import MethodType
from typing import NamedTuple

import geobench
import numpy as np
import torch.multiprocessing
from einops import repeat
from geobench.dataset import Stats
from torch.utils.data import Dataset, default_collate

from helios.data.dataset import S2_BANDS, HeliosSample

torch.multiprocessing.set_sharing_strategy("file_system")


GEOBENCH_S2_BAND_NAMES = [
    "01 - Coastal aerosol",
    "02 - Blue",
    "03 - Green",
    "04 - Red",
    "05 - Vegetation Red Edge",
    "06 - Vegetation Red Edge",
    "07 - Vegetation Red Edge",
    "08 - NIR",
    "08A - Vegetation Red Edge",
    "09 - Water vapour",
    "10 - SWIR - Cirrus",
    "11 - SWIR",
    "12 - SWIR",
]


def _geobench_band_index_from_helios_name(helios_name: str) -> int:
    for idx, band_name in enumerate(GEOBENCH_S2_BAND_NAMES):
        if helios_name.endswith(band_name.split(" ")[0][-2:]):
            return idx
    raise ValueError(f"Unmatched band name {helios_name}")


GEOBENCH_TO_HELIOS_S2_BANDS = [
    _geobench_band_index_from_helios_name(b) for b in S2_BANDS
]


class GeoBenchConfig(NamedTuple):
    """GeoBench configs."""

    benchmark_name: str
    imputes: list[tuple[str, str]]
    num_classes: int
    is_multilabel: bool


DATASET_TO_CONFIG = {
    "m-eurosat": GeoBenchConfig(
        benchmark_name="classification_v1.0",
        imputes=[],
        num_classes=10,
        is_multilabel=False,
    )
}


class GeobenchDataset(Dataset):
    """GeoBench dataset, returning data in the Helios format."""

    default_day_month_year = [1, 6, 2020]

    def __init__(
        self,
        geobench_dir: Path,
        dataset: str,
        split: str,
        partition: str,
        norm_method: str = "norm_no_clip",
    ):
        """Init GeoBench dataset."""
        config = DATASET_TO_CONFIG[dataset]
        self.config = config
        self.num_classes = config.num_classes
        self.is_multilabel = config.is_multilabel

        if split not in ["train", "valid", "test"]:
            raise ValueError(
                f"Excected split to be in ['train', 'valid', 'test'], got {split}"
            )
        assert split in ["train", "valid", "test"]

        self.split = split
        self.partition = partition

        for task in geobench.task_iterator(
            benchmark_name=config.benchmark_name,
            benchmark_dir=geobench_dir / config.benchmark_name,
        ):
            if task.dataset_name == dataset:
                break

        # hack: https://github.com/ServiceNow/geo-bench/issues/22
        task.get_dataset_dir = MethodType(
            lambda self: geobench_dir / config.benchmark_name / self.dataset_name,
            task,
        )

        self.dataset = task.get_dataset(split=self.split, partition_name=self.partition)
        original_band_names = [
            self.dataset[0].bands[i].band_info.name
            for i in range(len(self.dataset[0].bands))
        ]
        self.band_names = [x.name for x in task.bands_info]
        self.band_indices = [
            original_band_names.index(band_name) for band_name in self.band_names
        ]
        imputed_band_info = self._impute_normalization_stats(
            task.band_stats, config.imputes
        )
        self.mean, self.std = self._get_norm_stats(imputed_band_info)
        self.active_indices = range(int(len(self.dataset)))
        self.norm_method = norm_method

    @staticmethod
    def _get_norm_stats(
        imputed_band_info: dict[str, Stats],
    ) -> tuple[np.ndarray, np.ndarray]:
        means = []
        stds = []
        for band_name in GEOBENCH_S2_BAND_NAMES:
            assert band_name in imputed_band_info, f"{band_name} not found in band_info"
            means.append(imputed_band_info[band_name].mean)  # type: ignore
            stds.append(imputed_band_info[band_name].std)  # type: ignore
        return np.array(means), np.array(stds)

    @staticmethod
    def _impute_normalization_stats(
        band_info: dict, imputes: list[tuple[str, str]]
    ) -> dict:
        # band_info is a dictionary with band names as keys and statistics (mean / std) as values
        if not imputes:
            return band_info

        names_list = list(band_info.keys())
        new_band_info: dict = {}
        for band_name in GEOBENCH_S2_BAND_NAMES:
            new_band_info[band_name] = {}
            if band_name in names_list:
                # we have the band, so use it
                new_band_info[band_name] = band_info[band_name]
            else:
                # we don't have the band, so impute it
                for impute in imputes:
                    src, tgt = impute
                    if tgt == band_name:
                        # we have a match!
                        new_band_info[band_name] = band_info[src]
                        break

        return new_band_info

    @staticmethod
    def _impute_bands(
        image_list: list[np.ndarray],
        names_list: list[str],
        imputes: list[tuple[str, str]],
    ) -> list:
        # image_list should be one np.array per band, stored in a list
        # image_list and names_list should be ordered consistently!
        if not imputes:
            return image_list

        # create a new image list by looping through and imputing where necessary
        new_image_list = []
        for band_name in GEOBENCH_S2_BAND_NAMES:
            if band_name in names_list:
                # we have the band, so append it
                band_idx = names_list.index(band_name)
                new_image_list.append(image_list[band_idx])
            else:
                # we don't have the band, so impute it
                for impute in imputes:
                    src, tgt = impute
                    if tgt == band_name:
                        # we have a match!
                        band_idx = names_list.index(src)
                        new_image_list.append(image_list[band_idx])
                        break
        return new_image_list

    @staticmethod
    def _normalize_bands(
        image: np.ndarray, means: np.array, stds: np.array, method: str = "norm_no_clip"
    ) -> np.ndarray:
        original_dtype = image.dtype

        if method == "standardize":
            image = (image - means) / stds
        else:
            min_value = means - stds
            max_value = means + stds
            image = (image - min_value) / (max_value - min_value)

            if method == "norm_yes_clip":
                image = np.clip(image, 0, 1)
            elif method == "norm_yes_clip_int":
                # same as clipping between 0 and 1 but rounds to the nearest 1/255
                image = image * 255  # scale
                image = np.clip(image, 0, 255).astype(
                    np.uint8
                )  # convert to 8-bit integers
                image = (
                    image.astype(original_dtype) / 255
                )  # back to original_dtype between 0 and 1
            elif method == "norm_no_clip":
                pass
            else:
                raise ValueError(
                    f"norm type must be norm_yes_clip, norm_yes_clip_int, norm_no_clip, or standardize, not {method}"
                )
        return image

    def __getitem__(self, idx: int) -> tuple[HeliosSample, torch.Tensor]:
        """Return a single GeoBench data instance."""
        sample = self.dataset[idx]
        label = sample.label

        x_list = []
        for band_idx in self.band_indices:
            x_list.append(sample.bands[band_idx].data)

        x_list = self._impute_bands(x_list, self.band_names, self.config.imputes)

        x = np.stack(x_list, axis=2)  # (h, w, 13)
        assert (
            x.shape[-1] == 13
        ), f"All datasets must have 13 channels, not {x.shape[-1]}"
        if self.dataset == "m-so2sat":
            x = x * 10_000

        x = torch.tensor(
            self._normalize_bands(x, self.mean, self.std, self.norm_method)
        )

        # check if label is an object or a number
        if not (isinstance(label, int) or isinstance(label, list)):
            label = label.data
            # label is a memoryview object, convert it to a list, and then to a numpy array
            label = np.array(list(label))

        target = torch.tensor(label, dtype=torch.long)
        s2 = repeat(x, "h w c -> h w t c", t=1)[
            :,
            :,
            :,
            GEOBENCH_TO_HELIOS_S2_BANDS,
        ]
        timestamp = repeat(torch.tensor(self.default_day_month_year), "d -> t d", t=1)
        return HeliosSample(s2=s2.float(), timestamps=timestamp.long()), target

    def __len__(self) -> int:
        """Length of dataset."""
        return len(self.dataset)

    @staticmethod
    def collate_fn(
        batch: Sequence[tuple[HeliosSample, torch.Tensor]],
    ) -> tuple[HeliosSample, torch.Tensor]:
        """Collate function for DataLoaders."""
        samples, targets = zip(*batch)
        # we assume that the same values are consistently None
        collated_sample = default_collate(
            [s.as_dict(ignore_nones=True) for s in samples]
        )
        collated_target = default_collate([t for t in targets])
        return HeliosSample(**collated_sample), collated_target

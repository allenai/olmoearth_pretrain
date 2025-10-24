"""GeoBench datasets, returning data in the OlmoEarth Pretrain format."""

import logging
import os
from pathlib import Path
from types import MethodType

import matplotlib.pyplot as plt
import numpy as np
import torch.multiprocessing
from einops import repeat
from geobench.dataset import Stats
from geobench.task import load_task_specs
from torch.utils.data import Dataset

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataset import OlmoEarthSample
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

from .configs import dataset_to_config
from .constants import (
    EVAL_L8_BAND_NAMES,
    EVAL_S2_BAND_NAMES,
    EVAL_TO_HELIOS_L8_BANDS,
    EVAL_TO_HELIOS_S2_BANDS,
)
from .normalize import impute_normalization_stats, normalize_bands

torch.multiprocessing.set_sharing_strategy("file_system")

logger = logging.getLogger(__name__)


def _landsathelios2geobench_name(band_name: str) -> str:
    """Transform OlmoEarth Pretrain Landsat band name to Geobench Landsat band name."""
    # transforms are documented here:
    # https://github.com/ServiceNow/geo-bench/blob/main/geobench/dataset.py#L350
    transform = {
        "B1": "01 - Coastal aerosol",
        "B2": "02 - Blue",
        "B3": "03 - Green",
        "B4": "04 - Red",
        "B5": "05 - NIR",
        "B6": "06 - SWIR1",
        "B7": "07 - SWIR2",
        # B8 is the panchromatic band. Geobech does not include it.
        # its wavelength most overlaps with B3, so this is what we
        # will treat as B8 from the GeoBench dataset.
        "B8": "03 - Green",
        "B9": "09 - Cirrus",
        "B10": "10 - Tirs1",
        "B11": "10 - Tirs1",
    }
    return transform[band_name]


GEOBENCH_L8_BAND_NAMES = [_landsathelios2geobench_name(b) for b in EVAL_L8_BAND_NAMES]


class GeobenchDataset(Dataset):
    """GeoBench dataset, returning data in the OlmoEarth Pretrain format."""

    default_day_month_year = [1, 6, 2020]

    def __init__(
        self,
        geobench_dir: Path,
        dataset: str,
        split: str,
        partition: str,
        norm_stats_from_pretrained: bool = False,
        norm_method: str = "norm_no_clip",
        visualize_samples: bool = False,
    ):
        """Init GeoBench dataset.

        Args:
            geobench_dir: Path to the GeoBench directory
            dataset: Dataset name
            split: Split to use
            partition: Partition to use
            norm_stats_from_pretrained: Whether to use normalization stats from pretrained model
            norm_method: Normalization method to use, only when norm_stats_from_pretrained is False
            visualize_samples: Whether to visualize samples
        """
        config = dataset_to_config(dataset)
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
        self.norm_stats_from_pretrained = norm_stats_from_pretrained
        # If normalize with pretrained stats, we initialize the normalizer here
        if self.norm_stats_from_pretrained:
            from olmoearth_pretrain.data.normalize import Normalizer, Strategy

            self.normalizer_computed = Normalizer(Strategy.COMPUTED)
        # GEOBENCH cannot handle remote upath objects
        dataset_dir = geobench_dir / f"{config.task_type.value}_v1.0" / dataset
        task = load_task_specs(dataset_dir)  # Note: Cannot handle remote paths
        # hack: https://github.com/ServiceNow/geo-bench/issues/22
        task.get_dataset_dir = MethodType(
            lambda self: geobench_dir / f"{config.task_type.value}_v1.0" / dataset,
            task,
        )
        self.is_landsat = task.bands_info[0].__class__.__name__ == "Landsat8"

        self.dataset = task.get_dataset(split=self.split, partition_name=self.partition)

        original_band_names = [
            self.dataset[0].bands[i].band_info.name
            for i in range(len(self.dataset[0].bands))
        ]
        self.band_names = [x.name for x in task.bands_info]
        self.band_indices = [
            original_band_names.index(band_name) for band_name in self.band_names
        ]

        # this is only necessary for landsat
        self.original_band_indices_after_imputation: list[int] = []
        if self.is_landsat:
            band_order_in_geobench_names = [
                _landsathelios2geobench_name(b) for b in Modality.LANDSAT.band_order
            ]
            self.original_band_indices_after_imputation = [
                band_order_in_geobench_names.index(b) for b in original_band_names
            ]

        imputed_band_info = impute_normalization_stats(
            task.band_stats,
            config.imputes,
            all_bands=GEOBENCH_L8_BAND_NAMES if self.is_landsat else EVAL_S2_BAND_NAMES,
        )
        self.mean, self.std, self.min, self.max = self._get_norm_stats(
            imputed_band_info,
            all_bands=GEOBENCH_L8_BAND_NAMES if self.is_landsat else EVAL_S2_BAND_NAMES,
        )
        self.active_indices = range(int(len(self.dataset)))
        self.norm_method = norm_method
        self.visualize_samples = visualize_samples

        self.multiply_by_10_000 = False
        if dataset == "m-so2sat":
            logging.info(f"self.multiply_by_10_000 set to True for {dataset}")
            self.multiply_by_10_000 = True
        if self.multiply_by_10_000:
            self.mean = self.mean * 10_000
            self.std = self.std * 10_000
            self.min = self.min * 10_000
            self.max = self.max * 10_000

    @staticmethod
    def _get_norm_stats(
        imputed_band_info: dict[str, Stats],
        all_bands: list[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        means = []
        stds = []
        mins = []
        maxs = []
        for band_name in all_bands:
            assert band_name in imputed_band_info, f"{band_name} not found in band_info"
            stats = imputed_band_info[band_name]
            means.append(stats.mean)  # type: ignore
            stds.append(stats.std)  # type: ignore
            mins.append(stats.min)  # type: ignore
            maxs.append(stats.max)  # type: ignore
        return np.array(means), np.array(stds), np.array(mins), np.array(maxs)

    @staticmethod
    def _impute_bands(
        image_list: list[np.ndarray],
        names_list: list[str],
        imputes: list[tuple[str, str]],
        all_bands: list[str],
    ) -> list:
        # image_list should be one np.array per band, stored in a list
        # image_list and names_list should be ordered consistently!
        if not imputes:
            return image_list

        # create a new image list by looping through and imputing where necessary
        new_image_list = []
        for band_name in all_bands:
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

    def __getitem__(self, idx: int) -> tuple[MaskedOlmoEarthSample, torch.Tensor]:
        """Return a single GeoBench data instance."""
        sample = self.dataset[idx]
        label = sample.label

        x_list = [sample.bands[band_idx].data for band_idx in self.band_indices]

        x_list = self._impute_bands(
            x_list,
            self.band_names,
            self.config.imputes,
            all_bands=GEOBENCH_L8_BAND_NAMES if self.is_landsat else EVAL_S2_BAND_NAMES,
        )

        x = np.stack(x_list, axis=2)  # (h, w, 13)
        if self.visualize_samples:
            self.visualize_sample_bands(x, f"./visualizations/sample_{idx}")
        if self.is_landsat:
            assert x.shape[-1] == len(EVAL_L8_BAND_NAMES), (
                f"Instances must have {len(EVAL_L8_BAND_NAMES)} channels, not {x.shape[-1]}"
            )
        else:
            assert x.shape[-1] == len(EVAL_S2_BAND_NAMES), (
                f"Instances must have {len(EVAL_S2_BAND_NAMES)} channels, not {x.shape[-1]}"
            )
        if self.multiply_by_10_000:
            x = x * 10_000
        # Normalize using the downstream task's normalization stats
        if not self.norm_stats_from_pretrained:
            # log the shape of x
            # logger.info(f"x shape: {x.shape}")
            # keep a running min and max per channel in self.min_val and self.max_val
            x = torch.tensor(
                normalize_bands(
                    x, self.mean, self.std, self.min, self.max, self.norm_method
                )
            )
        # check if label is an object or a number
        if not (isinstance(label, int) or isinstance(label, list)):
            label = label.data
            # label is a memoryview object, convert it to a list, and then to a numpy array
            label = np.array(list(label))

        target = torch.tensor(label, dtype=torch.long)

        sample_dict = {}
        if self.is_landsat:
            landsat = repeat(x, "h w c -> h w t c", t=1)[
                :,
                :,
                :,
                EVAL_TO_HELIOS_L8_BANDS,
            ]
            # Normalize using the pretrained dataset's normalization stats
            if self.norm_stats_from_pretrained:
                landsat = self.normalizer_computed.normalize(Modality.LANDSAT, landsat)
                # For Landsat (ForestNet), only 5/11 bands are present, and the rest
                # are imputed. this means some of the means and standard deviations
                # from the pretrained stats are very different. To handle this, we
                # redo the imputation
                landsat_pre_imputation = [
                    landsat[:, :, :, idx]
                    for idx in self.original_band_indices_after_imputation
                ]
                landsat = np.stack(
                    self._impute_bands(
                        landsat_pre_imputation,
                        self.band_names,
                        self.config.imputes,
                        all_bands=GEOBENCH_L8_BAND_NAMES,
                    ),
                    axis=-1,
                )[:, :, :, EVAL_TO_HELIOS_L8_BANDS]

            sample_dict["landsat"] = torch.tensor(landsat).float()

        else:
            s2 = repeat(x, "h w c -> h w t c", t=1)[
                :,
                :,
                :,
                EVAL_TO_HELIOS_S2_BANDS,
            ]
            # Normalize using the pretrained dataset's normalization stats
            if self.norm_stats_from_pretrained:
                s2 = self.normalizer_computed.normalize(Modality.SENTINEL2_L2A, s2)
            sample_dict["sentinel2_l2a"] = torch.tensor(s2).float()

        timestamp = repeat(torch.tensor(self.default_day_month_year), "d -> t d", t=1)
        masked_sample = MaskedOlmoEarthSample.from_olmoearthsample(
            OlmoEarthSample(**sample_dict, timestamps=timestamp.long())
        )
        return masked_sample, target

    def __len__(self) -> int:
        """Length of dataset."""
        return len(self.dataset)

    def visualize_sample_bands(self, x: np.ndarray, output_dir: str) -> None:
        """Visualize each band from a given array, saving each plot as a PNG file in the specified output_dir.

        Args:
            x (np.ndarray): Array of shape (H, W, #bands).
            output_dir (str): Directory path where plots will be saved.
        """
        # Ensure the directory exists; if not, create it.
        os.makedirs(output_dir, exist_ok=True)

        # For each band in x
        for band_idx in range(x.shape[-1]):
            # Take the band slice
            band_data = x[..., band_idx]
            band_title = (
                self.band_names[band_idx]
                if band_idx < len(self.band_names)
                else f"Band_{band_idx}"
            )

            # Create figure & axis
            fig, ax = plt.subplots(figsize=(6, 4))
            im = ax.imshow(band_data, cmap="gray")
            ax.set_title(band_title)

            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Pixel Value")

            # Create target filename
            filename = f"{band_title.replace(' ', '_').replace('/', '_')}.png"
            save_path = os.path.join(output_dir, filename)

            # Save and close
            fig.savefig(save_path, bbox_inches="tight")
            plt.close(fig)

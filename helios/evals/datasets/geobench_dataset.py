"""GeoBench datasets, returning data in the Helios format."""

import logging
import os
from pathlib import Path
from types import MethodType

import geobench
import matplotlib.pyplot as plt
import numpy as np
import torch.multiprocessing
from einops import repeat
from geobench.dataset import Stats
from torch.utils.data import Dataset
from upath import UPath

from helios.data.constants import Modality
from helios.data.dataset import HeliosSample
from helios.train.masking import MaskedHeliosSample

from .configs import dataset_to_config
from .constants import EVAL_S2_BAND_NAMES, EVAL_TO_HELIOS_S2_BANDS
from .normalize import impute_normalization_stats, normalize_bands

torch.multiprocessing.set_sharing_strategy("file_system")

logger = logging.getLogger(__name__)

GEOBENCH_DIR = UPath("/weka/dfive-default/presto-geobench/dataset/geobench")


class GeobenchDataset(Dataset):
    """GeoBench dataset, returning data in the Helios format."""

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
            from helios.data.normalize import Normalizer, Strategy

            self.normalizer_computed = Normalizer(Strategy.COMPUTED)

        for task in geobench.task_iterator(
            # e.g. "classification_v1.0"
            benchmark_name=f"{config.task_type.value}_v1.0",
            benchmark_dir=geobench_dir / f"{config.task_type.value}_v1.0",
        ):
            if task.dataset_name == dataset:
                break

        # hack: https://github.com/ServiceNow/geo-bench/issues/22
        task.get_dataset_dir = MethodType(
            lambda self: geobench_dir / f"{config.task_type.value}_v1.0" / dataset,
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
        imputed_band_info = impute_normalization_stats(task.band_stats, config.imputes)
        self.mean, self.std = self._get_norm_stats(imputed_band_info)
        self.active_indices = range(int(len(self.dataset)))
        self.norm_method = norm_method
        self.visualize_samples = visualize_samples

        self.multiply_by_10_000 = False
        if dataset == "m-so2sat":
            logging.info(f"self.multiply_by_10_000 set to True for {dataset}")
            self.multiply_by_10_000 = True

    @staticmethod
    def _get_norm_stats(
        imputed_band_info: dict[str, Stats],
    ) -> tuple[np.ndarray, np.ndarray]:
        means = []
        stds = []
        for band_name in EVAL_S2_BAND_NAMES:
            assert band_name in imputed_band_info, f"{band_name} not found in band_info"
            means.append(imputed_band_info[band_name].mean)  # type: ignore
            stds.append(imputed_band_info[band_name].std)  # type: ignore
        return np.array(means), np.array(stds)

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
        for band_name in EVAL_S2_BAND_NAMES:
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

    def __getitem__(self, idx: int) -> tuple[MaskedHeliosSample, torch.Tensor]:
        """Return a single GeoBench data instance."""
        sample = self.dataset[idx]
        label = sample.label

        x_list = [sample.bands[band_idx].data for band_idx in self.band_indices]

        x_list = self._impute_bands(x_list, self.band_names, self.config.imputes)

        x = np.stack(x_list, axis=2)  # (h, w, 13)
        if self.visualize_samples:
            self.visualize_sample_bands(x, f"./visualizations/sample_{idx}")
        assert (
            x.shape[-1] == 13
        ), f"All datasets must have 13 channels, not {x.shape[-1]}"
        if self.multiply_by_10_000:
            x = x * 10_000

        # Normalize using the downstream task's normalization stats
        if not self.norm_stats_from_pretrained:
            x = torch.tensor(normalize_bands(x, self.mean, self.std, self.norm_method))
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
            EVAL_TO_HELIOS_S2_BANDS,
        ]
        # Normalize using the pretrained dataset's normalization stats
        if self.norm_stats_from_pretrained:
            s2 = torch.tensor(
                self.normalizer_computed.normalize(Modality.SENTINEL2_L2A, s2)
            )

        timestamp = repeat(torch.tensor(self.default_day_month_year), "d -> t d", t=1)
        masked_sample = MaskedHeliosSample.from_heliossample(
            HeliosSample(sentinel2_l2a=s2.float(), timestamps=timestamp.long())
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

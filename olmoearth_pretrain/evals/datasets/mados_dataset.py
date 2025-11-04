"""MADOS dataset class."""

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing
import torch.nn.functional as F
from einops import repeat
from PIL import Image
from torch.utils.data import Dataset

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataset import OlmoEarthSample
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

from .constants import EVAL_S2_BAND_NAMES, EVAL_TO_OLMOEARTH_S2_BANDS
from .normalize import normalize_bands
from .utils import load_min_max_stats

torch.multiprocessing.set_sharing_strategy("file_system")


BAND_STATS = {
    "01 - Coastal aerosol": {
        "mean": 0.05834943428635597,
        "std": 0.021028317511081696,
    },
    "02 - Blue": {"mean": 0.05259967967867851, "std": 0.024127380922436714},
    "03 - Green": {"mean": 0.044100478291511536, "std": 0.027077781036496162},
    "04 - Red": {"mean": 0.036041468381881714, "std": 0.028892582282423973},
    "05 - Vegetation Red Edge": {
        "mean": 0.03370574489235878,
        "std": 0.028282219544053078,
    },
    "06 - Vegetation Red Edge": {
        "mean": 0.034933559596538544,
        "std": 0.03273169696331024,
    },
    "07 - Vegetation Red Edge": {
        "mean": 0.03616137430071831,
        "std": 0.04026992991566658,
    },
    "08 - NIR": {"mean": 0.03185819461941719, "std": 0.03786701336503029},
    "08A - Vegetation Red Edge": {
        "mean": 0.0348929800093174,
        "std": 0.04325847327709198,
    },
    "09 - Water vapour": {"mean": 0.0348929800093174, "std": 0.04325847327709198},
    "10 - SWIR - Cirrus": {"mean": 0.029205497354269028, "std": 0.0350610613822937},
    "11 - SWIR": {"mean": 0.023518012836575508, "std": 0.02930651605129242},
    "12 - SWIR": {"mean": 0.017942268401384354, "std": 0.022308016195893288},
}


def split_and_filter_tensors(
    image_tensor: torch.Tensor, label_tensor: torch.Tensor
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Split image and label tensors into 9 tiles and filter based on label content.

    Args:
    image_tensor (torch.Tensor): Input tensor of shape (13, 240, 240)
    label_tensor (torch.Tensor): Label tensor of shape (240, 240)

    Returns:
    list of tuples: Each tuple contains (image_tile, label_tile)
    """
    assert image_tensor.shape == (
        13,
        240,
        240,
    ), "Image tensor must be of shape (13, 240, 240)"
    assert label_tensor.shape == (240, 240), "Label tensor must be of shape (240, 240)"

    tile_size = 80
    tiles = []
    labels = []

    for i in range(3):
        for j in range(3):
            # Extract image tile
            image_tile = image_tensor[
                :,
                i * tile_size : (i + 1) * tile_size,
                j * tile_size : (j + 1) * tile_size,
            ]

            # Extract corresponding label tile
            label_tile = label_tensor[
                i * tile_size : (i + 1) * tile_size, j * tile_size : (j + 1) * tile_size
            ]

            # Check if label tile has any non-zero values
            if torch.any(label_tile > 0):
                tiles.append(image_tile)
                labels.append(label_tile)

    return tiles, labels


class PrepMADOSDataset(Dataset):
    """Prep the MADOS dataset into .pt objects."""

    def __init__(self, root_dir: str, split_file: str) -> None:
        """Init PrepMADOSDataset class."""
        self.root_dir = root_dir

        with open(os.path.join(root_dir, "splits", split_file)) as f:
            self.scene_list = [line.strip() for line in f]

    def __len__(self) -> int:
        """The number of scenes being processed."""
        return len(self.scene_list)

    def __getitem__(self, idx: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Retrieve one scene and split it up."""
        scene_name = self.scene_list[idx]
        scene_num_1 = scene_name.split("_")[1]
        scene_num_2 = scene_name.split("_")[2]

        # Load all bands
        B1 = self._load_band(scene_num_1, scene_num_2, [442, 443], 60)
        B2 = self._load_band(scene_num_1, scene_num_2, [492], 10)
        B3 = self._load_band(scene_num_1, scene_num_2, [559, 560], 10)
        B4 = self._load_band(scene_num_1, scene_num_2, [665], 10)
        B5 = self._load_band(scene_num_1, scene_num_2, [704], 20)
        B7 = self._load_band(scene_num_1, scene_num_2, [780, 783], 20)
        B8 = self._load_band(scene_num_1, scene_num_2, [833], 10)
        B8A = self._load_band(scene_num_1, scene_num_2, [864, 865], 20)
        B11 = self._load_band(scene_num_1, scene_num_2, [1610, 1614], 20)
        B12 = self._load_band(scene_num_1, scene_num_2, [2186, 2202], 20)

        B1 = self._resize(B1)
        B5 = self._resize(B5)
        B7 = self._resize(B7)
        B8A = self._resize(B8A)
        B11 = self._resize(B11)
        B12 = self._resize(B12)

        # Interpolate missing bands
        B6 = (B5 + B7) / 2
        B9 = B8A
        B10 = (B8A + B11) / 2

        image = torch.cat(
            [B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12], axis=1
        ).squeeze(0)  # (13, 240, 240)
        mask = (
            self._load_mask(scene_num_1, scene_num_2).squeeze(0).squeeze(0)
        )  # (240, 240)
        images, masks = split_and_filter_tensors(image, mask)

        return images, masks

    def _load_band(
        self, scene_num_1: str, scene_num_2: str, bands: list[int], resolution: int
    ) -> torch.Tensor:
        for band in bands:
            band_path = f"{self.root_dir}/Scene_{scene_num_1}/{resolution}/Scene_{scene_num_1}_L2R_rhorc_{band}_{scene_num_2}.tif"
            if os.path.exists(band_path):
                return (
                    torch.from_numpy(np.array(Image.open(band_path)))
                    .float()
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
        print(f"COULDNT FIND {scene_num_1, scene_num_2, bands, resolution}")

    def _resize(self, image: torch.Tensor) -> torch.Tensor:
        return F.interpolate(image, size=240, mode="bilinear", align_corners=False)

    def _load_mask(self, scene_num_1: str, scene_num_2: str) -> torch.Tensor:
        mask_path = f"{self.root_dir}/Scene_{scene_num_1}/10/Scene_{scene_num_1}_L2R_cl_{scene_num_2}.tif"
        return (
            torch.from_numpy(np.array(Image.open(mask_path)))
            .long()
            .unsqueeze(0)
            .unsqueeze(0)
        )


def process_mados(
    save_path: str, root_dir: str = "MADOS", split_file: str = "test_X.txt"
) -> None:
    """Process the MADOS dataset."""
    dataset = PrepMADOSDataset(root_dir=root_dir, split_file=split_file)
    all_images = []
    all_masks = []
    for i in dataset:
        all_images += i[0]
        all_masks += i[1]

    split_images = torch.stack(all_images)  # shape (N, 13, 80, 80)
    split_masks = torch.stack(all_masks)  # shape (N, 80, 80)
    torch.save(obj={"images": split_images, "labels": split_masks}, f=save_path)


class MADOSDataset(Dataset):
    """MADOS dataset to be used by evals."""

    default_day_month_year = [1, 6, 2020]

    def __init__(
        self,
        path_to_splits: Path,
        split: str,
        partition: str,
        norm_stats_from_pretrained: bool = False,
        norm_method: str = "norm_no_clip",
    ):
        """Init MADOS dataset.

        Args:
            path_to_splits: Path where .pt objects returned by process_mados have been saved
            split: Split to use
            partition: Partition to use
            norm_stats_from_pretrained: Whether to use normalization stats from pretrained model
            norm_method: Normalization method to use, only when norm_stats_from_pretrained is False
        """
        # NOTE: I imputed bands for this dataset before saving the tensors, so no imputation is necessary
        assert split in ["train", "val", "valid", "test"]
        if split == "valid":
            split = "val"

        self.min_max_stats = load_min_max_stats()["mados"]
        # Merge BAND_STATS and min/max stats, keeping means and stds from BAND_STATS and min/max from min_max_stats
        minmax = self.min_max_stats["sentinel2_l2a"]
        merged_band_stats = {
            band_name: {
                **(
                    {k: BAND_STATS[band_name][k] for k in ("mean", "std")}
                    if band_name in BAND_STATS
                    else {}
                ),
                **(
                    {k: minmax[band_name][k] for k in ("min", "max")}
                    if band_name in minmax
                    else {}
                ),
            }
            for band_name in EVAL_S2_BAND_NAMES
        }
        self.means, self.stds, self.mins, self.maxs = self._get_norm_stats(
            merged_band_stats
        )
        self.split = split
        self.norm_method = norm_method

        self.norm_stats_from_pretrained = norm_stats_from_pretrained
        # If normalize with pretrained stats, we initialize the normalizer here
        if self.norm_stats_from_pretrained:
            from olmoearth_pretrain.data.normalize import Normalizer, Strategy

            self.normalizer_computed = Normalizer(Strategy.COMPUTED)

        torch_obj = torch.load(path_to_splits / f"MADOS_{split}.pt")
        self.images = torch_obj["images"]
        self.labels = torch_obj["labels"]

        if (partition != "default") and (split == "train"):
            with open(path_to_splits / f"{partition}_partition.json") as json_file:
                subset_indices = json.load(json_file)

            self.images = self.images[subset_indices]
            self.labels = self.labels[subset_indices]

    @staticmethod
    def _get_norm_stats(
        imputed_band_info: dict[str, dict[str, float]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        means = []
        stds = []
        mins = []
        maxs = []
        for band_name in EVAL_S2_BAND_NAMES:
            assert band_name in imputed_band_info, f"{band_name} not found in band_info"
            means.append(imputed_band_info[band_name]["mean"])  # type: ignore
            stds.append(imputed_band_info[band_name]["std"])  # type: ignore
            mins.append(imputed_band_info[band_name]["min"])  # type: ignore
            maxs.append(imputed_band_info[band_name]["max"])  # type: ignore
        return np.array(means), np.array(stds), np.array(mins), np.array(maxs)

    def __len__(self) -> int:
        """Lenth of the dataset."""
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> tuple[MaskedOlmoEarthSample, torch.Tensor]:
        """Return a single MADOS data instance."""
        image = self.images[idx]  # (80, 80, 13)
        label = self.labels[idx]  # (80, 80)

        if not self.norm_stats_from_pretrained:
            image = normalize_bands(
                image.numpy(),
                self.means,
                self.stds,
                self.mins,
                self.maxs,
                self.norm_method,
            )
        image = repeat(image, "h w c -> h w t c", t=1)[
            :,
            :,
            :,
            EVAL_TO_OLMOEARTH_S2_BANDS,
        ]

        if self.norm_stats_from_pretrained:
            image = self.normalizer_computed.normalize(Modality.SENTINEL2_L2A, image)

        timestamp = repeat(torch.tensor(self.default_day_month_year), "d -> t d", t=1)
        masked_sample = MaskedOlmoEarthSample.from_olmoearthsample(
            OlmoEarthSample(
                sentinel2_l2a=torch.tensor(image).float(), timestamps=timestamp.long()
            )
        )
        return masked_sample, label

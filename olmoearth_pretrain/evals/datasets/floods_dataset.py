"""Floods eval dataset, based on Sen1Floods11."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray
import torch
from einops import rearrange, repeat
from torch.utils.data import Dataset
from tqdm import tqdm
from upath import UPath

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataset import OlmoEarthSample
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

from .constants import EVAL_S1_BAND_NAMES, EVAL_TO_HELIOS_S1_BANDS
from .normalize import normalize_bands
from .utils import load_min_max_stats

BAND_STATS = {
    "vv": {"mean": -11.27174944, "std": 4.81716083},
    "vh": {"mean": -18.4847947, "std": 5.79660676},
}


FLOODS_DIR = UPath("/weka/dfive-default/presto_eval_sets/floods")


class Sen1Floods11Processor:
    """Class for preprocessing floods dataset."""

    input_hw = 512
    output_tile_size = 64

    s1_bands = ("VV", "VH")
    s2_bands = (
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B9",
        "B10",
        "B11",
        "B12",
    )

    def __init__(self, folder: Path, split_path: Path):
        """Class for preprocessing floods dataset."""
        split_labelnames = pd.read_csv(split_path, header=None)[1].tolist()
        all_labels = list(folder.glob("LabelHand/*.tif"))
        split_labels = []
        for label in all_labels:
            if label.name in split_labelnames:
                split_labels.append(label)
        self.all_labels = split_labels

    def __len__(self) -> int:
        """Length of preprocessor."""
        return len(self.all_labels)

    @classmethod
    def split_and_filter_tensors(
        cls, s1: torch.Tensor, s2: torch.Tensor, labels: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Split image and label tensors into 9 tiles and filter based on label content."""
        assert s1.shape == (
            len(cls.s1_bands),
            cls.input_hw,
            cls.input_hw,
        ), (
            f"s1 tensor must be of shape ({len(cls.s1_bands)}, {cls.input_hw}, {cls.input_hw}), "
            f"got {s1.shape}"
        )
        assert s2.shape == (
            len(cls.s2_bands),
            cls.input_hw,
            cls.input_hw,
        ), (
            f"s2 tensor must be of shape ({len(cls.s2_bands)}, {cls.input_hw}, {cls.input_hw})"
        )
        assert labels.shape == (
            1,
            cls.input_hw,
            cls.input_hw,
        ), f"labels tensor must be of shape (1, {cls.input_hw}, {cls.input_hw})"

        tile_size = cls.output_tile_size
        s1_list, s2_list, labels_list = [], [], []

        num_tiles_per_dim = cls.input_hw // cls.output_tile_size
        for i in range(num_tiles_per_dim):
            for j in range(num_tiles_per_dim):
                # Extract image tile
                s1_tile = s1[
                    :,
                    i * tile_size : (i + 1) * tile_size,
                    j * tile_size : (j + 1) * tile_size,
                ]

                s2_tile = s2[
                    :,
                    i * tile_size : (i + 1) * tile_size,
                    j * tile_size : (j + 1) * tile_size,
                ]

                # Extract corresponding label tile
                label_tile = labels[
                    :,
                    i * tile_size : (i + 1) * tile_size,
                    j * tile_size : (j + 1) * tile_size,
                ]

                # Check if label tile has any non-zero values
                if torch.any(label_tile > 0):
                    s1_list.append(s1_tile)
                    s2_list.append(s2_tile)
                    labels_list.append(label_tile)

        return s1_list, s2_list, labels_list

    @staticmethod
    def _label_to(label: Path, to: str = "s1") -> Path:
        sen_root = label.parents[1]
        location, tile_id, _ = label.stem.split("_")
        if to == "s1":
            return sen_root / f"s1/{location}_{tile_id}_S1Hand.tif"
        elif to == "s2":
            return sen_root / f"s2/{location}_{tile_id}_S2Hand.tif"
        else:
            raise ValueError(f"Expected `to` to be s1 or s2, got {to}")

    def __getitem__(
        self, idx: int
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Process a single instance in the dataset."""
        labels_path = self.all_labels[idx]

        with rioxarray.open_rasterio(labels_path) as ds:  # type: ignore
            labels = torch.from_numpy(ds.values)  # type: ignore

        with rioxarray.open_rasterio(self._label_to(labels_path, "s1")) as ds:  # type: ignore
            s1 = torch.from_numpy(ds.values)  # type: ignore

        with rioxarray.open_rasterio(self._label_to(labels_path, "s2")) as ds:  # type: ignore
            s2 = torch.from_numpy(ds.values)  # type: ignore
        return self.split_and_filter_tensors(s1, s2, labels)


def get_sen1floods11(
    flood_folder: Path, split_name: str = "flood_bolivia_data.csv"
) -> None:
    """Calls Sen1Floods11Processor to preprocess the floods dataset."""
    split_path = flood_folder / split_name
    dataset = Sen1Floods11Processor(folder=flood_folder, split_path=split_path)
    all_s1, all_s2, all_labels = [], [], []
    for i in tqdm(range(len(dataset))):
        b = dataset[i]
        all_s1 += b[0]
        all_s2 += b[1]
        all_labels += b[2]

    save_path = flood_folder / f"{split_path.stem}.pt"
    torch.save(
        obj={
            "s1": torch.stack(all_s1),
            "labels": torch.stack(all_labels),
            "s2": torch.stack(all_s2),
        },
        f=save_path,
    )


class Sen1Floods11Dataset(Dataset):
    """Sen1Floods eval dataset."""

    default_day_month_year = [1, 6, 2020]

    def __init__(
        self,
        path_to_splits: Path,
        split: str,
        partition: str,
        norm_stats_from_pretrained: bool = False,
        norm_method: str = "norm_no_clip",
        mode: str = "s1",  # not sure if we would ever want s2?
    ):
        """Sen1Floods eval dataset."""
        assert split in ["train", "val", "valid", "test", "bolivia"]
        if split == "val":
            split = "valid"

        self.min_max_stats = load_min_max_stats()["sen1floods11"]
        # Merge BAND_STATS and min/max stats
        minmax = self.min_max_stats["sentinel1"]
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
            for band_name in EVAL_S1_BAND_NAMES
        }
        self.means, self.stds, self.mins, self.maxs = self._get_norm_stats(
            merged_band_stats
        )

        self.split = split

        torch_obj = torch.load(path_to_splits / f"flood_{split}_data.pt")
        self.s1 = torch_obj["s1"]  # (N, 2, 64, 64)
        self.s1 = rearrange(self.s1, "n c h w -> n h w c")
        # print(f"Before removing nans, we have {self.s1.shape[0]} tiles")
        self.labels = torch_obj["labels"]
        self.s1, self.labels = self._remove_nan(
            self.s1, self.labels
        )  # should we remove the tile or impute the pixel?
        # print(f"After removing nans, we have {self.s1.shape[0]} tiles")

        if (partition != "default") and (split == "train"):
            with open(path_to_splits / f"{partition}_partition.json") as json_file:
                subset_indices = json.load(json_file)

            self.s1 = self.s1[subset_indices]
            self.labels = self.labels[subset_indices]

        self.norm_stats_from_pretrained = norm_stats_from_pretrained
        self.norm_method = norm_method
        # If normalize with pretrained stats, we initialize the normalizer here
        if self.norm_stats_from_pretrained:
            from olmoearth_pretrain.data.normalize import Normalizer, Strategy

            self.normalizer_computed = Normalizer(Strategy.COMPUTED)

        if mode != "s1":
            raise ValueError(f"Modes other than s1 not yet supported, got {mode}")

    @staticmethod
    def _remove_nan(
        s1: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # s1 is shape (N, H, W, C)
        # target is shape (N, H, W)

        new_s1, new_target = [], []
        for i in range(s1.shape[0]):
            if torch.any(torch.isnan(s1[i])) or torch.any(torch.isinf(s1[i])):
                continue
            new_s1.append(s1[i])
            new_target.append(target[i])

        return torch.stack(new_s1), torch.stack(new_target)

    @staticmethod
    def _get_norm_stats(
        imputed_band_info: dict[str, dict[str, float]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        means = []
        stds = []
        mins = []
        maxs = []
        for band_name in EVAL_S1_BAND_NAMES:
            assert band_name in imputed_band_info, f"{band_name} not found in band_info"
            means.append(imputed_band_info[band_name]["mean"])  # type: ignore
            stds.append(imputed_band_info[band_name]["std"])  # type: ignore
            mins.append(imputed_band_info[band_name]["min"])  # type: ignore
            maxs.append(imputed_band_info[band_name]["max"])  # type: ignore
        return np.array(means), np.array(stds), np.array(mins), np.array(maxs)

    def __len__(self) -> int:
        """Length of eval set."""
        return self.s1.shape[0]

    def __getitem__(self, idx: int) -> tuple[MaskedOlmoEarthSample, torch.Tensor]:
        """Return an instance of the sen1floods11 eval set."""
        image = self.s1[idx]  # (64, 64, 2)
        label = self.labels[idx][0]  # (64, 64)

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
            EVAL_TO_HELIOS_S1_BANDS,
        ]
        if self.norm_stats_from_pretrained:
            image = self.normalizer_computed.normalize(Modality.SENTINEL1, image)

        timestamp = repeat(torch.tensor(self.default_day_month_year), "d -> t d", t=1)
        masked_sample = MaskedOlmoEarthSample.from_olmoearthsample(
            OlmoEarthSample(
                sentinel1=torch.tensor(image).float(), timestamps=timestamp.long()
            )
        )
        return masked_sample, label.long()

"""Breizhcrops eval dataset."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from breizhcrops import BreizhCrops
from breizhcrops.datasets.breizhcrops import SELECTED_BANDS
from einops import repeat
from torch.utils.data import ConcatDataset, Dataset
from upath import UPath

from helios.train.masking import HeliosSample, MaskedHeliosSample, Modality

from .constants import EVAL_S2_BAND_NAMES, EVAL_TO_HELIOS_S2_BANDS
from .normalize import normalize_bands

LEVEL = "L1C"


def _helios2bc_name(band_name: str) -> str:
    """Transform Helios S2 band name to Breizhcrops S2 band name."""
    band_number = band_name.split(" ")[0]
    if band_number.startswith("0"):
        band_number = band_number[1:]
    return f"B{band_number}"


INPUT_TO_OUTPUT_BAND_MAPPING = [
    SELECTED_BANDS[LEVEL].index(_helios2bc_name(b)) for b in EVAL_S2_BAND_NAMES
]

BAND_STATS = {
    "01 - Coastal aerosol": {"mean": 3254.1433, "std": 2148.5647},
    "02 - Blue": {"mean": 288.4604, "std": 544.2625},
    "03 - Green": {"mean": 2729.1228, "std": 1146.0743},
    "04 - Red": {"mean": 1857.3398, "std": 985.2388},
    "05 - Vegetation Red Edge": {"mean": 2999.3413, "std": 2194.9316},
    "06 - Vegetation Red Edge": {"mean": 2742.9236, "std": 2055.1450},
    "07 - Vegetation Red Edge": {"mean": 2749.7593, "std": 2285.5239},
    "08 - NIR": {"mean": 2992.1721, "std": 2134.8782},
    "08A - Vegetation Red Edge": {"mean": 3702.4248, "std": 1794.7379},
    "09 - Water vapour": {"mean": 4056.3201, "std": 1752.6676},
    "10 - SWIR - Cirrus": {"mean": 3914.2307, "std": 1649.3500},
    "11 - SWIR": {"mean": 4290.2134, "std": 11693.7297},
    "12 - SWIR": {"mean": 1697.6628, "std": 1239.9095},
}

BREIZHCROPS_DIR = UPath("/weka/dfive-default/skylight/presto_eval_sets/breizhcrops")


class BreizhCropsDataset(Dataset):
    """The Breizhcrops dataset."""

    def __init__(
        self,
        path_to_splits: Path,
        split: str,
        partition: str,
        norm_stats_from_pretrained: bool = False,
        norm_method: str = "norm_no_clip",
        monthly_average: bool = True,
    ):
        """The Breizhcrops dataset.

        https://isprs-archives.copernicus.org/articles/XLIII-B2-2020/1545/2020/
        isprs-archives-XLIII-B2-2020-1545-2020.pdf

        We partitioned all acquired field parcels
        according to the NUTS-3 regions and suggest to subdivide the
        dataset into training (FRH01, FRH02), validation (FRH03), and
        evaluation (FRH04) subsets based on these spatially distinct
        regions.

        Args:
            path_to_splits: Path where .pt objects returned by process_mados have been saved
            split: Split to use
            partition: Partition to use
            norm_stats_from_pretrained: Whether to use normalization stats from pretrained model
            norm_method: Normalization method to use, only when norm_stats_from_pretrained is False
            monthly_average: Whether to compute a monthly average of the timesteps
        """
        kwargs = {
            "root": path_to_splits,
            "preload_ram": False,
            "level": LEVEL,
            "transform": raw_transform,
            "target_transform": default_target_transform,
        }
        # belle-ille is small, so its useful for testing
        assert split in ["train", "valid", "test", "belle-ile"]
        if split == "train":
            self.ds: Dataset = ConcatDataset(
                [BreizhCrops(region=r, **kwargs) for r in ["frh01", "frh02"]]
            )
        elif split == "valid":
            self.ds = BreizhCrops(region="frh03", **kwargs)
        elif split == "test":
            self.ds = BreizhCrops(region="frh04", **kwargs)
        else:
            self.ds = BreizhCrops(region="belle-ile", **kwargs)
        self.monthly_average = monthly_average

        self.means, self.stds = self._get_norm_stats(BAND_STATS)
        if partition != "default":
            raise NotImplementedError(f"partition {partition} not implemented yet")

        self.norm_stats_from_pretrained = norm_stats_from_pretrained
        self.norm_method = norm_method
        # If normalize with pretrained stats, we initialize the normalizer here
        if self.norm_stats_from_pretrained:
            from helios.data.normalize import Normalizer, Strategy

            self.normalizer_computed = Normalizer(Strategy.COMPUTED)

    @staticmethod
    def _get_norm_stats(
        imputed_band_info: dict[str, dict[str, float]],
    ) -> tuple[np.ndarray, np.ndarray]:
        means = []
        stds = []
        for band_name in EVAL_S2_BAND_NAMES:
            assert band_name in imputed_band_info, f"{band_name} not found in band_info"
            means.append(imputed_band_info[band_name]["mean"])  # type: ignore
            stds.append(imputed_band_info[band_name]["std"])  # type: ignore
        return np.array(means), np.array(stds)

    def __len__(self) -> int:
        """Length of the dataset."""
        return len(self.ds)

    def __getitem__(self, idx: int) -> tuple[MaskedHeliosSample, torch.Tensor]:
        """Return a Breizhcrops instance."""
        x, y_true, _ = self.ds[idx]
        if self.monthly_average:
            x = self._average_over_month(x)  # T, C
        months = torch.from_numpy(x[:, SELECTED_BANDS[LEVEL].index("doa")])
        days = torch.ones_like(months)
        # from the Breizhcrops paper: The dataset is composed of Sentinel-2 image time series
        # extracted from January 1, 2017 to December 31, 2017
        years = torch.ones_like(months) * 2017
        timestamp = torch.stack([days, months, years], dim=-1)  # t, c=3
        if not self.norm_stats_from_pretrained:
            # The first 13 bands are the S2 bands so to apply stats we first filter to those
            x = x[:, : len(BAND_STATS)]
            x = normalize_bands(x, self.means, self.stds, self.norm_method)
        image = repeat(x, "t c -> h w t c", w=1, h=1)[
            :,
            :,
            :,
            INPUT_TO_OUTPUT_BAND_MAPPING,
        ][:, :, :, EVAL_TO_HELIOS_S2_BANDS]
        if self.norm_stats_from_pretrained:
            image = self.normalizer_computed.normalize(Modality.SENTINEL2_L2A, image)

        masked_sample = MaskedHeliosSample.from_heliossample(
            HeliosSample(
                sentinel2_l2a=torch.tensor(image).float(), timestamps=timestamp.long()
            )
        )
        return masked_sample, y_true.long()

    @staticmethod
    def _average_over_month(x: np.ndarray) -> np.ndarray:
        # doa == date of acquisition
        x[:, SELECTED_BANDS[LEVEL].index("doa")] = np.array(
            [
                t.month - 1
                for t in pd.to_datetime(x[:, SELECTED_BANDS[LEVEL].index("doa")])
            ]
        )
        per_month = np.split(
            x, np.unique(x[:, SELECTED_BANDS[LEVEL].index("doa")], return_index=True)[1]
        )[1:]
        return np.array([per_month[idx].mean(axis=0) for idx in range(len(per_month))])


def raw_transform(input_timeseries: np.ndarray) -> np.ndarray:
    """A raw transform, for the Breizhcrops transforms."""
    return input_timeseries


def default_target_transform(y: np.ndarray) -> torch.Tensor:
    """The default label transform, for the Breizhcrops labels."""
    return torch.tensor(y, dtype=torch.long)

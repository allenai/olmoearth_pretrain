import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from breizhcrops import BreizhCrops
from breizhcrops.datasets.breizhcrops import SELECTED_BANDS
from einops import repeat
from torch.utils.data import ConcatDataset, Dataset

from .constants import EVAL_S2_BAND_NAMES, EVAL_TO_HELIOS_S2_BANDS
from .normalize import normalize_bands

LEVEL = "L1C"
OUTPUT_BAND_ORDER = [
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
]
INPUT_TO_OUTPUT_BAND_MAPPING = [SELECTED_BANDS[LEVEL].index(b) for b in OUTPUT_BAND_ORDER]

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
    "12 - SWIR": {"mean": 1697.6628, "std": 1239.9095}
}


class BreizhCropsDataset(Dataset):
    def __init__(
        self,
        path_to_splits: Path,
        split: str,
        norm_operation,
        augmentation,
        partition,
        monthly_average: bool = True,
    ):
        """
        https://isprs-archives.copernicus.org/articles/XLIII-B2-2020/1545/2020/
        isprs-archives-XLIII-B2-2020-1545-2020.pdf

        We partitioned all acquired field parcels
        according to the NUTS-3 regions and suggest to subdivide the
        dataset into training (FRH01, FRH02), validation (FRH03), and
        evaluation (FRH04) subsets based on these spatially distinct
        regions.
        """
        kwargs = {
            "root": path_to_splits,
            "preload_ram": False,
            "level": LEVEL,
            "transform": raw_transform,
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

        with (Path(__file__).parents[0] / Path("configs_v2") / Path("breizhcrops.json")).open(
            "r"
        ) as f:
            config = json.load(f)
        self.band_info = config["band_info"]
        self.norm_operation = norm_operation
        self.augmentation = augmentation
        warnings.warn("Augmentations ignored for time series")
        if partition != "default":
            raise NotImplementedError(f"partition {partition} not implemented yet")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y_true, _ = self.ds[idx]
        if self.monthly_average:
            x = self.average_over_month(x)
        eo = normalize_bands(
            x[:, INPUT_TO_OUTPUT_BAND_MAPPING], self.norm_operation, self.band_info
        )
        eo = repeat(eo, "t d -> h w t d", h=1, w=1)
        months = x[:, SELECTED_BANDS[LEVEL].index("doa")]
        return {"s2": torch.tensor(eo), "months": torch.tensor(months), "target": y_true}

    @staticmethod
    def average_over_month(x: np.ndarray):
        x[:, SELECTED_BANDS[LEVEL].index("doa")] = np.array(
            [t.month - 1 for t in pd.to_datetime(x[:, SELECTED_BANDS[LEVEL].index("doa")])]
        )
        per_month = np.split(
            x, np.unique(x[:, SELECTED_BANDS[LEVEL].index("doa")], return_index=True)[1]
        )[1:]
        return np.array([per_month[idx].mean(axis=0) for idx in range(len(per_month))])


def raw_transform(input_timeseries):
    return input_timeseries

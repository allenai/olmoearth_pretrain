"""CropHarvest dataset."""

import logging
from pathlib import Path

import numpy as np
import torch
from einops import repeat
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from upath import UPath

from olmoearth_pretrain.data.dataset import OlmoEarthSample
from olmoearth_pretrain.data.normalize import Normalizer, Strategy
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample, Modality

from .configs import dataset_to_config
from .constants import (
    EVAL_S1_BAND_NAMES,
    EVAL_S2_BAND_NAMES,
    EVAL_TO_HELIOS_S1_BANDS,
    EVAL_TO_HELIOS_S2_BANDS,
    EVAL_TO_HELIOS_SRTM_BANDS,
)
from .cropharvest_package.bands import BANDS
from .cropharvest_package.datasets import CropHarvest
from .cropharvest_package.utils import memoized
from .normalize import normalize_bands
from .utils import load_min_max_stats

logger = logging.getLogger("__main__")


# ```
# from olmoearth_pretrain.evals.datasets.cropharvest_package.utils import load_normalizing_dict
# d = load_normalizing_dict("normalizing_dict.h5")
# X_MEAN = d["mean"].tolist()
# X_STD = d["std"].tolist()
# ```
# these are the mean and standard deviation for the
# full cropharvest arrays (all the bands described in .cropharvest_package.bands import BANDS)
X_MEAN = np.array(
    [
        # S1
        -11.402222508898584,
        -18.737524460893283,
        # S2
        1648.2152191941907,
        1573.4610874943905,
        1615.5489068522484,
        1858.2716898520655,
        2601.3555762760366,
        2965.1262098261586,
        2848.7703773525336,
        3200.2996027800878,
        988.3951744805588,
        2388.875987761158,
        1603.9434589134196,
        # ERA5
        291.10068757630137,
        0.002899168594319935,
        # SRTM
        623.3725288552483,
        4.772892120134574,
        # NDVI
        0.3286759541524565,
    ]
)
X_STD = np.array(
    [
        # S1
        3.9990006137706025,
        4.746129503502093,
        # S2
        1381.082565250503,
        1291.5253102557383,
        1476.6850103027462,
        1422.7793815055568,
        1343.5806285212616,
        1396.6833646173918,
        1336.0620748659192,
        1406.3412200105693,
        834.5202295842477,
        1099.279298427952,
        947.4737894434511,
        # ERA5
        9.808570091118522,
        0.0033269562766611444,
        # SRTM
        655.0334629731693,
        6.18644770735272,
        # NDVI
        0.17878359339732122,
    ]
)

# Build X_MIN and X_MAX from minmax_stats.json
_minmax_stats = load_min_max_stats()["cropharvest"]
X_MIN = np.array(
    [
        # S1
        _minmax_stats["sentinel1"]["vv"]["min"],
        _minmax_stats["sentinel1"]["vh"]["min"],
        # S2 (B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12)
        _minmax_stats["sentinel2_l2a"]["02 - Blue"]["min"],
        _minmax_stats["sentinel2_l2a"]["03 - Green"]["min"],
        _minmax_stats["sentinel2_l2a"]["04 - Red"]["min"],
        _minmax_stats["sentinel2_l2a"]["05 - Vegetation Red Edge"]["min"],
        _minmax_stats["sentinel2_l2a"]["06 - Vegetation Red Edge"]["min"],
        _minmax_stats["sentinel2_l2a"]["07 - Vegetation Red Edge"]["min"],
        _minmax_stats["sentinel2_l2a"]["08 - NIR"]["min"],
        _minmax_stats["sentinel2_l2a"]["08A - Vegetation Red Edge"]["min"],
        _minmax_stats["sentinel2_l2a"]["09 - Water vapour"]["min"],
        _minmax_stats["sentinel2_l2a"]["11 - SWIR"]["min"],
        _minmax_stats["sentinel2_l2a"]["12 - SWIR"]["min"],
        # ERA5 (no min/max stats available)
        0.0,
        0.0,
        # SRTM (elevation only, slope uses same stats)
        _minmax_stats["srtm"]["srtm"]["min"],
        _minmax_stats["srtm"]["srtm"]["min"],
        # NDVI (no min/max stats available)
        0.0,
    ]
)
X_MAX = np.array(
    [
        # S1
        _minmax_stats["sentinel1"]["vv"]["max"],
        _minmax_stats["sentinel1"]["vh"]["max"],
        # S2 (B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12)
        _minmax_stats["sentinel2_l2a"]["02 - Blue"]["max"],
        _minmax_stats["sentinel2_l2a"]["03 - Green"]["max"],
        _minmax_stats["sentinel2_l2a"]["04 - Red"]["max"],
        _minmax_stats["sentinel2_l2a"]["05 - Vegetation Red Edge"]["max"],
        _minmax_stats["sentinel2_l2a"]["06 - Vegetation Red Edge"]["max"],
        _minmax_stats["sentinel2_l2a"]["07 - Vegetation Red Edge"]["max"],
        _minmax_stats["sentinel2_l2a"]["08 - NIR"]["max"],
        _minmax_stats["sentinel2_l2a"]["08A - Vegetation Red Edge"]["max"],
        _minmax_stats["sentinel2_l2a"]["09 - Water vapour"]["max"],
        _minmax_stats["sentinel2_l2a"]["11 - SWIR"]["max"],
        _minmax_stats["sentinel2_l2a"]["12 - SWIR"]["max"],
        # ERA5 (no min/max stats available)
        1000.0,
        1.0,
        # SRTM (elevation only, slope uses same stats)
        _minmax_stats["srtm"]["srtm"]["max"],
        _minmax_stats["srtm"]["srtm"]["max"],
        # NDVI (no min/max stats available)
        1.0,
    ]
)


def _s2helios2ch_name(band_name: str) -> str:
    """Transform OlmoEarth Pretrain S2 band name to CropHarvest (and Breizhcrops) S2 band name."""
    band_number = band_name.split(" ")[0]
    if band_number.startswith("0"):
        band_number = band_number[1:]
    return f"B{band_number}"


def _s1helios2ch_name(band_name: str) -> str:
    """Transform OlmoEarth Pretrain S1 band name to CropHarvest (and Breizhcrops) S2 band name."""
    return band_name.upper()


S2_INPUT_TO_OUTPUT_BAND_MAPPING = [
    BANDS.index(_s2helios2ch_name(b))
    for b in EVAL_S2_BAND_NAMES
    if _s2helios2ch_name(b) in BANDS
]
S2_EVAL_BANDS_BEFORE_IMPUTATION = [
    b for b in EVAL_S2_BAND_NAMES if _s2helios2ch_name(b) in BANDS
]

S1_INPUT_TO_OUTPUT_BAND_MAPPING = [
    BANDS.index(_s1helios2ch_name(b)) for b in EVAL_S1_BAND_NAMES
]

# we don't support slope
SRTM_TO_OUTPUT_BAND_MAPPING = [BANDS.index("elevation")]


@memoized
def _get_eval_datasets(root: Path) -> list:
    return CropHarvest.create_benchmark_datasets(
        root=root, balance_negative_crops=False, normalize=False
    )


def _download_cropharvest_data(root: Path) -> None:
    if not root.exists():
        root.mkdir()
        CropHarvest(root, download=True)


class CropHarvestDataset(Dataset):
    """CropHarvest dataset for the Togo cropland classification task."""

    START_MONTH = 1
    DEFAULT_SEED = 42  # mirrors nasaharvest/galileo

    """
    TODO:
      Add partitions
    """

    def __init__(
        self,
        cropharvest_dir: UPath,
        country: str,
        split: str,
        partition: str,
        norm_stats_from_pretrained: bool = False,
        norm_method: str = "norm_no_clip",
        timesteps: int = 12,
        input_modalities: list[str] = [],
    ) -> None:
        """CropHarvest dataset.

        cropharvest_dir: Where the cropharvest data is (or where it should be downloaded)
        country: Country to use in eval. Choice of ["China", "Togo", "Kenya", "Brazil"]
        split: Split to use
        partition: Partition to use
        norm_stats_from_pretrained: Whether to use normalization stats from pretrained model
        norm_method: Normalization method to use, only when norm_stats_from_pretrained is False
        visualize_samples: Whether to visualize samples
        timesteps: num timesteps to use
        """
        self.timesteps = timesteps
        self.config = dataset_to_config("cropharvest")

        _download_cropharvest_data(cropharvest_dir)

        evaluation_datasets = _get_eval_datasets(cropharvest_dir)
        evaluation_datasets = [d for d in evaluation_datasets if country in d.id]
        assert len(evaluation_datasets) == 1
        self.dataset: CropHarvest = evaluation_datasets[0]
        assert self.dataset.task.normalize is False

        if split in ["train", "val", "valid"]:
            array, latlons, labels = self.dataset.as_array()
            array, val_array, latlons, val_latlons, labels, val_labels = (
                train_test_split(
                    # 0.2 is arbitrary
                    array,
                    latlons,
                    labels,
                    test_size=0.2,
                    random_state=self.DEFAULT_SEED,
                    stratify=labels,
                )
            )
            if split == "train":
                self.array = array
                self.latlons = latlons
                self.labels = labels
            else:
                self.array = val_array
                self.latlons = val_latlons
                self.labels = val_labels
        elif split == "test":
            arrays, latlons, labels = [], [], []
            for _, test_instance in self.dataset.test_data(max_size=10000):
                arrays.append(test_instance.x)
                labels.append(test_instance.y)
                latlons.append(
                    np.stack([test_instance.lats, test_instance.lons], axis=-1)
                )
            self.array = np.concatenate(arrays, axis=0)
            self.labels = np.concatenate(labels, axis=0)
            self.latlons = np.concatenate(latlons, axis=0)
        else:
            raise ValueError(f"Unrecognized split {split}")

        self.norm_stats_from_pretrained = norm_stats_from_pretrained
        # We will always need the normalized to normalize latlons
        self.normalizer_computed = Normalizer(Strategy.COMPUTED)
        self.normalizer_predefined = Normalizer(Strategy.PREDEFINED)
        self.norm_method = norm_method
        self.input_modalities = input_modalities
        if len(self.input_modalities) == 0:
            raise ValueError("Expected at least one input modality, got none")

    @staticmethod
    def _normalize_from_ch_stats(array: np.ndarray) -> np.ndarray:
        return (array - X_MEAN) / X_STD

    def __len__(self) -> int:
        """Length of the dataset."""
        return self.array.shape[0]

    @staticmethod
    def _impute_bands(
        image_list: list[np.ndarray],
        names_list: list[str],
        imputes: list[tuple[str, str]],
    ) -> np.ndarray:
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
        return np.stack(new_image_list, axis=-1)  # (h, w, 13)

    def __getitem__(self, idx: int) -> tuple[MaskedOlmoEarthSample, torch.Tensor]:
        """Return the sample at idx."""
        x = self.array[idx]
        y = self.labels[idx]
        latlon = self.normalizer_predefined.normalize(
            Modality.LATLON, self.latlons[idx]
        )

        if not self.norm_stats_from_pretrained:
            x = normalize_bands(x, X_MEAN, X_STD, X_MIN, X_MAX, method=self.norm_method)

        x_hw = repeat(x, "t c -> h w t c", w=1, h=1)

        s2 = x_hw[:, :, : self.timesteps, S2_INPUT_TO_OUTPUT_BAND_MAPPING]

        # for s2, we need to impute missing bands
        s2 = self._impute_bands(
            [s2[:, :, :, idx] for idx in range(s2.shape[-1])],
            S2_EVAL_BANDS_BEFORE_IMPUTATION,
            self.config.imputes,
        )

        s2 = s2[:, :, :, EVAL_TO_HELIOS_S2_BANDS]
        s1 = x_hw[:, :, : self.timesteps, S1_INPUT_TO_OUTPUT_BAND_MAPPING][
            :, :, :, EVAL_TO_HELIOS_S1_BANDS
        ]

        # srtm is only one timestep
        srtm = x_hw[:, :, :1, SRTM_TO_OUTPUT_BAND_MAPPING][
            :, :, :, EVAL_TO_HELIOS_SRTM_BANDS
        ]

        months = torch.tensor(
            np.fmod(
                np.arange(self.START_MONTH - 1, self.START_MONTH - 1 + self.timesteps),
                12,
            )
        ).long()
        days = torch.ones_like(months)
        # TODO: years are not used. Currently (20250710) flexivit only uses the months when
        # computing the composite encodings
        years = torch.ones_like(months) * 2017
        timestamp = torch.stack([days, months, years], dim=-1)  # t, c=3

        if self.norm_stats_from_pretrained:
            s2 = self.normalizer_computed.normalize(Modality.SENTINEL2_L2A, s2)
            s1 = self.normalizer_computed.normalize(Modality.SENTINEL1, s1)
            srtm = self.normalizer_computed.normalize(Modality.SRTM, srtm)

        input_dict: dict[str, torch.Tensor] = {
            "timestamps": timestamp,
        }
        if Modality.SENTINEL2_L2A.name in self.input_modalities:
            input_dict[Modality.SENTINEL2_L2A.name] = torch.tensor(s2).float()
        if Modality.SENTINEL1.name in self.input_modalities:
            input_dict[Modality.SENTINEL1.name] = torch.tensor(s1).float()
        if Modality.SRTM.name in self.input_modalities:
            input_dict[Modality.SRTM.name] = torch.tensor(srtm).float()
        if Modality.LATLON.name in self.input_modalities:
            input_dict[Modality.LATLON.name] = torch.tensor(latlon).float()
        return MaskedOlmoEarthSample.from_olmoearthsample(
            OlmoEarthSample(**input_dict)
        ), y

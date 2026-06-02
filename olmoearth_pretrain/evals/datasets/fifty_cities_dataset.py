"""50Cities (S2 + S1) single-timestep land-cover segmentation eval dataset.

Consumes the per-tile files written by :mod:`fifty_cities_processor`. Each tile
is a 64x64 patch with raw uint16 S2 reflectance, dB S1, and an int class label.

Train/valid/test splits are defined *in code* here (not in a generated file), so
the dataset is fully reproducible from the repo plus the per-tile tensors. Three
split modes are supported, all derived by a deterministic seeded shuffle:

* ``random``       -- tiles shuffled across all cities.
* ``by_city``      -- whole cities assigned to disjoint splits.
* ``by_continent`` -- whole continents assigned to disjoint splits.

Only the per-city tile counts come from the data (``manifest.json``); which
city/continent lands in which split is decided by the constants below.
"""

import json
import logging
import random
from pathlib import Path

import einops
import numpy as np
import torch
from torch.utils.data import Dataset

from olmoearth_pretrain.data.constants import Modality, ModalitySpec
from olmoearth_pretrain.data.dataset import OlmoEarthSample
from olmoearth_pretrain.evals.datasets.constants import (
    EVAL_S2_L2A_BAND_NAMES,
    EVAL_TO_OLMOEARTH_S2_L2A_BANDS,
)
from olmoearth_pretrain.evals.datasets.normalize import normalize_bands
from olmoearth_pretrain.evals.datasets.utils import load_min_max_stats
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

logger = logging.getLogger(__name__)

# Channel order of S1 as stored in the per-tile files (== raw S1.tif order; see
# FiftyCitiesProcessor.S1_TIF_BAND_NAMES). Used to map the stored channels to the
# model's S1 band order and to order the dataset's own S1 norm stats.
S1_STORED_BAND_NAMES = ["vh", "vv"]
S1_STORED_TO_MODEL = [
    S1_STORED_BAND_NAMES.index(b) for b in Modality.SENTINEL1.band_order
]

# Per-band mean/std of this dataset, used only when norm_stats_from_pretrained
# is False. Min/max live in config/minmax_stats.json under "fifty_cities".
# Regenerate all of these with scripts/tools/20260602_fifty_cities_norm_stats.py.
S2_BAND_STATS = {
    "01 - Coastal aerosol": {"mean": 929.987, "std": 903.761},
    "02 - Blue": {"mean": 1050.686, "std": 1019.675},
    "03 - Green": {"mean": 1282.192, "std": 1099.984},
    "04 - Red": {"mean": 1390.080, "std": 1255.881},
    "05 - Vegetation Red Edge": {"mean": 1643.835, "std": 1261.506},
    "06 - Vegetation Red Edge": {"mean": 2045.000, "std": 1276.813},
    "07 - Vegetation Red Edge": {"mean": 2199.791, "std": 1330.759},
    "08 - NIR": {"mean": 2253.265, "std": 1380.702},
    "08A - Vegetation Red Edge": {"mean": 2295.290, "std": 1352.824},
    "09 - Water vapour": {"mean": 2345.512, "std": 1454.604},
    "11 - SWIR": {"mean": 2099.266, "std": 1370.791},
    "12 - SWIR": {"mean": 1754.644, "std": 1325.780},
}
S1_BAND_STATS = {
    "vh": {"mean": -17.401, "std": 5.338},
    "vv": {"mean": -10.663, "std": 5.930},
}

# Continent of every city in the raw 50Cities dataset. Only labeled cities (the
# ones with tiles) are used at runtime; the rest are listed for robustness so
# that labeling more cities later doesn't require touching the split logic.
CITY_TO_CONTINENT = {
    "Athens": "Europe",
    "Auckland": "Oceania",
    "Bangkok": "Asia",
    "Barcelona": "Europe",
    "Berlin": "Europe",
    "Buenos Aires": "South America",
    "Cairo": "Africa",
    "Calgary": "North America",
    "Cape Town": "Africa",
    "Caracas": "South America",
    "Denver": "North America",
    "Houston": "North America",
    "Istanbul": "Europe",
    "Itaituba": "South America",
    "Jakarta": "Asia",
    "Johannesburg": "Africa",
    "Kampala": "Africa",
    "Kansas City": "North America",
    "Kigali": "Africa",
    "Kinshasa": "Africa",
    "Kuala Lumpur": "Asia",
    "Kyoto": "Asia",
    "Lagos": "Africa",
    "Lima": "South America",
    "Los Angeles": "North America",
    "Lugano": "Europe",
    "Marrakesh": "Africa",
    "Matterhorn": "Europe",
    "Mecca": "Asia",
    "Melbourne": "Oceania",
    "Mexico City": "North America",
    "Montevideo": "South America",
    "Mumbai": "Asia",
    "Nairobi": "Africa",
    "New York": "North America",
    "Oslo": "Europe",
    "Perth": "Oceania",
    "Port-au-Prince": "North America",
    "Reykjavik": "Europe",
    "Rio de Janeiro": "South America",
    "Rome": "Europe",
    "San Francisco": "North America",
    "San Jose": "North America",
    "Santiago": "South America",
    "Shanghai": "Asia",
    "Suva": "Oceania",
    "Sydney": "Oceania",
    "Toronto": "North America",
    "Wolfsburg": "Europe",
    "Wuhan": "Asia",
}

# (train, valid, test) fractions used by every split mode, and the seed for the
# deterministic shuffle that assigns units (tiles / cities / continents).
SPLIT_RATIOS = (0.70, 0.15, 0.15)
SPLIT_SEED = 42

SPLIT_MODES = ("random", "by_city", "by_continent")


def _deterministic_partition(
    units: list, ratios: tuple[float, float, float], seed: int
) -> dict[str, list]:
    """Deterministically shuffle ``units`` and split into train/valid/test."""
    units = sorted(units)
    random.Random(seed).shuffle(units)
    n = len(units)
    n_train = int(round(n * ratios[0]))
    n_valid = int(round(n * ratios[1]))
    if n >= 3:
        # Guarantee non-empty valid and test (matters for the small unit counts
        # of by_city / by_continent).
        n_train = min(n_train, n - 2)
        n_valid = max(1, min(n_valid, n - n_train - 1))
    else:
        n_valid = min(n_valid, n - n_train)
    return {
        "train": units[:n_train],
        "valid": units[n_train : n_train + n_valid],
        "test": units[n_train + n_valid :],
    }


class FiftyCitiesDataset(Dataset):
    """50Cities single-timestep S2/S1 segmentation dataset."""

    allowed_modalities = [Modality.SENTINEL1.name, Modality.SENTINEL2_L2A.name]

    # The source imagery has no acquisition date; use a fixed placeholder so the
    # model still receives a valid (single-step) timestamp.
    default_day_month_year = [1, 6, 2020]

    def __init__(
        self,
        path_to_splits: Path,
        split: str = "train",
        split_mode: str = "random",
        input_modalities: list[str] = [],
        norm_stats_from_pretrained: bool = True,
        norm_method: str = "norm_no_clip_2_std",
        label_fraction: float = 1.0,
        label_fraction_seed: int = 42,
    ):
        """Init the 50Cities dataset.

        Args:
            path_to_splits: Output dir from ``FiftyCitiesProcessor`` (holds
                ``tiles/``, ``manifest.json`` and ``colormap.json``).
            split: ``train``, ``valid``/``val`` or ``test``.
            split_mode: ``random``, ``by_city`` or ``by_continent``.
            input_modalities: Subset of ``["sentinel1", "sentinel2_l2a"]``.
            norm_stats_from_pretrained: If True, normalize with the pretrained
                ``COMPUTED`` stats (S2 raw reflectance, S1 in dB). If False, use
                this dataset's own committed stats (``S2_BAND_STATS`` /
                ``S1_BAND_STATS`` here, plus min/max from ``minmax_stats.json``).
            norm_method: Normalization method when not using pretrained stats.
            label_fraction: Fraction of train tiles to keep (low-label evals).
            label_fraction_seed: Seed for the label-fraction subsample.
        """
        if split == "val":
            split = "valid"
        assert split in ["train", "valid", "test"], f"bad split {split}"
        assert split_mode in SPLIT_MODES, f"bad split_mode {split_mode}"
        assert len(input_modalities) > 0, "input_modalities must be set"
        assert all(m in self.allowed_modalities for m in input_modalities), (
            f"input_modalities must be a subset of {self.allowed_modalities}"
        )

        self.path_to_splits = Path(path_to_splits)
        self.tiles_dir = self.path_to_splits / "tiles"
        self.input_modalities = input_modalities
        self.split = split
        self.norm_stats_from_pretrained = norm_stats_from_pretrained
        self.norm_method = norm_method

        self.index = self._build_index(split, split_mode)

        if not 0 < label_fraction <= 1:
            raise ValueError("label_fraction must be in (0, 1].")
        if label_fraction < 1.0 and split == "train":
            rng = np.random.RandomState(label_fraction_seed)
            n_keep = max(1, int(len(self.index) * label_fraction))
            keep = rng.permutation(len(self.index))[:n_keep]
            self.index = [self.index[i] for i in sorted(keep)]

        if self.norm_stats_from_pretrained:
            from olmoearth_pretrain.data.normalize import Normalizer, Strategy

            self.normalizer_computed = Normalizer(Strategy.COMPUTED)
        else:
            self._load_dataset_norm_stats()

    # ------------------------------------------------------------------ #
    # Split construction (in-code, deterministic)
    # ------------------------------------------------------------------ #
    def _tile_counts(self) -> dict[str, int]:
        """Per-city tile counts from the stage-1 manifest."""
        with open(self.path_to_splits / "manifest.json") as f:
            return json.load(f)["counts"]

    def _build_index(self, split: str, split_mode: str) -> list[tuple[str, int]]:
        """Build the (city, tile_idx) index for a split, deterministically."""
        counts = self._tile_counts()
        cities = sorted(counts)
        unknown = [c for c in cities if c not in CITY_TO_CONTINENT]
        assert not unknown, (
            f"Cities missing from CITY_TO_CONTINENT: {unknown}. Add them to "
            "fifty_cities_dataset.py."
        )

        if split_mode == "random":
            units = [(c, i) for c in cities for i in range(counts[c])]
            return _deterministic_partition(units, SPLIT_RATIOS, SPLIT_SEED)[split]

        if split_mode == "by_city":
            chosen = set(
                _deterministic_partition(cities, SPLIT_RATIOS, SPLIT_SEED)[split]
            )
        else:  # by_continent
            continents = sorted({CITY_TO_CONTINENT[c] for c in cities})
            split_conts = set(
                _deterministic_partition(continents, SPLIT_RATIOS, SPLIT_SEED)[split]
            )
            chosen = {c for c in cities if CITY_TO_CONTINENT[c] in split_conts}

        return [(c, i) for c in cities if c in chosen for i in range(counts[c])]

    # ------------------------------------------------------------------ #
    # Normalization
    # ------------------------------------------------------------------ #
    def _load_dataset_norm_stats(self) -> None:
        """Build per-band mean/std/min/max from the in-repo dataset stats.

        Mean/std are hardcoded (``S2_BAND_STATS`` / ``S1_BAND_STATS``); min/max
        come from ``config/minmax_stats.json``. This mirrors how PASTIS / floods
        / MADOS source their dataset stats (all committed to git). Regenerate the
        numbers with ``scripts/tools/20260602_fifty_cities_norm_stats.py``.
        """
        minmax = load_min_max_stats()["fifty_cities"]
        merged_s2 = {
            band: {**S2_BAND_STATS[band], **minmax[Modality.SENTINEL2_L2A.name][band]}
            for band in EVAL_S2_L2A_BAND_NAMES
        }
        # S1 stats are ordered to match the *stored* channel order, since they
        # are applied before the channels are reordered to the model's order.
        merged_s1 = {
            band: {**S1_BAND_STATS[band], **minmax[Modality.SENTINEL1.name][band]}
            for band in S1_STORED_BAND_NAMES
        }
        self.s2_means, self.s2_stds, self.s2_mins, self.s2_maxs = self._stats_arrays(
            merged_s2, EVAL_S2_L2A_BAND_NAMES
        )
        self.s1_means, self.s1_stds, self.s1_mins, self.s1_maxs = self._stats_arrays(
            merged_s1, S1_STORED_BAND_NAMES
        )

    @staticmethod
    def _stats_arrays(
        band_stats: dict[str, dict[str, float]], band_names: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        means, stds, mins, maxs = [], [], [], []
        for name in band_names:
            s = band_stats[name]
            means.append(s["mean"])
            stds.append(s["std"])
            mins.append(s["min"])
            maxs.append(s["max"])
        return np.array(means), np.array(stds), np.array(mins), np.array(maxs)

    def __len__(self) -> int:
        """Number of tiles in this split."""
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[MaskedOlmoEarthSample, torch.Tensor]:
        """Return a single 64x64 tile as a (masked sample, label) pair."""
        city, tile_idx = self.index[idx]
        tile = torch.load(self.tiles_dir / city / f"{tile_idx}.pt")

        # Labels are pre-baked with SEGMENTATION_IGNORE_LABEL for nodata pixels.
        labels = tile["label"].long()  # (64, 64)

        sample_dict: dict[str, torch.Tensor] = {}
        if Modality.SENTINEL2_L2A.name in self.input_modalities:
            sample_dict[Modality.SENTINEL2_L2A.name] = self._load_modality(
                tile["s2"],
                Modality.SENTINEL2_L2A,
                EVAL_TO_OLMOEARTH_S2_L2A_BANDS,
                getattr(self, "s2_means", None),
                getattr(self, "s2_stds", None),
                getattr(self, "s2_mins", None),
                getattr(self, "s2_maxs", None),
            )
        if Modality.SENTINEL1.name in self.input_modalities:
            sample_dict[Modality.SENTINEL1.name] = self._load_modality(
                tile["s1"],
                Modality.SENTINEL1,
                S1_STORED_TO_MODEL,
                getattr(self, "s1_means", None),
                getattr(self, "s1_stds", None),
                getattr(self, "s1_mins", None),
                getattr(self, "s1_maxs", None),
            )

        timestamps = torch.tensor([self.default_day_month_year], dtype=torch.long)
        sample_dict["timestamps"] = timestamps

        masked_sample = MaskedOlmoEarthSample.from_olmoearthsample(
            OlmoEarthSample(**sample_dict)
        )
        return masked_sample, labels

    def _load_modality(
        self,
        chw: torch.Tensor,
        modality: ModalitySpec,
        band_reorder: list[int],
        means: np.ndarray | None,
        stds: np.ndarray | None,
        mins: np.ndarray | None,
        maxs: np.ndarray | None,
    ) -> torch.Tensor:
        """Normalize and reorder one modality tile to (H, W, T=1, C_model)."""
        # (c, h, w) -> (h, w, t=1, c) in the stored eval-band order.
        img = einops.rearrange(chw.numpy(), "c h w -> h w c")[:, :, np.newaxis, :]
        if not self.norm_stats_from_pretrained:
            img = normalize_bands(img, means, stds, mins, maxs, self.norm_method)
        img = img[:, :, :, band_reorder]  # reorder to model band_order
        if self.norm_stats_from_pretrained:
            img = self.normalizer_computed.normalize(modality, img)
        return torch.from_numpy(np.ascontiguousarray(img)).float()

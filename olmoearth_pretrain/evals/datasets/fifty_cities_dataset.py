"""50Cities (S2 + S1) single-timestep land-cover segmentation eval dataset.

Consumes the per-tile files written by :mod:`fifty_cities_processor` and a split
index file (``splits/random.json`` or ``splits/by_city.json``). Each tile is a
64x64 patch with raw uint16 S2 reflectance, dB S1, and an int class label.
"""

import json
import logging
from pathlib import Path

import einops
import numpy as np
import torch
from torch.utils.data import Dataset

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataset import OlmoEarthSample
from olmoearth_pretrain.evals.datasets.constants import (
    EVAL_TO_OLMOEARTH_S1_BANDS,
    EVAL_TO_OLMOEARTH_S2_L2A_BANDS,
)
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

logger = logging.getLogger(__name__)


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
                ``tiles/``, ``splits/``, ``manifest.json``, ``colormap.json``).
            split: ``train``, ``valid``/``val`` or ``test``.
            split_mode: Which split index to use: ``random`` (tiles shuffled
                across cities) or ``by_city`` (disjoint cities per split).
            input_modalities: Subset of ``["sentinel1", "sentinel2_l2a"]``.
            norm_stats_from_pretrained: Must be True -- 50Cities has no
                dataset-specific min/max stats, so we always normalize with the
                pretrained ``COMPUTED`` stats (S2 raw reflectance, S1 in dB).
            norm_method: Unused (kept for a consistent dataset signature).
            label_fraction: Fraction of train tiles to keep (low-label evals).
            label_fraction_seed: Seed for the label-fraction subsample.
        """
        if split == "val":
            split = "valid"
        assert split in ["train", "valid", "test"], f"bad split {split}"
        assert split_mode in ["random", "by_city"], f"bad split_mode {split_mode}"
        assert len(input_modalities) > 0, "input_modalities must be set"
        assert all(m in self.allowed_modalities for m in input_modalities), (
            f"input_modalities must be a subset of {self.allowed_modalities}"
        )
        if not norm_stats_from_pretrained:
            raise NotImplementedError(
                "50Cities only supports norm_stats_from_pretrained=True "
                "(no dataset-specific min/max stats are computed)."
            )

        self.path_to_splits = Path(path_to_splits)
        self.input_modalities = input_modalities
        self.split = split

        with open(self.path_to_splits / f"splits/{split_mode}.json") as f:
            split_payload = json.load(f)
        self.tiles_dir = self.path_to_splits / split_payload.get("tiles_dir", "tiles")
        self.index: list[tuple[str, int]] = [
            (city, int(idx)) for city, idx in split_payload["splits"][split]
        ]

        if not 0 < label_fraction <= 1:
            raise ValueError("label_fraction must be in (0, 1].")
        if label_fraction < 1.0 and split == "train":
            rng = np.random.RandomState(label_fraction_seed)
            n_keep = max(1, int(len(self.index) * label_fraction))
            keep = rng.permutation(len(self.index))[:n_keep]
            self.index = [self.index[i] for i in sorted(keep)]

        from olmoearth_pretrain.data.normalize import Normalizer, Strategy

        self.normalizer_computed = Normalizer(Strategy.COMPUTED)

    def __len__(self) -> int:
        """Number of tiles in this split."""
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[MaskedOlmoEarthSample, torch.Tensor]:
        """Return a single 64x64 tile as a (masked sample, label) pair."""
        city, tile_idx = self.index[idx]
        tile = torch.load(self.tiles_dir / city / f"{tile_idx}.pt")

        # Labels are pre-baked with SEGMENTATION_IGNORE_LABEL for nodata pixels.
        labels = tile["label"].long()  # (64, 64)

        # Single timestep: (c, h, w) -> (h, w, t=1, c), then reorder to model bands.
        sample_dict: dict[str, torch.Tensor] = {}
        if Modality.SENTINEL2_L2A.name in self.input_modalities:
            s2 = einops.rearrange(tile["s2"].numpy(), "c h w -> h w c")
            s2 = s2[:, :, np.newaxis, EVAL_TO_OLMOEARTH_S2_L2A_BANDS]  # (h, w, 1, c)
            s2 = self.normalizer_computed.normalize(Modality.SENTINEL2_L2A, s2)
            sample_dict[Modality.SENTINEL2_L2A.name] = torch.from_numpy(s2).float()
        if Modality.SENTINEL1.name in self.input_modalities:
            s1 = einops.rearrange(tile["s1"].numpy(), "c h w -> h w c")
            s1 = s1[:, :, np.newaxis, EVAL_TO_OLMOEARTH_S1_BANDS]  # (h, w, 1, c)
            s1 = self.normalizer_computed.normalize(Modality.SENTINEL1, s1)
            sample_dict[Modality.SENTINEL1.name] = torch.from_numpy(s1).float()

        timestamps = torch.tensor([self.default_day_month_year], dtype=torch.long)
        sample_dict["timestamps"] = timestamps

        masked_sample = MaskedOlmoEarthSample.from_olmoearthsample(
            OlmoEarthSample(**sample_dict)
        )
        return masked_sample, labels

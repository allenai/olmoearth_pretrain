"""Similar But Different (SBD) dataset class for evals.

Ten-class ESA WorldCover classification on 30,927 Sentinel-2 L2A 32x32 patches
(12 bands), built so the visible bands are uninformative by construction -- a probe
for whether a model uses the non-visible bands. Prepared into per-split tensors by
``sbd_processor``. See ``sbd_processor`` for the source/layout, and
https://huggingface.co/datasets/calebrob6/similar-but-different for the dataset.

Mirrors the FiftyCities/MADOS L2A path: stored patches are raw uint16 surface
reflectance in ``EVAL_S2_L2A_BAND_NAMES`` order; they are normalized (either with the
pretrained model's stats or this dataset's committed stats) and reindexed to the model
band order via ``EVAL_TO_OLMOEARTH_S2_L2A_BANDS`` before being wrapped in an
``OlmoEarthSample``.
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from einops import repeat
from torch.utils.data import Dataset

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataset import OlmoEarthSample
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

from .constants import EVAL_S2_L2A_BAND_NAMES, EVAL_TO_OLMOEARTH_S2_L2A_BANDS
from .normalize import NormMethod, normalize_bands

logger = logging.getLogger(__name__)


class SBDDataset(Dataset):
    """Similar But Different classification dataset (Sentinel-2 L2A, 10 classes)."""

    # Single fixed timestamp: SBD ships one composite date per patch (no true date),
    # matching the single-timestamp convention of the other patch eval sets.
    default_day_month_year = [1, 6, 2021]

    def __init__(
        self,
        path_to_splits: Path,
        split: str,
        label_fraction: float = 1.0,
        norm_stats_from_pretrained: bool = True,
        norm_method: str = NormMethod.NORM_NO_CLIP_2_STD,
    ):
        """Init the SBD dataset.

        Args:
            path_to_splits: dir written by sbd_processor (SBD_{split}.pt + norm_stats.json).
            split: one of train / val / valid / test.
            label_fraction: fraction of train patches to keep (stratified by class).
            norm_stats_from_pretrained: if True, normalize with the pretrained model's
                stats (Normalizer COMPUTED); else use this dataset's committed stats.
            norm_method: normalization method used only when not using pretrained stats.
        """
        assert split in ["train", "val", "valid", "test"]
        if split == "valid":
            split = "val"
        self.split = split
        self.norm_method = norm_method
        self.norm_stats_from_pretrained = norm_stats_from_pretrained

        path_to_splits = Path(path_to_splits)
        obj = torch.load(path_to_splits / f"SBD_{split}.pt")
        self.images = obj["images"]  # (N, 32, 32, 12) uint16
        self.labels = obj["labels"].long()  # (N,)
        self.patch_ids = obj["patch_ids"]

        if norm_stats_from_pretrained:
            from olmoearth_pretrain.data.normalize import Normalizer, Strategy

            self.normalizer_computed = Normalizer(Strategy.COMPUTED)
        else:
            stats = json.loads((path_to_splits / "norm_stats.json").read_text())
            self.means, self.stds, self.mins, self.maxs = self._stats_arrays(stats)

        if not 0 < label_fraction <= 1:
            raise ValueError("label_fraction must be in (0, 1].")
        if label_fraction < 1.0 and split == "train":
            self._subsample_train(label_fraction)

    @staticmethod
    def _stats_arrays(
        stats: dict[str, dict[str, float]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        means, stds, mins, maxs = [], [], [], []
        for band in EVAL_S2_L2A_BAND_NAMES:
            s = stats[band]
            means.append(s["mean"])
            stds.append(s["std"])
            mins.append(s["min"])
            maxs.append(s["max"])
        return np.array(means), np.array(stds), np.array(mins), np.array(maxs)

    def _subsample_train(self, label_fraction: float) -> None:
        """Keep a class-stratified fraction of the train patches (deterministic)."""
        rng = np.random.default_rng(42)
        keep = []
        labels = self.labels.numpy()
        for cls in np.unique(labels):
            idx = np.where(labels == cls)[0]
            n = max(1, int(round(len(idx) * label_fraction)))
            keep.append(rng.choice(idx, size=n, replace=False))
        keep_idx = np.sort(np.concatenate(keep))
        # torch can't advanced-index a uint16 CPU tensor, so select via numpy.
        self.images = torch.from_numpy(self.images.numpy()[keep_idx])
        self.labels = self.labels[torch.from_numpy(keep_idx)]
        self.patch_ids = [self.patch_ids[i] for i in keep_idx]

    def __len__(self) -> int:
        """Number of patches in this split."""
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> tuple[MaskedOlmoEarthSample, torch.Tensor]:
        """Return one patch as a (masked sample, scalar label) pair."""
        image = self.images[idx].numpy().astype(np.float32)  # (32, 32, 12), raw SR

        if not self.norm_stats_from_pretrained:
            image = normalize_bands(
                image, self.means, self.stds, self.mins, self.maxs, self.norm_method
            )
        # (H, W, C) -> (H, W, T=1, C), then reindex to the model's S2 L2A band order.
        image = repeat(image, "h w c -> h w t c", t=1)[
            :, :, :, EVAL_TO_OLMOEARTH_S2_L2A_BANDS
        ]
        if self.norm_stats_from_pretrained:
            image = self.normalizer_computed.normalize(Modality.SENTINEL2_L2A, image)

        timestamp = repeat(torch.tensor(self.default_day_month_year), "d -> t d", t=1)
        masked_sample = MaskedOlmoEarthSample.from_olmoearthsample(
            OlmoEarthSample(
                sentinel2_l2a=torch.tensor(image).float(), timestamps=timestamp.long()
            )
        )
        return masked_sample, self.labels[idx]

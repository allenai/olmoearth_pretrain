"""SwissCrop25 (S2 monthly mosaics) crop-type segmentation dataset for linear-probe / kNN evals.

Reads the tensors written by ``swisscrop_processor.py`` and serves them like PASTIS-R:
one ``MaskedOlmoEarthSample`` (Sentinel-2 only, 12 timesteps) plus a per-pixel target.
Splits follow the SwissCrop25 leave-one-year-out (LOYO) protocol; the default fold is S5
(test 2025, val 2024, train 2019-2023). Background pixels (no parcel / excluded LNF code)
are mapped to the segmentation ignore label; the 70 non-excluded ``Crop_Label`` classes
(65 crops + 5 non-crop land-cover classes) become 0..69.
"""

import glob
import json
import logging
from pathlib import Path

import einops
import numpy as np
import torch
from torch.utils.data import Dataset

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, OlmoEarthSample
from olmoearth_pretrain.evals.metrics import SEGMENTATION_IGNORE_LABEL

logger = logging.getLogger(__name__)

# Leave-one-year-out folds from the SwissCrop25 paper / README: (train, valid, test) years.
LOYO_FOLDS: dict[str, tuple[list[int], list[int], list[int]]] = {
    "S1": ([2019, 2022, 2023, 2024, 2025], [2020], [2021]),
    "S2": ([2019, 2020, 2023, 2024, 2025], [2021], [2022]),
    "S3": ([2019, 2020, 2021, 2024, 2025], [2022], [2023]),
    "S4": ([2019, 2020, 2021, 2022, 2025], [2023], [2024]),
    "S5": ([2019, 2020, 2021, 2022, 2023], [2024], [2025]),
}
NUM_CLASSES = 70  # non-excluded Crop_Label classes (65 crops + Built-up, Forest, Unproductive Area, Water, Wetland)


class SwissCropDataset(Dataset):
    """SwissCrop25 monthly-mosaic segmentation dataset."""

    allowed_modalities = [Modality.SENTINEL2_L2A.name]

    def __init__(
        self,
        path_to_processed: Path | str,
        split: str = "train",
        fold: str = "S5",
        tile_size: int = 64,
        label_fraction: float = 1.0,
        norm_stats_from_pretrained: bool = True,
        norm_method: str = "norm_no_clip_2_std",
        input_modalities: list[str] = [],
        seed: int = 42,
    ):
        """Init.

        Args:
            path_to_processed: output dir of swisscrop_processor.py ({year}/{tile}.pt + classes.json)
            split: train / valid / test
            fold: LOYO fold name (S1..S5)
            tile_size: 128 (whole cube) or 64 (four quadrants per cube)
            label_fraction: fraction of training cubes to keep (deterministic subsample)
            norm_stats_from_pretrained: normalize with the pretraining Normalizer (COMPUTED stats)
            norm_method: only used when norm_stats_from_pretrained is False
            input_modalities: must be [sentinel2_l2a]
            seed: seed for the label-fraction subsample
        """
        assert split in ["train", "valid", "test"], split
        assert fold in LOYO_FOLDS, f"fold must be one of {list(LOYO_FOLDS)}"
        assert tile_size in (64, 128)
        assert input_modalities == [Modality.SENTINEL2_L2A.name], (
            f"SwissCrop25 provides Sentinel-2 only, got {input_modalities}"
        )
        self.root = Path(path_to_processed)
        self.split = split
        self.tile_size = tile_size
        self.input_modalities = input_modalities
        self.norm_stats_from_pretrained = norm_stats_from_pretrained
        self.norm_method = norm_method
        meta = json.load(open(self.root / "classes.json"))
        self.classes: list[str] = meta["classes"]
        assert len(self.classes) == NUM_CLASSES, (len(self.classes), NUM_CLASSES)

        years = LOYO_FOLDS[fold][{"train": 0, "valid": 1, "test": 2}[split]]
        files: list[str] = []
        for y in years:
            fs = sorted(glob.glob(str(self.root / str(y) / "*.pt")))
            if not fs:
                logger.warning(
                    f"SwissCrop25: no processed cubes for {y} under {self.root}"
                )
            files.extend(fs)
        if split == "train" and label_fraction < 1.0:
            rng = np.random.default_rng(seed)
            keep = rng.choice(
                len(files), size=max(1, int(len(files) * label_fraction)), replace=False
            )
            files = [files[i] for i in sorted(keep)]
        n_q = 1 if tile_size == 128 else 4
        self.items = [(f, q) for f in files for q in range(n_q)]
        logger.info(
            f"SwissCrop25 fold {fold} {split}: years {years}, {len(files)} cubes, {len(self.items)} samples"
        )

        if norm_stats_from_pretrained:
            from olmoearth_pretrain.data.normalize import Normalizer, Strategy

            self.normalizer_computed = Normalizer(Strategy.COMPUTED)
        else:
            # Dataset-stats path: mean/std of the pretraining S2 stats, min/max from the
            # processed cubes are not tracked; fall back to the pretraining normalizer.
            logger.warning(
                "SwissCrop25: norm_stats_from_pretrained=False not supported; using pretraining stats"
            )
            from olmoearth_pretrain.data.normalize import Normalizer, Strategy

            self.normalizer_computed = Normalizer(Strategy.COMPUTED)

    def __len__(self) -> int:
        """Number of samples."""
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[MaskedOlmoEarthSample, torch.Tensor]:
        """Return (masked sample, target)."""
        path, q = self.items[idx]
        d = torch.load(path)
        s2 = d["s2"].numpy()  # (T=12, C=12, 128, 128) uint16, OlmoEarth band order
        target = d["target"].numpy().astype(np.int64)  # (128, 128), 0 = background
        if self.tile_size == 64:
            r, c = divmod(q, 2)
            s2 = s2[:, :, r * 64 : (r + 1) * 64, c * 64 : (c + 1) * 64]
            target = target[r * 64 : (r + 1) * 64, c * 64 : (c + 1) * 64]
        s2 = einops.rearrange(s2.astype(np.float32), "t c h w -> h w t c")
        s2 = self.normalizer_computed.normalize(Modality.SENTINEL2_L2A, s2)

        months = d["months"].tolist()  # yyyymm
        timestamps = torch.tensor(
            [[15, int(m) % 100 - 1, int(m) // 100] for m in months], dtype=torch.long
        )  # (day, month 0-11, year)
        target = torch.from_numpy(target)
        target = torch.where(
            target == 0,
            torch.tensor(SEGMENTATION_IGNORE_LABEL, dtype=torch.long),
            target - 1,
        )

        sample = OlmoEarthSample(
            sentinel2_l2a=torch.from_numpy(np.ascontiguousarray(s2)).float(),
            timestamps=timestamps,
        )
        return MaskedOlmoEarthSample.from_olmoearthsample(sample), target

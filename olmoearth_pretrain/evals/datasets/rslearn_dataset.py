"""Convert rslearn dataset to OlmoEarth Pretrain evaluation dataset format."""

import json
from collections import defaultdict
from datetime import datetime
from importlib.resources import files
from typing import Any

import numpy as np
import torch
from dateutil.relativedelta import relativedelta
from einops import rearrange
from rslearn.dataset.dataset import Dataset as RslearnDataset
from rslearn.train.dataset import DataInput as RsDataInput
from rslearn.train.dataset import ModelDataset as RsModelDataset
from rslearn.train.dataset import SplitConfig as RsSplitConfig
from rslearn.train.tasks.classification import (
    ClassificationTask as RsClassificationTask,
)
from rslearn.train.transforms.pad import Pad as RsPad
from torch.utils.data import Dataset
from upath import UPath

from olmoearth_pretrain.data.constants import YEAR_NUM_TIMESTEPS
from olmoearth_pretrain.data.constants import Modality as DataModality
from olmoearth_pretrain.data.utils import convert_to_db
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample, OlmoEarthSample

from .normalize import normalize_bands

# rslearn layer name -> (olmoearth modality name, all bands)
RSLEARN_TO_OLMOEARTH: dict[str, tuple[str, list[str]]] = {
    "sentinel2": ("sentinel2_l2a", DataModality.SENTINEL2_L2A.band_order),
    "sentinel1": ("sentinel1", DataModality.SENTINEL1.band_order),
    "sentinel1_ascending": ("sentinel1", DataModality.SENTINEL1.band_order),
    "sentinel1_descending": ("sentinel1", DataModality.SENTINEL1.band_order),
    "landsat": ("landsat", DataModality.LANDSAT.band_order),
}


def build_rslearn_model_dataset(
    rslearn_dataset: RslearnDataset,
    layers: list[str],
    classes: list[str],
    rslearn_dataset_groups: list[str] | None = None,
    input_size: int | None = None,
    split: str = "train",
    property_name: str = "category",
    skip_targets: bool = False,
) -> RsModelDataset:
    """Build an rslearn ModelDataset.

    Args:
        rslearn_dataset: The source RslearnDataset.
        rslearn_dataset_groups: Optional list of dataset group names to include.
        layers: List of rslearn layer names to use as model inputs.
            Example: "sentinel2". Only provide the base name, do not include
            layer names such as "sentinel2.1" or "sentinel2.n".
        input_size: Optional input patch size (pixels) to crop/resize samples to.
        split: Dataset split to use (e.g., "train", "val", "test").
        property_name: The property in the dataset to use as the target label.
        classes: List of class names. If None, inferred from dataset.
        skip_targets: Whether or not to skip the target, if True, property_name and classes can be placeholder.

    Returns:
        RsModelDataset: A dataset object ready for training or evaluation.
    """
    if not layers:
        raise ValueError(
            "`layers` must be a non-empty list of rslearn layer names, "
            f"allowed: {list(RSLEARN_TO_OLMOEARTH.keys())}"
        )
    if split not in ("train", "val", "test"):
        raise ValueError(f"Invalid split {split}, must be one of train/val/test")

    # Validate input layers
    unknown = [m for m in layers if m not in RSLEARN_TO_OLMOEARTH]
    if unknown:
        raise ValueError(
            f"Unknown rslearn layer(s): {unknown}. "
            f"Allowed: {list(RSLEARN_TO_OLMOEARTH.keys())}"
        )

    if classes is None:
        raise ValueError("`classes` must be provided and cannot be None.")

    # Group rslearn layers by their OlmoEarth Pretrain modality key
    layers_by_olmoearth: dict[str, list[str]] = defaultdict(list)
    bands_by_olmoearth: dict[str, list[str]] = {}

    for rslearn_layer in layers:
        olmoearth_key, band_order = RSLEARN_TO_OLMOEARTH[rslearn_layer]
        layers_by_olmoearth[olmoearth_key].append(rslearn_layer)
        bands_by_olmoearth[olmoearth_key] = band_order

    transforms = []
    if input_size is not None:
        transforms.append(
            RsPad(
                size=input_size,
                mode="center",
                image_selectors=list(layers_by_olmoearth.keys()),
            )
        )

    inputs: dict[str, RsDataInput] = {}
    # Expand each rslearn layer name to time-indexed variants, keep the first *per base layer*
    for olmoearth_key, per_key_layers in layers_by_olmoearth.items():
        expanded: list[str] = []
        for base in per_key_layers:
            # convention: base, then base.1 ... base.(YEAR_NUM_TIMESTEPS-1)
            expanded.append(base)
            expanded.extend(f"{base}.{i}" for i in range(1, YEAR_NUM_TIMESTEPS))
        inputs[olmoearth_key] = RsDataInput(
            data_type="raster",
            layers=expanded,
            bands=bands_by_olmoearth[olmoearth_key],
            passthrough=True,
            load_all_layers=True,
        )

    inputs["targets"] = RsDataInput(
        data_type="vector",
        layers=["label"],
        is_target=True,
    )

    split_config = RsSplitConfig(
        transforms=transforms,
        groups=rslearn_dataset_groups,
        skip_targets=skip_targets,
        tags={"helios_split": split}
        if split
        else {},  # must stay as helios because it is tagged that way in the dataset
    )

    return RsModelDataset(
        dataset=rslearn_dataset,
        split_config=split_config,
        inputs=inputs,
        # TODO: add task type later if we also want to support segmentation task
        task=RsClassificationTask(
            property_name=property_name,
            classes=classes,
        ),
        workers=32,
    )


def get_timestamps(
    start_time: str,
    end_time: str,
) -> list[torch.Tensor]:
    """Return first YEAR_NUM_TIMESTEPS monthly (day, month0, year) long tensors."""
    start = datetime.strptime(start_time, "%Y-%m-%d").replace(day=1)
    end = datetime.strptime(end_time, "%Y-%m-%d")

    months_diff = (end.year - start.year) * 12 + (end.month - start.month) + 1
    if months_diff < YEAR_NUM_TIMESTEPS:
        raise ValueError(
            f"Not enough months in range ({months_diff}) to cover {YEAR_NUM_TIMESTEPS}"
        )

    dates: list[torch.Tensor] = []
    cur = start
    while cur <= end and len(dates) < YEAR_NUM_TIMESTEPS:
        # month stored 0-indexed
        dates.append(
            torch.tensor(
                [int(cur.day), int(cur.month) - 1, int(cur.year)], dtype=torch.long
            )
        )
        cur += relativedelta(months=1)
    return dates


class RslearnToOlmoEarthDataset(Dataset):
    """Convert rslearn ModelDataset to OlmoEarth Pretrain MaskedOlmoEarthSample dataset.

    Expects rslearn ModelDataset to yield: (inputs_dict, target, metadata).
    inputs_dict[<modality>] shape: (T*C, H, W) after rslearn transforms.
    We reshape to (H, W, T, C), normalize, attach timestamps, and wrap as OlmoEarthSample.
    """

    allowed_modalities = {
        DataModality.SENTINEL2_L2A.name,
        DataModality.SENTINEL1.name,
        DataModality.LANDSAT.name,
    }

    def __init__(
        self,
        ds_path: str,
        layers: list[str],
        classes: list[str],
        input_modalities: list[str],
        ds_groups: list[str] | None = None,
        input_size: int | None = None,
        split: str = "train",
        property_name: str = "category",
        partition: str = "default",
        norm_stats_from_pretrained: bool = True,
        norm_method: str = "norm_no_clip",
        ds_norm_stats_json: str | None = None,
        start_time: str = "2022-09-01",
        end_time: str = "2023-09-01",
    ):
        """Initialize RslearnToOlmoEarthDataset."""
        if split not in ("train", "val", "valid", "test"):
            raise ValueError(f"Invalid split {split}")

        if not norm_stats_from_pretrained and ds_norm_stats_json is None:
            raise ValueError(
                "norm_stats_from_pretrained=False requires a JSON file with dataset stats "
                "(set ds_norm_stats_json)."
            )

        if not input_modalities:
            raise ValueError("Must specify at least one input modality")
        if not all(m in self.allowed_modalities for m in input_modalities):
            raise ValueError(f"Input modalities must be in {self.allowed_modalities}")

        # Build rslearn ModelDataset for the split
        dataset = RslearnDataset(UPath(ds_path))
        self.dataset = build_rslearn_model_dataset(
            rslearn_dataset=dataset,
            rslearn_dataset_groups=ds_groups,
            layers=layers,
            input_size=input_size,
            split="val" if split == "valid" else split,  # rslearn uses 'val'
            property_name=property_name,
            classes=classes,
        )

        self.norm_stats_from_pretrained = norm_stats_from_pretrained
        self.input_modalities = input_modalities
        self.timestamps = torch.stack(get_timestamps(start_time, end_time))  # (T, 3)

        if self.norm_stats_from_pretrained:
            from olmoearth_pretrain.data.normalize import Normalizer, Strategy

            self.normalizer_computed = Normalizer(Strategy.COMPUTED)
        else:
            self.dataset_norm_stats = self._get_norm_stats(ds_norm_stats_json)  # type: ignore
            self.norm_method = norm_method

    @staticmethod
    def _get_norm_stats(ds_norm_stats_json: str) -> dict:
        """Load dataset norm stats."""
        with (
            files("olmoearth_pretrain.evals.datasets.config") / ds_norm_stats_json
        ).open() as f:
            blob = json.load(f)
        out = {}
        for modality, per_band in blob.items():
            band_order = DataModality.get(modality).band_order
            means, stds, mins, maxs = [], [], [], []

            for band in band_order:
                band_stats = (
                    per_band.get(band)
                    or per_band.get(band.upper())
                    or per_band.get(band.lower())
                )
                if band_stats is None:
                    raise ValueError(f"Missing stats for {band} in modality {modality}")
                means.append(band_stats["mean"])
                stds.append(band_stats["std"])
                mins.append(band_stats["min"])
                maxs.append(band_stats["max"])

            out[modality] = {
                "means": np.array(means, dtype=np.float32),
                "stds": np.array(stds, dtype=np.float32),
                "mins": np.array(mins, dtype=np.float32),
                "maxs": np.array(maxs, dtype=np.float32),
            }

        return out

    def __len__(self) -> int:
        """Length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[MaskedOlmoEarthSample, torch.Tensor]:
        """Return a MaskedOlmoEarthSample and target tensor."""
        input_dict, target, _ = self.dataset[idx]

        sample_dict: dict[str, Any] = {}
        T = YEAR_NUM_TIMESTEPS

        for modality in self.input_modalities:
            if modality not in input_dict:
                raise ValueError(f"Modality {modality} not found in dataset inputs")
            num_bands = DataModality.get(modality).num_bands
            x = input_dict[modality]
            # Turn into numpy array for compatibility with normalizer
            if not isinstance(x, np.ndarray):
                x = x.numpy()
            if x.ndim != 3:
                raise ValueError(
                    f"Expected (T*C, H, W) for {modality}, got {tuple(x.shape)}"
                )
            # Convert to dB for Sentinel-1
            if modality == DataModality.SENTINEL1.name:
                x = convert_to_db(x)
            x = rearrange(x, "(t c) h w -> h w t c", t=T, c=num_bands)

            if self.norm_stats_from_pretrained:
                x = self.normalizer_computed.normalize(DataModality.get(modality), x)
            else:
                modality_stats = self.dataset_norm_stats[modality]
                x = normalize_bands(
                    image=x,
                    means=modality_stats["means"],
                    stds=modality_stats["stds"],
                    mins=modality_stats["mins"],
                    maxs=modality_stats["maxs"],
                    method=self.norm_method,
                )

            sample_dict[modality] = torch.as_tensor(x, dtype=torch.float32)

        sample_dict["timestamps"] = self.timestamps

        olmoearth_sample = OlmoEarthSample(**sample_dict)
        masked_sample = MaskedOlmoEarthSample.from_olmoearthsample(olmoearth_sample)
        return masked_sample, target["class"].long()

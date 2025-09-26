"""Convert rslearn dataset to Helios evaluation dataset format."""

# --- must be directly after the module docstring ---
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

import torch
from dateutil.relativedelta import relativedelta
from einops import rearrange

# rslearn
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

from helios.data.constants import YEAR_NUM_TIMESTEPS

# helios
from helios.data.constants import Modality as DataModality
from helios.data.utils import convert_to_db
from helios.train.masking import HeliosSample, MaskedHeliosSample

# rslearn layer name -> (helios modality name, all bands)
RSLEARN_TO_HELIOS: dict[str, tuple[str, list[str]]] = {
    "sentinel2": ("sentinel2_l2a", DataModality.SENTINEL2_L2A.all_bands),
    "sentinel1": ("sentinel1", DataModality.SENTINEL1.all_bands),
    "sentinel1_ascending": ("sentinel1", DataModality.SENTINEL1.all_bands),
    "sentinel1_descending": ("sentinel1", DataModality.SENTINEL1.all_bands),
    "landsat": ("landsat", DataModality.LANDSAT.all_bands),
}


def build_rslearn_model_dataset(
    rslearn_dataset: RslearnDataset,
    rslearn_dataset_groups: list[str] | None = None,
    layers: list[str] | None = None,
    input_size: int | None = None,
    split: str = "train",
    property_name: str = "category",
    classes: list[str] | None = None,
) -> RsModelDataset:
    """Build an rslearn ModelDataset."""
    if not layers:
        raise ValueError(
            "`layers` must be a non-empty list of rslearn layer names, "
            f"allowed: {list(RSLEARN_TO_HELIOS.keys())}"
        )
    if split not in ("train", "val", "test"):
        raise ValueError(f"Invalid split {split}, must be one of train/val/test")

    # Validate input layers
    unknown = [m for m in layers if m not in RSLEARN_TO_HELIOS]
    if unknown:
        raise ValueError(
            f"Unknown rslearn layer(s): {unknown}. "
            f"Allowed: {list(RSLEARN_TO_HELIOS.keys())}"
        )

    # Group rslearn layers by their Helios modality key
    layers_by_helios: dict[str, list[str]] = defaultdict(list)
    bands_by_helios: dict[str, list[str]] = {}

    for rslearn_layer in layers:
        helios_key, band_order = RSLEARN_TO_HELIOS[rslearn_layer]
        layers_by_helios[helios_key].append(rslearn_layer)
        bands_by_helios[helios_key] = band_order

    transforms = []
    if input_size is not None:
        # Use the rslearn Pad to match its loader pipeline (avoid import collisions)
        transforms.append(
            RsPad(
                size=input_size,
                mode="center",
                image_selectors=list(layers_by_helios.keys()),
            )
        )

    inputs: dict[str, RsDataInput] = {}
    # Expand each rslearn layer name to time-indexed variants; keep the first *per base layer*
    for helios_key, per_key_layers in layers_by_helios.items():
        expanded: list[str] = []
        for base in per_key_layers:
            # convention: base, then base.1 ... base.(YEAR_NUM_TIMESTEPS-1)
            expanded.append(base)
            expanded.extend(f"{base}.{i}" for i in range(1, YEAR_NUM_TIMESTEPS))
        inputs[helios_key] = RsDataInput(
            data_type="raster",
            layers=expanded,
            bands=bands_by_helios[helios_key],
            passthrough=True,
            load_all_layers=True,
        )

    # Always include the targets layer if it exists
    inputs["targets"] = RsDataInput(
        data_type="vector",
        layers=["label"],
        is_target=True,
    )

    split_config = RsSplitConfig(
        transforms=transforms,
        groups=rslearn_dataset_groups,
        skip_targets=False,
        tags={"split": split} if split else {},
    )

    return RsModelDataset(
        dataset=rslearn_dataset,
        split_config=split_config,
        inputs=inputs,
        # Dummy task to allow vector labels to flow (not used for metrics here)
        task=RsClassificationTask(
            # TODO: add property name and classes as args
            property_name=property_name,
            classes=classes,
        ),
        workers=0,
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


class RslearnToHeliosDataset(Dataset):
    """Convert rslearn ModelDataset to Helios MaskedHeliosSample dataset.

    Expects rslearn ModelDataset to yield: (inputs_dict, target, metadata).
    inputs_dict[<modality>] shape: (T*C, H, W) after rslearn transforms.
    We reshape to (H, W, T, C), normalize, attach timestamps, and wrap as HeliosSample.
    """

    allowed_modalities = {
        DataModality.SENTINEL2_L2A.name,
        DataModality.SENTINEL1.name,
        DataModality.LANDSAT.name,
    }

    def __init__(
        self,
        ds_path: str,
        ds_groups: list[str] | None = None,
        layers: list[str] | None = None,
        input_size: int | None = None,
        split: str = "train",
        property_name: str = "category",
        classes: list[str] | None = None,
        partition: str = "default",  # accepted but unused (rslearn)
        norm_stats_from_pretrained: bool = True,
        norm_method: str = "norm_no_clip",  # accepted but unused (rslearn)
        input_modalities: list[str] | None = None,
        start_time: str = "2022-09-01",
        end_time: str = "2023-09-01",
    ):
        """Initialize RslearnToHeliosDataset."""
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split {split}")
        if not norm_stats_from_pretrained:
            raise ValueError("Only norm_stats_from_pretrained=True is supported")

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
            split=split,
            property_name=property_name,
            classes=classes,
        )

        self.norm_stats_from_pretrained = norm_stats_from_pretrained
        self.input_modalities = input_modalities
        self.timestamps = torch.stack(
            get_timestamps(start_time, end_time)
        )  # (T, 3) long

        if self.norm_stats_from_pretrained:
            from helios.data.normalize import (  # lazy import to avoid heavy deps on import time
                Normalizer,
                Strategy,
            )

            self.normalizer_computed = Normalizer(Strategy.COMPUTED)

    def __len__(self) -> int:
        """Length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[MaskedHeliosSample, torch.Tensor]:
        """Return a MaskedHeliosSample and target tensor."""
        input_dict, target, _ = self.dataset[idx]

        sample_dict: dict[str, Any] = {}
        T = YEAR_NUM_TIMESTEPS

        for modality in self.input_modalities:
            if modality not in input_dict:
                raise ValueError(f"Modality {modality} not found in dataset inputs")

            x = input_dict[modality]
            # Expect (T*C, H, W)
            if x.ndim != 3:
                raise ValueError(
                    f"Expected (T*C, H, W) for {modality}, got {tuple(x.shape)}"
                )

            # Convert to dB for Sentinel-1
            if modality == DataModality.SENTINEL1.name:
                x = convert_to_db(x)

            # (T*C, H, W) -> (H, W, T, C)
            x = rearrange(x, "(t c) h w -> h w t c", t=T)

            if self.norm_stats_from_pretrained:
                x = self.normalizer_computed.normalize(DataModality.get(modality), x)

            # ensure float32 tensor
            sample_dict[modality] = torch.as_tensor(x, dtype=torch.float32)

        sample_dict["timestamps"] = self.timestamps  # (T, 3) long

        helios_sample = HeliosSample(**sample_dict)
        masked_sample = MaskedHeliosSample.from_heliossample(helios_sample)
        return masked_sample, target["class"].long()


# # Example usage:
# from torch.utils.data import DataLoader


# def main():
#     ds_path = "/weka/dfive-default/rslearn-eai/datasets/crop/kenya_nandi/20250625"

#     dataset = RslearnToHeliosDataset(
#         ds_path=ds_path,
#         ds_groups=None,
#         layers=["sentinel2"],
#         input_size=32,
#         split="train",
#         input_modalities=["sentinel2_l2a"],
#         start_time="2022-09-01",
#         end_time="2023-08-31",
#     )

#     loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

#     for masked_samples, targets in loader:
#         print(masked_samples)
#         print(targets.shape)
#         break


# if __name__ == "__main__":
#     import multiprocessing as mp

#     try:
#         mp.set_start_method("fork")
#     except RuntimeError:
#         pass
#     main()

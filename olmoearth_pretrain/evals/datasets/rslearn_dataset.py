"""Convert rslearn dataset to OlmoEarth Pretrain evaluation dataset format."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from importlib.resources import files
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from olmoearth_pretrain.evals.studio_ingest.schema import EvalDatasetEntry

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
from rslearn.train.tasks.segmentation import SegmentationTask as RsSegmentationTask
from rslearn.train.model_context import RasterImage
from rslearn.train.transforms.crop import Crop as RsCrop
from rslearn.train.transforms.pad import Pad as RsPad
from torch.utils.data import Dataset
from upath import UPath

from olmoearth_pretrain.data.constants import YEAR_NUM_TIMESTEPS
from olmoearth_pretrain.data.constants import Modality as DataModality
from olmoearth_pretrain.data.utils import convert_to_db
from olmoearth_pretrain.evals.metrics import SEGMENTATION_IGNORE_LABEL
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

# Reverse mapping: olmoearth modality name -> default rslearn layer name
OLMOEARTH_TO_RSLEARN: dict[str, str] = {
    "sentinel2_l2a": "sentinel2",
    "SENTINEL2_L2A": "sentinel2",
    "sentinel1": "sentinel1",
    "SENTINEL1": "sentinel1",
    "landsat": "landsat",
    "LANDSAT": "landsat",
}



# Why are we extracting the info if we could just build this from the config using rslearn?
# SO that it is maintainable if the rslearn set up or api or config changes in the future?
def build_rslearn_model_dataset(
    rslearn_dataset: RslearnDataset,
    layers: list[str],
    classes: list[str],
    task_type: str = "classification",
    rslearn_dataset_groups: list[str] | None = None,
    split: str = "train",
    property_name: str = "category",
    skip_targets: bool = False,
    max_samples: int | None = None,
    split_tag_key: str = "split",
    target_layer_name: str = "label",
    target_data_type: str = "vector",
    target_bands: list[str] | None = None,
    crop_size: int | None = None,
    pad_size: int | None = None,
) -> RsModelDataset:
    """Build an rslearn ModelDataset.

    Args:
        rslearn_dataset: The source RslearnDataset.
        layers: List of rslearn layer names to use as model inputs.
        classes: List of class names.
        task_type: "classification" or "segmentation".
        rslearn_dataset_groups: Optional list of dataset group names to include.
        split: Dataset split to use (e.g., "train", "val", "test").
        property_name: The property in the dataset to use as the target label.
        skip_targets: Whether to skip the target loading.
        split_tag_key: The tag key used to identify splits (e.g., "split" or "helios_split").
        target_layer_name: rslearn layer name for targets (e.g., "label", "label_raster").
        target_data_type: "vector" for classification, "raster" for segmentation.
        target_bands: List of band names for raster targets (e.g., ["label"]).
        crop_size: If set, apply Crop transform to this size.
        pad_size: If set, apply Pad transform to this size (center mode).

    Returns:
        RsModelDataset: A dataset object ready for training or evaluation.
    """
    if not layers:
        raise ValueError(
            f"`layers` must be a non-empty list of rslearn layer names, "
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

    # Build image selectors for transforms (input modalities + target selectors)
    # For plain SegmentationTask, targets are at "target/classes" and "target/valid" (not nested)
    # Note: MultiTask would nest under task name, but we use plain SegmentationTask here
    input_selectors = list(layers_by_olmoearth.keys())
    if task_type == "segmentation":
        target_selectors = ["target/classes", "target/valid"]
    else:
        target_selectors = []
    all_selectors = input_selectors + target_selectors

    transforms = []
    # Apply crop transform if specified
    if crop_size is not None:
        transforms.append(
            RsCrop(
                crop_size=crop_size,
                image_selectors=all_selectors,
            )
        )
    # Apply pad transform if specified
    if pad_size is not None:
        transforms.append(
            RsPad(
                size=pad_size,
                mode="center",
                image_selectors=all_selectors,
            )
        )

    inputs: dict[str, RsDataInput] = {}
    # NOTE: Some datasets use layer suffixes (sentinel2.1, sentinel2.2, ...) for timesteps,
    # others use load_all_item_groups=True with a single layer name. The rslearn DataInput
    # handles this via load_all_layers and load_all_item_groups options.
    # passthrough=False so rslearn converts to tensors (not RasterImage objects)
    for olmoearth_key, per_key_layers in layers_by_olmoearth.items():
        inputs[olmoearth_key] = RsDataInput(
            data_type="raster",
            layers=per_key_layers,
            bands=bands_by_olmoearth[olmoearth_key],
            passthrough=True,  # Must be True so data ends up in input_dict for transforms
            load_all_layers=True,
            # TODO: This needs to be configurable in the dataset entry but we hardcode for now
            load_all_item_groups=True,
        )

    # Build target input based on task type
    target_input_kwargs: dict[str, Any] = {
        "data_type": target_data_type,
        "layers": [target_layer_name],
        "is_target": True,
    }
    if target_bands:
        target_input_kwargs["bands"] = target_bands
    inputs["targets"] = RsDataInput(**target_input_kwargs)

    split_config = RsSplitConfig(
        transforms=transforms,
        groups=rslearn_dataset_groups,
        skip_targets=skip_targets,
        tags={split_tag_key: split} if split else {},
        num_samples=max_samples,
    )

    # Build task based on task_type
    if task_type == "segmentation":
        task = RsSegmentationTask(
            num_classes=len(classes),
            zero_is_invalid=True, # TODO: This should be passed in as well
        )
    else:
        # Default to classification
        task = RsClassificationTask(
            property_name=property_name,
            classes=classes,
        )

    return RsModelDataset(
        dataset=rslearn_dataset,
        split_config=split_config,
        inputs=inputs,
        task=task,
        workers=32,
        use_index=True,
        # refresh_index=True,
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
        split: str = "train",
        property_name: str = "category",
        partition: str = "default",
        norm_stats_from_pretrained: bool = True,
        norm_method: str = "norm_no_clip",
        ds_norm_stats_json: str | None = None,
        start_time: str = "2022-09-01",
        end_time: str = "2023-09-01",
        task_type: str = "classification",
        max_samples: int | None = None,
        split_tag_key: str = "split",
        target_layer_name: str = "label",
        target_data_type: str = "vector",
        target_bands: list[str] | None = None,
        crop_size: int | None = None,
        pad_size: int | None = None,
        num_timesteps: int = 12,
    ):
        """Initialize RslearnToOlmoEarthDataset.

        Args:
            max_samples: If set, limits the dataset to this many samples (for fast debugging).
            split_tag_key: The tag key used to identify splits (e.g., "split" or "helios_split").
            target_layer_name: rslearn layer name for targets (e.g., "label", "label_raster").
            target_data_type: "vector" for classification, "raster" for segmentation.
            target_bands: List of band names for raster targets.
            crop_size: If set, apply Crop transform to this size.
            pad_size: If set, apply Pad transform to this size (center mode).
            num_timesteps: Number of timesteps per sample.
        """
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
            raise ValueError(f"Input modalities must be in {self.allowed_modalities} but got {input_modalities}")

        # Build rslearn ModelDataset for the split
        print(f"Building rslearn ModelDataset for {ds_path}")
        dataset = RslearnDataset(UPath(ds_path))
        self.dataset = build_rslearn_model_dataset(
            rslearn_dataset=dataset,
            rslearn_dataset_groups=ds_groups,
            layers=layers,
            classes=classes,
            task_type=task_type,
            split="val" if split == "valid" else split,  # rslearn uses 'val'
            property_name=property_name,
            max_samples=max_samples,
            split_tag_key=split_tag_key,
            target_layer_name=target_layer_name,
            target_data_type=target_data_type,
            target_bands=target_bands,
            crop_size=crop_size,
            pad_size=pad_size,
        )

        self.norm_stats_from_pretrained = norm_stats_from_pretrained
        self.input_modalities = input_modalities
        self.num_timesteps = num_timesteps
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
        #TODO: We will need to use the registry to get this information.
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
        T = self.num_timesteps

        for modality in self.input_modalities:
            if modality not in input_dict:
                raise ValueError(f"Modality {modality} not found in dataset inputs")
            num_bands = DataModality.get(modality).num_bands
            x = input_dict[modality]
            print(f"x shape: {x.shape} for modality {modality} ")
            # Extract tensor and rearrange to OlmoEarth format (H, W, T, C)
            if isinstance(x, RasterImage):
                # RasterImage.image has shape (C, T, H, W)
                x = rearrange(x.image, "c t h w -> h w t c")
            else:
                # Tensor with shape (T*C, H, W) from rslearn
                if not isinstance(x, np.ndarray):
                    print("using tensor")
                    x = x.numpy()
                x = rearrange(x, "(t c) h w -> h w t c", t=T, c=num_bands)

            # Convert to dB for Sentinel-1
            if modality == DataModality.SENTINEL1.name:
                x = convert_to_db(x)

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

        #TODO: Pass through zero is invalid to set this or not
        # Extract target classes and valid mask
        # For segmentation, rslearn provides both "classes" and "valid" in target dict
        #TODO: Pass through zero is invalid to set this or not
        classes = target["classes"].long()
        valid = target.get("valid", torch.ones_like(classes, dtype=torch.float32))

        # Set invalid pixels to SEGMENTATION_IGNORE_LABEL (excluded from loss/metrics)
        classes = torch.where(
            valid > 0, classes, torch.tensor(SEGMENTATION_IGNORE_LABEL, dtype=classes.dtype)
        )

        return masked_sample, classes

#Do we need crop size or something else passed through

def from_registry_entry(
    entry: EvalDatasetEntry,
    split: str = "train",
    input_layers: list[str] | None = None,
    partition: str = "default",
    norm_method: str = "norm_no_clip",
    norm_stats_from_pretrained: bool | None = None,
    max_samples: int | None = None,
    input_modalities_override: list[str] | None = None,
) -> RslearnToOlmoEarthDataset:
    """Build RslearnToOlmoEarthDataset from a registry EvalDatasetEntry.

    Args:
        entry: Registry entry containing dataset metadata.
        split: Dataset split to load ("train", "val", "test").
        input_layers: Optional rslearn layer names. If None, derived from entry modalities.
        partition: Dataset partition (e.g., "default", "0.10x_train").
        norm_method: Normalization method when not using pretrain stats.
        norm_stats_from_pretrained: Override for entry.use_pretrain_norm.
        input_modalities_override: Override modalities from entry. For multi-modal datasets,
            allows using only a subset (e.g., just S1 or just S2).

    Returns:
        Configured RslearnToOlmoEarthDataset instance.

    Example:
        from olmoearth_pretrain.evals.studio_ingest import get_dataset_entry

        entry = get_dataset_entry("tolbi_crops")
        dataset = from_registry_entry(entry, split="val")
    """
    # Use override if provided, otherwise use modalities from entry
    if input_modalities_override:
        input_modalities = [m.lower() for m in input_modalities_override]
    else:
        input_modalities = [m.lower() for m in entry.modalities]

    # Derive rslearn layer names from modalities if not provided
    if not input_layers:
        input_layers = [OLMOEARTH_TO_RSLEARN.get(m, m) for m in input_modalities]

    # Get classes as list of strings
    classes = entry.classes if entry.classes else [str(i) for i in range(entry.num_classes or 0)]

    # Parse temporal range
    start_time, end_time = entry.temporal_range
    # TODO: This will need to be added to the registry.
    if not start_time:
        start_time = "2022-09-01"  # default
    if not end_time:
        end_time = "2023-09-01"  # default

    # Use override if provided, otherwise use entry's setting
    use_pretrain_norm = norm_stats_from_pretrained if norm_stats_from_pretrained is not None else entry.use_pretrain_norm

    # Determine norm stats path
    ds_norm_stats_json = None
    if not use_pretrain_norm and entry.norm_stats_path:
        ds_norm_stats_json = entry.norm_stats_path

    # Select sizing based on split: train uses train_* sizing, val/test uses eval_* sizing
    if split == "train":
        crop_size = entry.train_crop_size
        pad_size = entry.train_pad_size
    else:
        crop_size = entry.eval_crop_size
        pad_size = entry.eval_pad_size

    return RslearnToOlmoEarthDataset(
        ds_path=entry.source_path,
        layers=input_layers,
        classes=classes,
        input_modalities=input_modalities,
        ds_groups=entry.groups if entry.groups else None,
        split=split,
        property_name=entry.target_property,
        partition=partition,
        norm_stats_from_pretrained=use_pretrain_norm,
        norm_method=norm_method,
        ds_norm_stats_json=ds_norm_stats_json,
        start_time=start_time,
        end_time=end_time,
        task_type=entry.task_type,
        max_samples=max_samples,
        split_tag_key=entry.split_tag_key,
        target_layer_name=entry.target_layer_name,
        target_data_type=entry.target_data_type,
        target_bands=entry.target_bands,
        crop_size=crop_size,
        pad_size=pad_size,
        num_timesteps=entry.num_timesteps,
    )

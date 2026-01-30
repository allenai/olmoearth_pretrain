"""Convert rslearn dataset to OlmoEarth Pretrain evaluation dataset format."""

from __future__ import annotations

import json
from datetime import datetime
from importlib.resources import files
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from olmoearth_pretrain.evals.studio_ingest.schema import EvalDatasetEntry
    from olmoearth_pretrain.evals.datasets.rslearn_builder import RuntimeConfig

import numpy as np
import torch
from dateutil.relativedelta import relativedelta
from einops import rearrange
from rslearn.train.dataset import ModelDataset as RsModelDataset
from rslearn.train.model_context import RasterImage
from torch.utils.data import Dataset

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


def get_timestamps(
    start_time: str,
    end_time: str,
    num_timesteps: int | None = None,
) -> list[torch.Tensor]:
    """Return monthly (day, month0, year) long tensors for the specified range.

    Args:
        start_time: Start date in YYYY-MM-DD format.
        end_time: End date in YYYY-MM-DD format.
        num_timesteps: Number of timesteps to generate. If None, uses YEAR_NUM_TIMESTEPS.

    Returns:
        List of tensors, each containing [day, month (0-indexed), year].
    """
    if num_timesteps is None:
        num_timesteps = YEAR_NUM_TIMESTEPS

    start = datetime.strptime(start_time, "%Y-%m-%d").replace(day=1)
    end = datetime.strptime(end_time, "%Y-%m-%d")

    months_diff = (end.year - start.year) * 12 + (end.month - start.month) + 1
    if months_diff < num_timesteps:
        raise ValueError(
            f"Not enough months in range ({months_diff}) to cover {num_timesteps}"
        )

    dates: list[torch.Tensor] = []
    cur = start
    while cur <= end and len(dates) < num_timesteps:
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

    Requires a pre-built ModelDataset from RuntimeConfig (via jsonargparse).
    Use from_runtime_config() or from_registry_entry() to construct.
    """

    allowed_modalities = {
        DataModality.SENTINEL2_L2A.name,
        DataModality.SENTINEL1.name,
        DataModality.LANDSAT.name,
    }

    def __init__(
        self,
        model_dataset: RsModelDataset,
        input_modalities: list[str],
        target_task_name: str | None = None,
        target_task_type: str = "segmentation",
        norm_stats_from_pretrained: bool = True,
        norm_method: str = "norm_no_clip",
        ds_norm_stats_json: str | None = None,
        start_time: str = "2022-09-01",
        end_time: str = "2023-09-01",
        num_timesteps: int = 12,
    ):
        """Initialize RslearnToOlmoEarthDataset.

        Args:
            model_dataset: Pre-built rslearn ModelDataset (from RuntimeConfig).
            input_modalities: OlmoEarth modality names (e.g., ["sentinel2_l2a"]).
            target_task_name: For MultiTask, the sub-task name (e.g., "segment").
                If None, assumes single task and accesses target dict directly.
            target_task_type: Type of task ("segmentation", "classification", "regression").
                Determines how to parse the target dict.
            norm_stats_from_pretrained: Use pretrain normalization stats.
            norm_method: Normalization method when not using pretrain stats.
            ds_norm_stats_json: Path to dataset norm stats JSON.
            start_time: Start time for timestamp generation.
            end_time: End time for timestamp generation.
            num_timesteps: Number of timesteps per sample.
        """
        if not norm_stats_from_pretrained and ds_norm_stats_json is None:
            raise ValueError(
                "norm_stats_from_pretrained=False requires a JSON file with dataset stats "
                "(set ds_norm_stats_json)."
            )

        if not input_modalities:
            raise ValueError("Must specify at least one input modality")
        if not all(m in self.allowed_modalities for m in input_modalities):
            raise ValueError(f"Input modalities must be in {self.allowed_modalities} but got {input_modalities}")

        self.dataset = model_dataset
        self.norm_stats_from_pretrained = norm_stats_from_pretrained
        self.input_modalities = input_modalities
        print(f"Input modalities: {self.input_modalities}")

        # Store temporal config for per-sample timestamp generation
        self.start_time = start_time
        self.end_time = end_time
        self.max_timesteps = num_timesteps  # Max expected timesteps (for validation)

        # Target parsing config - derived from Task structure
        self.target_task_name = target_task_name  # For MultiTask, e.g., "segment"
        self.target_task_type = target_task_type  # "segmentation", "classification", etc.

        if self.norm_stats_from_pretrained:
            from olmoearth_pretrain.data.normalize import Normalizer, Strategy

            self.normalizer_computed = Normalizer(Strategy.COMPUTED)
        else:
            self.dataset_norm_stats = self._get_norm_stats(ds_norm_stats_json)  # type: ignore
            self.norm_method = norm_method

    @classmethod
    def from_runtime_config(
        cls,
        runtime_config: "RuntimeConfig",
        source_path: str,
        split: str = "val",
        input_modalities: list[str] | None = None,
        norm_stats_from_pretrained: bool = True,
        norm_method: str = "norm_no_clip",
        ds_norm_stats_json: str | None = None,
        start_time: str = "2022-09-01",
        end_time: str = "2023-09-01",
        max_samples: int | None = None,
        num_timesteps: int = 12,
    ) -> "RslearnToOlmoEarthDataset":
        """Build from RuntimeConfig using jsonargparse-instantiated objects.

        This is the main way to build datasets from model.yaml.

        Args:
            runtime_config: RuntimeConfig with parsed model.yaml.
            source_path: Path to rslearn dataset.
            split: Dataset split ("train", "val", "test").
            input_modalities: OlmoEarth modality names. If None, derived from config.
            norm_stats_from_pretrained: Use pretrain norm stats.
            norm_method: Normalization method.
            ds_norm_stats_json: Path to dataset norm stats.
            start_time: Start time for timestamps (used for timestamp generation).
            end_time: End time for timestamps (used for timestamp generation).
            max_samples: Optional sample limit.
            num_timesteps: Max expected timesteps from config (actual per-sample
                timesteps are derived from data).

        Returns:
            RslearnToOlmoEarthDataset instance.
        """
        from olmoearth_pretrain.evals.datasets.rslearn_builder import (
            build_model_dataset_from_config,
        )

        # Build ModelDataset using jsonargparse
        print(f"Building ModelDataset from RuntimeConfig for {source_path}")
        model_dataset = build_model_dataset_from_config(
            runtime_config=runtime_config,
            source_path=source_path,
            split=split,
            max_samples=max_samples,
        )

        # Derive input modalities from runtime config if not provided
        if input_modalities is None:
            modality_layers = runtime_config.get_modality_layers()
            # Map rslearn layer names to OlmoEarth modality names
            input_modalities = []
            for layer in modality_layers:
                if layer in RSLEARN_TO_OLMOEARTH:
                    olmoearth_name, _ = RSLEARN_TO_OLMOEARTH[layer]
                    input_modalities.append(olmoearth_name)
                else:
                    input_modalities.append(layer)

        # Get task structure info for parsing targets
        task_info = runtime_config.get_task_info()
        print(f"Task info: {task_info}")

        return cls(
            model_dataset=model_dataset,
            input_modalities=input_modalities,
            target_task_name=task_info["task_name"],
            target_task_type=task_info["task_type"],
            norm_stats_from_pretrained=norm_stats_from_pretrained,
            norm_method=norm_method,
            ds_norm_stats_json=ds_norm_stats_json,
            start_time=start_time,
            end_time=end_time,
            num_timesteps=num_timesteps,
        )

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
        sample_timesteps: int | None = None  # Will be set from actual data

        for modality in self.input_modalities:
            if modality not in input_dict:
                raise ValueError(f"Modality {modality} not found in dataset inputs")
            num_bands = DataModality.get(modality).num_bands
            x = input_dict[modality]
            print(f"x shape: {x.shape} for modality {modality} ")
            # Extract tensor and rearrange to OlmoEarth format (H, W, T, C)
            if isinstance(x, RasterImage):
                # RasterImage.image has shape (C, T, H, W)
                img = x.image
                if isinstance(img, torch.Tensor):
                    img = img.numpy()
                x = rearrange(img, "c t h w -> h w t c")
            else:
                # Tensor with shape (T*C, H, W) from rslearn
                if isinstance(x, torch.Tensor):
                    x = x.numpy()
                # Infer T from the actual data shape
                actual_t = x.shape[0] // num_bands
                x = rearrange(x, "(t c) h w -> h w t c", t=actual_t, c=num_bands)

            # Track actual timesteps from data (should be consistent across modalities)
            if sample_timesteps is None:
                sample_timesteps = x.shape[2]

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
            print(f"x shape: {x.shape} for modality {modality} after normalization")
            sample_dict[modality] = torch.as_tensor(x, dtype=torch.float32)

        # Generate timestamps for this sample's actual number of timesteps
        sample_timesteps = sample_timesteps or self.max_timesteps
        timestamps = get_timestamps(self.start_time, self.end_time, num_timesteps=sample_timesteps)
        sample_dict["timestamps"] = torch.stack(timestamps)

        olmoearth_sample = OlmoEarthSample(**sample_dict)
        masked_sample = MaskedOlmoEarthSample.from_olmoearthsample(olmoearth_sample)
        # ensure modality and modality mask have same hw raise error if not
        # Error is likely padding the mask wrong maybe or something?
        from olmoearth_pretrain.data.constants import Modality
        for modality in self.input_modalities:
            modality_spec = Modality.get(modality)
            if modality_spec.is_spatial:
                mask_attr_name = MaskedOlmoEarthSample.get_masked_modality_name(modality)
                masked_attr = getattr(masked_sample, mask_attr_name)
                if masked_attr is None:
                    raise ValueError(f"Modality mask {mask_attr_name} not found for modality {modality}")
                # hw is only dims 1 and 2
                if masked_attr.shape[1:3] != sample_dict[modality].shape[1:3]:
                    raise ValueError(
                        f"Modality mask {mask_attr_name} and modality {modality} have different hw shapes: "
                        f"{masked_attr.shape[1:3]} != {sample_dict[modality].shape[1:3]}"
                    )
        # For MultiTask: target[task_name] contains the sub-task's output
        # For single Task: target contains the output directly
        if self.target_task_name:
            # MultiTask - access sub-task by name
            data_dict = target.get(self.target_task_name, {})
        else:
            # Single task - target dict is the data directly
            data_dict = target

        # Parse target based on task type
        # - SegmentationTask: {"classes": RasterImage, "valid": RasterImage}
        # - ClassificationTask: {"class": tensor, "valid": tensor}
        # - RegressionTask: {"value": tensor, "valid": tensor}
        if self.target_task_type == "segmentation":
            classes_raw = data_dict.get("classes", None)
            valid_raw = data_dict.get("valid", None)
        elif self.target_task_type == "classification":
            classes_raw = data_dict.get("class", None)
            valid_raw = data_dict.get("valid", None)
        else:
            # Default fallback
            classes_raw = data_dict.get("classes", data_dict.get("class", None))
            valid_raw = data_dict.get("valid", None)

        # Extract tensors from RasterImage if needed
        # rslearn tasks wrap outputs in RasterImage with shape (1, 1, H, W)
        classes = self._extract_target_tensor(classes_raw)
        valid = self._extract_target_tensor(valid_raw)

        # actually we want to map valid onto target and fill with the ignore label index and not return valid
        if valid is not None:
            # what is the right value to fill with?
            classes = classes.masked_fill(valid == 0, SEGMENTATION_IGNORE_LABEL)
        return masked_sample, classes

    def _extract_target_tensor(self, raw: Any) -> torch.Tensor | None:
        """Extract tensor from RasterImage or return tensor directly."""
        if raw is None:
            return None
        if isinstance(raw, RasterImage):
            # RasterImage.image has shape (C, T, H, W), squeeze to (H, W)
            arr = raw.image.squeeze()  # Remove singleton dims
            return torch.as_tensor(arr, dtype=torch.long)
        if isinstance(raw, torch.Tensor):
            return raw
        if isinstance(raw, np.ndarray):
            return torch.as_tensor(raw, dtype=torch.long)
        # Fallback - try to convert
        return torch.as_tensor(raw)

def from_registry_entry(
    entry: EvalDatasetEntry,
    split: str = "train",
    norm_method: str = "norm_no_clip",
    norm_stats_from_pretrained: bool | None = None,
    max_samples: int | None = None,
    input_modalities_override: list[str] | None = None,
) -> RslearnToOlmoEarthDataset:
    """Build RslearnToOlmoEarthDataset from a registry EvalDatasetEntry.

    Uses jsonargparse to build ModelDataset directly from model.yaml.
    Requires entry.model_config_path to be set.

    Args:
        entry: Registry entry containing dataset metadata.
        split: Dataset split to load ("train", "val", "test").
        norm_method: Normalization method when not using pretrain stats.
        norm_stats_from_pretrained: Override for entry.use_pretrain_norm.
        max_samples: Optional limit on number of samples.
        input_modalities_override: Override modalities from entry. For multi-modal datasets,
            allows using only a subset (e.g., just S1 or just S2).

    Returns:
        Configured RslearnToOlmoEarthDataset instance.

    Raises:
        ValueError: If entry.model_config_path is not set.

    Example:
        from olmoearth_pretrain.evals.studio_ingest import get_dataset_entry

        entry = get_dataset_entry("tolbi_crops")
        dataset = from_registry_entry(entry, split="val")
    """
    import logging
    log = logging.getLogger(__name__)

    if not entry.model_config_path:
        raise ValueError(
            f"Registry entry '{entry.name}' has no model_config_path. "
            "Cannot build dataset without model.yaml. "
            "Re-ingest with olmoearth_run_config_path set."
        )

    # Use override if provided, otherwise use modalities from entry
    if input_modalities_override:
        input_modalities = [m.lower() for m in input_modalities_override]
    else:
        input_modalities = [m.lower() for m in entry.modalities]

    # Use override if provided, otherwise use entry's setting
    use_pretrain_norm = norm_stats_from_pretrained if norm_stats_from_pretrained is not None else entry.use_pretrain_norm

    # Determine norm stats path
    ds_norm_stats_json = None
    if not use_pretrain_norm and entry.norm_stats_path:
        ds_norm_stats_json = entry.norm_stats_path

    # Get temporal range from entry
    start_time, end_time = entry.temporal_range
    if not start_time:
        start_time = "2022-09-01"
    if not end_time:
        end_time = "2023-09-01"

    # Load runtime config and build dataset
    from olmoearth_pretrain.evals.datasets.rslearn_builder import load_runtime_config

    model_yaml_path = f"{entry.model_config_path}/model.yaml"
    log.info(f"Loading RuntimeConfig from {model_yaml_path}")
    runtime_config = load_runtime_config(model_yaml_path, entry.source_path)

    if not runtime_config.model_config:
        raise ValueError(
            f"Failed to load model.yaml from {model_yaml_path}. "
            "Check that the file exists and is valid YAML."
        )

    log.info(f"Building dataset from RuntimeConfig for {entry.name}")
    return RslearnToOlmoEarthDataset.from_runtime_config(
        runtime_config=runtime_config,
        source_path=entry.source_path,
        split="val" if split == "valid" else split,
        input_modalities=input_modalities,
        norm_stats_from_pretrained=use_pretrain_norm,
        norm_method=norm_method,
        ds_norm_stats_json=ds_norm_stats_json,
        start_time=start_time,
        end_time=end_time,
        max_samples=max_samples,
        num_timesteps=entry.num_timesteps,
    )

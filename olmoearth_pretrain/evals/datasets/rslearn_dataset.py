"""Convert rslearn dataset to OlmoEarth Pretrain evaluation dataset format."""

from __future__ import annotations

import json
from datetime import datetime
from importlib.resources import files
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from olmoearth_pretrain.evals.datasets.rslearn_builder import RuntimeConfig
    from olmoearth_pretrain.evals.studio_ingest.schema import EvalDatasetEntry

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
from olmoearth_pretrain.evals.constants import RSLEARN_TO_OLMOEARTH
from olmoearth_pretrain.evals.metrics import SEGMENTATION_IGNORE_LABEL
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample, OlmoEarthSample

from .normalize import normalize_bands


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
        # Default to 2std no clip - this matches what our model sees in pretraining,
        # so when using dataset stats (e.g. for MADOS) consistency is important.
        norm_method: str = "norm_no_clip_2_std",
        ds_norm_stats_json: str | None = None,
        ds_norm_stats: dict[str, Any] | None = None,
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
            ds_norm_stats: Dataset norm stats blob (e.g. from registry entry).
            start_time: Start time for timestamp generation.
            end_time: End time for timestamp generation.
            num_timesteps: Number of timesteps per sample.
        """
        if (
            not norm_stats_from_pretrained
            and ds_norm_stats_json is None
            and ds_norm_stats is None
        ):
            raise ValueError(
                "norm_stats_from_pretrained=False requires a JSON file with dataset stats "
                "or registry stats (set ds_norm_stats_json or ds_norm_stats)."
            )

        if not input_modalities:
            raise ValueError("Must specify at least one input modality")
        if not all(m in self.allowed_modalities for m in input_modalities):
            raise ValueError(
                f"Input modalities must be in {self.allowed_modalities} but got {input_modalities}"
            )

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
        self.target_task_type = (
            target_task_type  # "segmentation", "classification", etc.
        )

        if self.norm_stats_from_pretrained:
            from olmoearth_pretrain.data.normalize import Normalizer, Strategy

            self.normalizer_computed = Normalizer(Strategy.COMPUTED)
        else:
            if ds_norm_stats is not None:
                self.dataset_norm_stats = self._parse_norm_stats(ds_norm_stats)
            else:
                self.dataset_norm_stats = self._get_norm_stats(ds_norm_stats_json)  # type: ignore[arg-type]
            self.norm_method = norm_method

    @classmethod
    def from_runtime_config(
        cls,
        runtime_config: RuntimeConfig,
        source_path: str,
        split: str = "val",
        input_modalities: list[str] | None = None,
        norm_stats_from_pretrained: bool = True,
        norm_method: str = "norm_no_clip",
        ds_norm_stats_json: str | None = None,
        ds_norm_stats: dict[str, Any] | None = None,
        start_time: str = "2022-09-01",
        end_time: str = "2023-09-01",
        max_samples: int | None = None,
        num_timesteps: int = 12,
        groups_override: list[str] | None = None,
        tags_override: dict[str, str] | None = None,
    ) -> RslearnToOlmoEarthDataset:
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
            ds_norm_stats: Dataset norm stats blob (e.g. from registry entry).
            start_time: Start time for timestamps (used for timestamp generation).
            end_time: End time for timestamps (used for timestamp generation).
            max_samples: Optional sample limit.
            num_timesteps: Max expected timesteps from config (actual per-sample
                timesteps are derived from data).
            groups_override: Optional list of groups to use instead of model.yaml groups.
            tags_override: Optional dict of tags to filter windows (e.g., {"eval_split": "val"}).

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
            groups_override=groups_override,
            tags_override=tags_override,
        )

        # Derive input modalities from runtime config if not provided
        if input_modalities is None:
            modality_layers = runtime_config.get_modality_layers()
            # Map rslearn layer names to OlmoEarth modality names.
            # Also handles "pre_"/"post_" prefixed names for compatibility
            # with older datasets.
            input_modalities = []
            for layer in modality_layers:
                resolved = layer
                if layer not in RSLEARN_TO_OLMOEARTH:
                    for prefix in ("pre_", "post_"):
                        if (
                            layer.startswith(prefix)
                            and layer[len(prefix) :] in RSLEARN_TO_OLMOEARTH
                        ):
                            resolved = layer[len(prefix) :]
                            break
                if resolved in RSLEARN_TO_OLMOEARTH:
                    input_modalities.append(RSLEARN_TO_OLMOEARTH[resolved].name)
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
            ds_norm_stats=ds_norm_stats,
            start_time=start_time,
            end_time=end_time,
            num_timesteps=num_timesteps,
        )

    @staticmethod
    def _parse_norm_stats(
        raw_stats: dict[str, Any],
    ) -> dict[str, dict[str, np.ndarray]]:
        """Convert raw stats into modality arrays keyed by band order."""
        out: dict[str, dict[str, np.ndarray]] = {}
        for modality, per_band in raw_stats.items():
            modality_name = modality.lower()
            band_order = DataModality.get(modality_name).band_order

            # Also support pre-aggregated format: {"means": [...], "stds": [...], ...}
            if all(
                key in per_band for key in ("means", "stds", "mins", "maxs")
            ) and isinstance(per_band.get("means"), list | tuple):
                means = np.array(per_band["means"], dtype=np.float32)
                stds = np.array(per_band["stds"], dtype=np.float32)
                mins = np.array(per_band["mins"], dtype=np.float32)
                maxs = np.array(per_band["maxs"], dtype=np.float32)
                if not (
                    len(means) == len(stds) == len(mins) == len(maxs) == len(band_order)
                ):
                    raise ValueError(
                        f"Invalid aggregated norm stats for modality {modality_name}: "
                        f"expected {len(band_order)} bands, got "
                        f"{len(means)}, {len(stds)}, {len(mins)}, {len(maxs)}"
                    )
                out[modality_name] = {
                    "means": means,
                    "stds": stds,
                    "mins": mins,
                    "maxs": maxs,
                }
                continue

            means, stds, mins, maxs = [], [], [], []
            for band in band_order:
                band_stats = (
                    per_band.get(band)
                    or per_band.get(band.upper())
                    or per_band.get(band.lower())
                )
                if band_stats is None:
                    raise ValueError(
                        f"Missing stats for {band} in modality {modality_name}"
                    )
                means.append(band_stats["mean"])
                stds.append(band_stats["std"])
                mins.append(band_stats["min"])
                maxs.append(band_stats["max"])

            out[modality_name] = {
                "means": np.array(means, dtype=np.float32),
                "stds": np.array(stds, dtype=np.float32),
                "mins": np.array(mins, dtype=np.float32),
                "maxs": np.array(maxs, dtype=np.float32),
            }
        return out

    @staticmethod
    def _get_norm_stats(ds_norm_stats_json: str) -> dict:
        """Load dataset norm stats from a JSON file."""
        with (
            files("olmoearth_pretrain.evals.datasets.config") / ds_norm_stats_json
        ).open() as f:
            blob = json.load(f)
        return RslearnToOlmoEarthDataset._parse_norm_stats(blob)

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
            # print(f"x shape: {x.shape} for modality {modality} ")
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
            # print(f"x shape: {x.shape} for modality {modality} after normalization")
            sample_dict[modality] = torch.as_tensor(x, dtype=torch.float32)

        # TODO: WE should be reading this from the metadata.json of each window/is there a way to enable in rslearn
        # Generate timestamps for this sample's actual number of timesteps
        sample_timesteps = sample_timesteps or self.max_timesteps
        timestamps = get_timestamps(
            self.start_time, self.end_time, num_timesteps=sample_timesteps
        )
        sample_dict["timestamps"] = torch.stack(timestamps)

        olmoearth_sample = OlmoEarthSample(**sample_dict)
        masked_sample = MaskedOlmoEarthSample.from_olmoearthsample(olmoearth_sample)
        # ensure modality and modality mask have same hw raise error if not
        # Error is likely padding the mask wrong maybe or something?
        from olmoearth_pretrain.data.constants import Modality

        for modality in self.input_modalities:
            modality_spec = Modality.get(modality)
            if modality_spec.is_spatial:
                mask_attr_name = MaskedOlmoEarthSample.get_masked_modality_name(
                    modality
                )
                masked_attr = getattr(masked_sample, mask_attr_name)
                if masked_attr is None:
                    raise ValueError(
                        f"Modality mask {mask_attr_name} not found for modality {modality}"
                    )
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

        if valid is not None:
            assert classes is not None, "valid mask present but no classes tensor"
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
    groups_override: list[str] | None = None,
    tags_override: dict[str, str] | None = None,
) -> RslearnToOlmoEarthDataset:
    """Build RslearnToOlmoEarthDataset from a registry EvalDatasetEntry.

    Uses jsonargparse to build ModelDataset directly from model.yaml.
    Requires model.yaml at entry.weka_path/model.yaml (set during ingestion).

    Uses the split tags written during ingestion to filter windows by default.

    Args:
        entry: Registry entry containing dataset metadata.
        split: Dataset split to load ("train", "val", "valid", "test").
        norm_method: Normalization method when not using pretrain stats.
        norm_stats_from_pretrained: Override for entry.use_pretrain_norm.
        max_samples: Optional limit on number of samples.
        input_modalities_override: Override modalities from entry. For multi-modal datasets,
            allows using only a subset (e.g., just S1 or just S2).
        groups_override: Override groups. If None, no group filtering is applied.
        tags_override: Override tags. If None, uses entry.split_tag_key with the
            appropriate split value (e.g., {"eval_split": "val"}).

    Returns:
        Configured RslearnToOlmoEarthDataset instance.

    Raises:
        ValueError: If entry has no weka_path.

    Example:
        from olmoearth_pretrain.evals.studio_ingest import get_dataset_entry

        entry = get_dataset_entry("tolbi_crops")
        dataset = from_registry_entry(entry, split="val")
    """
    import logging

    log = logging.getLogger(__name__)

    dataset_path = entry.weka_path if entry.weka_path else entry.source_path
    if not dataset_path:
        raise ValueError(f"Entry '{entry.name}' has no weka_path or source_path.")

    if not entry.weka_path:
        raise ValueError(
            f"Registry entry '{entry.name}' has no weka_path. "
            "model.yaml must be at weka_path/model.yaml. Run migrate_model_yaml or re-ingest."
        )

    model_yaml_path = f"{entry.weka_path}/model.yaml"

    # Use override if provided, otherwise use modalities from entry
    if input_modalities_override:
        input_modalities = [m.lower() for m in input_modalities_override]
    else:
        input_modalities = [m.lower() for m in entry.modalities]

    # Use override if provided, otherwise use entry's setting
    use_pretrain_norm = (
        norm_stats_from_pretrained
        if norm_stats_from_pretrained is not None
        else entry.use_pretrain_norm
    )

    # Normalize split name: "valid" -> "val"
    normalized_split = "val" if split == "valid" else split

    # Resolve splits based on split_type from ingestion.
    # "tags": all windows may live under one group dir, split by metadata tag
    # "groups": windows are in separate group dirs (windows/train/, windows/val/, etc.)
    split_value_map = {
        "train": entry.train_split,
        "val": entry.val_split,
        "test": entry.test_split,
    }
    split_value = split_value_map.get(normalized_split, normalized_split)

    effective_tags = tags_override
    if effective_tags is None:
        if entry.split_type == "tags" and entry.split_tag_key:
            effective_tags = {entry.split_tag_key: split_value}
            # Clear groups so rslearn scans all dirs, then filters by tag
            if groups_override is None:
                groups_override = []
            log.info(f"Using tag-based splits: {entry.split_tag_key}={split_value}")
        elif entry.split_type == "groups":
            # Use split name as the group directory
            if groups_override is None:
                groups_override = [split_value]
            log.info(f"Using group-based splits: groups={groups_override}")

    # Load runtime config and build dataset
    from olmoearth_pretrain.evals.datasets.rslearn_builder import load_runtime_config

    log.info(f"Loading RuntimeConfig from {model_yaml_path}")
    runtime_config = load_runtime_config(model_yaml_path, dataset_path)

    if not runtime_config.model_config:
        raise ValueError(
            f"Failed to load model.yaml from {model_yaml_path}. "
            "Check that the file exists and is valid YAML."
        )

    log.info(
        f"Building dataset from RuntimeConfig for {entry.name} (path: {dataset_path})"
    )
    if not use_pretrain_norm and not entry.norm_stats:
        raise ValueError(
            f"Dataset '{entry.name}' has use_pretrain_norm=False but no norm_stats in registry."
        )
    return RslearnToOlmoEarthDataset.from_runtime_config(
        runtime_config=runtime_config,
        source_path=dataset_path,
        split=normalized_split,
        input_modalities=input_modalities,
        norm_stats_from_pretrained=use_pretrain_norm,
        norm_method=norm_method,
        ds_norm_stats_json=None,  # Not currently used
        ds_norm_stats=entry.norm_stats if not use_pretrain_norm else None,
        max_samples=max_samples,
        groups_override=groups_override,
        tags_override=effective_tags,
    )

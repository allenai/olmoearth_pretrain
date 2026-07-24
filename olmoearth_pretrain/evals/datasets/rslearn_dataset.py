"""Convert rslearn dataset to OlmoEarth Pretrain evaluation dataset format."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime
from importlib.resources import files
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from olmoearth_pretrain.evals.studio_ingest.schema import EvalDatasetEntry

import numpy as np
import torch
from dateutil.relativedelta import relativedelta
from einops import rearrange
from rslearn.train.dataset import ModelDataset as RsModelDataset
from rslearn.train.model_context import RasterImage
from torch.utils.data import Dataset, IterableDataset, Subset

from olmoearth_pretrain.data.constants import (
    EMBEDDING_PRODUCT_MODALITIES,
    YEAR_NUM_TIMESTEPS,
    Modality,
)
from olmoearth_pretrain.data.normalize import Normalizer, Strategy
from olmoearth_pretrain.data.utils import convert_to_db
from olmoearth_pretrain.evals.constants import (
    RSLEARN_TO_OLMOEARTH,
    resolve_rslearn_layer_name,
)
from olmoearth_pretrain.evals.datasets.normalize import NormMethod
from olmoearth_pretrain.evals.datasets.rslearn_builder import (
    build_model_dataset,
    get_modality_layers,
    get_task_info,
    parse_model_config,
)
from olmoearth_pretrain.evals.metrics import SEGMENTATION_IGNORE_LABEL
from olmoearth_pretrain.evals.task_types import TaskType
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample, OlmoEarthSample

from .normalize import normalize_bands

# Fallback imagery time range for timestamp synthesis, used when the registry
# entry does not record the dataset's actual time range.
DEFAULT_START_TIME = "2022-09-01"
DEFAULT_END_TIME = "2023-09-01"


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

    Requires a pre-built ModelDataset (via RslearnDataModule + jsonargparse).
    Use from_model_config() or build_rslearn_eval_dataset() to construct.
    """

    allowed_modalities = {
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
        Modality.LANDSAT.name,
        # Precomputed embedding products, baked in as layers by the embedding
        # materializer (olmoearth_pretrain/evals/embedding_materializer).
        Modality.GSE.name,
        Modality.TESSERA.name,
    }

    def __init__(
        self,
        model_dataset: RsModelDataset,
        input_modalities: list[str],
        target_task_name: str | None = None,
        target_task_type: TaskType | str = TaskType.SEGMENTATION,
        norm_stats_from_pretrained: bool = True,
        norm_method: str = NormMethod.NORM_NO_CLIP_2_STD,
        ds_norm_stats_json: str | None = None,
        ds_norm_stats: dict[str, Any] | None = None,
        start_time: str = DEFAULT_START_TIME,
        end_time: str = DEFAULT_END_TIME,
        num_timesteps: int = 12,
        window_size: int | None = None,
        label_at_center_pixel: bool = False,
        tile_samples: bool = False,
        sample_size: int | None = None,
    ):
        """Initialize RslearnToOlmoEarthDataset.

        Args:
            model_dataset: Pre-built rslearn ModelDataset.
            input_modalities: OlmoEarth modality names (e.g., ["sentinel2_l2a"]).
            target_task_name: For MultiTask, the sub-task name (e.g., "segment").
                If None, assumes single task and accesses target dict directly.
            target_task_type: Type of task ("segmentation" or "classification").
                Determines how to parse the target dict.
            norm_stats_from_pretrained: Use pretrain normalization stats.
            norm_method: Normalization method when not using pretrain stats.
            ds_norm_stats_json: Path to dataset norm stats JSON.
            ds_norm_stats: Dataset norm stats blob (e.g. from registry entry).
            start_time: Start time for timestamp generation.
            end_time: End time for timestamp generation.
            num_timesteps: Number of timesteps per sample.
            window_size: If set, center-crop every sample (imagery and label
                rasters) to window_size x window_size, fixing the spatial
                context each embedding is computed from (per-pixel
                embedding-product convention). Unlike the PASTIS tiling
                window_size, this crops (one window per sample) because these
                datasets carry a single labeled pixel. Segmentation targets
                only.
            label_at_center_pixel: If set, the segmentation label raster is
                reduced to the single labeled pixel's class and the sample is
                emitted as a classification example. The crop (window_size) is
                centered on that labeled pixel, so with use_center_token the
                probe reads exactly the token that carries the label.
                Requires a segmentation target with exactly one labeled pixel
                per sample (extra labeled pixels: the one nearest the raster
                center is used).
            tile_samples: If set (with window_size), every stored sample (and
                its label raster) is tiled into non-overlapping
                window_size x window_size windows at load time — the PASTIS
                dense-label convention — instead of center-cropping one window
                per sample. Requires sample_size, a dense (segmentation or
                per-pixel regression) target, and is mutually exclusive with
                label_at_center_pixel.
            sample_size: Stored sample height/width in pixels (required with
                tile_samples; must be divisible by window_size).
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

        # Store temporal config for per-sample timestamp generation
        self.start_time = start_time
        self.end_time = end_time
        self.max_timesteps = num_timesteps  # Max expected timesteps (for validation)

        # Target parsing config - derived from Task structure
        self.target_task_name = target_task_name  # For MultiTask, e.g., "segment"
        self.target_task_type = TaskType(target_task_type)
        if self.target_task_type not in {
            TaskType.SEGMENTATION,
            TaskType.CLASSIFICATION,
            TaskType.PER_PIXEL_REGRESSION,
            TaskType.WINDOW_REGRESSION,
        }:
            raise ValueError(
                f"Unsupported target task type: {self.target_task_type.value}"
            )

        if (
            window_size is not None or label_at_center_pixel
        ) and self.target_task_type != TaskType.SEGMENTATION:
            raise ValueError(
                "window_size and label_at_center_pixel require a segmentation "
                f"target, got {self.target_task_type.value}"
            )
        self.window_size = window_size
        self.label_at_center_pixel = label_at_center_pixel

        if tile_samples:
            if label_at_center_pixel:
                raise ValueError(
                    "tile_samples and label_at_center_pixel are mutually exclusive"
                )
            if window_size is None or sample_size is None:
                raise ValueError(
                    "tile_samples requires both window_size and sample_size"
                )
            if sample_size % window_size != 0:
                raise ValueError(
                    f"window_size {window_size} must divide sample_size {sample_size}"
                )
        self.sample_size = sample_size
        # Each stored sample yields _tiles_per_side^2 windows (1 = no tiling).
        self._tiles_per_side = (
            sample_size // window_size if tile_samples else 1  # type: ignore[operator]
        )
        # Last (base_idx, (input_dict, target)) loaded in tiled mode. A tiled
        # sample's windows have consecutive indices, so with sequential access
        # this avoids re-reading the full stored sample for every tile. Each
        # DataLoader worker holds its own copy post-fork. Reuse is safe because
        # _transform_sample doesn't mutate its inputs.
        self._cached_base: tuple[int, Any] | None = None

        if self.norm_stats_from_pretrained:
            self.normalizer_computed = Normalizer(Strategy.COMPUTED)
        else:
            if ds_norm_stats is not None:
                self.dataset_norm_stats = self._parse_norm_stats(ds_norm_stats)
            else:
                self.dataset_norm_stats = self._get_norm_stats(ds_norm_stats_json)  # type: ignore[arg-type]
            self.norm_method = norm_method

    @classmethod
    def from_model_config(
        cls,
        model_config: dict[str, Any],
        source_path: str,
        split: str = "val",
        input_modalities: list[str] | None = None,
        norm_stats_from_pretrained: bool = True,
        norm_method: str = NormMethod.NORM_NO_CLIP_2_STD,
        ds_norm_stats_json: str | None = None,
        ds_norm_stats: dict[str, Any] | None = None,
        start_time: str = DEFAULT_START_TIME,
        end_time: str = DEFAULT_END_TIME,
        max_samples: int | None = None,
        num_timesteps: int = 12,
        groups_override: list[str] | None = None,
        tags_override: dict[str, str] | None = None,
        label_fraction: float = 1.0,
        label_fraction_seed: int = 42,
        window_size: int | None = None,
        label_at_center_pixel: bool = False,
        tile_samples: bool = False,
        sample_size: int | None = None,
    ) -> RslearnToOlmoEarthDataset:
        """Build from a parsed model.yaml config dict.

        Uses RslearnDataModule (via jsonargparse) to construct the underlying
        ModelDataset, keeping us in sync with rslearn's config merging logic.

        Args:
            model_config: Parsed model.yaml dict.
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
            tags_override: Optional dict of tags to filter windows.
            label_fraction: Fraction of train labels to use for map-style train
                datasets. Non-train splits always use the full split.
            label_fraction_seed: Seed for the deterministic label_fraction
                subsample so the same low-label subset is used across runs.
            window_size: Center-crop every sample to window_size x window_size
                (see RslearnToOlmoEarthDataset).
            label_at_center_pixel: Emit the labeled pixel's class as a scalar
                classification label (see RslearnToOlmoEarthDataset).
            tile_samples: Tile every sample into window_size x window_size
                windows instead of center-cropping (see
                RslearnToOlmoEarthDataset).
            sample_size: Stored sample height/width, required with tile_samples.
        """
        if not 0 < label_fraction <= 1:
            raise ValueError("label_fraction must be in (0, 1].")
        if label_fraction != 1.0 and split != "train":
            label_fraction = 1.0
        if label_fraction != 1.0 and max_samples is not None:
            raise ValueError("Use either max_samples or label_fraction, not both.")

        model_dataset = build_model_dataset(
            model_config=model_config,
            source_path=source_path,
            split=split,
            max_samples=max_samples,
            groups_override=groups_override,
            tags_override=tags_override,
        )
        if label_fraction != 1.0:
            if isinstance(model_dataset, IterableDataset) or not hasattr(
                model_dataset, "__len__"
            ):
                raise ValueError(
                    "label_fraction is only supported for map-style rslearn train datasets."
                )
            num_samples = max(1, int(len(model_dataset) * label_fraction))
            generator = torch.Generator().manual_seed(label_fraction_seed)
            indices = torch.randperm(len(model_dataset), generator=generator)[
                :num_samples
            ].tolist()
            model_dataset = Subset(model_dataset, indices)

        if input_modalities is None:
            layers = get_modality_layers(model_config)
            input_modalities = []
            for layer in layers:
                resolved = resolve_rslearn_layer_name(layer)
                if resolved is not None:
                    input_modalities.append(RSLEARN_TO_OLMOEARTH[resolved].name)
                else:
                    input_modalities.append(layer)

        task_info = get_task_info(model_config)

        return wrap_rslearn_dataset(
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
            window_size=window_size,
            label_at_center_pixel=label_at_center_pixel,
            tile_samples=tile_samples,
            sample_size=sample_size,
        )

    @staticmethod
    def _parse_norm_stats(
        raw_stats: dict[str, Any],
    ) -> dict[str, dict[str, np.ndarray]]:
        """Convert raw stats into modality arrays keyed by band order."""
        out: dict[str, dict[str, np.ndarray]] = {}
        for modality, per_band in raw_stats.items():
            modality_name = modality.lower()
            band_order = Modality.get(modality_name).band_order

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

    def _locate_labeled_pixel(self, classes: torch.Tensor) -> tuple[int, int]:
        """Locate the labeled pixel in a (H, W) segmentation raster.

        With multiple labeled pixels, the one nearest the raster center wins
        (these datasets carry a single labeled center pixel by construction).
        """
        valid = (classes != SEGMENTATION_IGNORE_LABEL).nonzero()
        if len(valid) == 0:
            raise ValueError(
                "label_at_center_pixel requires at least one labeled pixel, "
                "but the sample's label raster is entirely ignore-labeled"
            )
        h, w = classes.shape
        center = torch.tensor([(h - 1) / 2, (w - 1) / 2])
        distances = ((valid.float() - center) ** 2).sum(dim=1)
        row, col = valid[distances.argmin()].tolist()
        return row, col

    def _label_crop_slices(
        self, classes: torch.Tensor
    ) -> tuple[slice | None, slice | None, torch.Tensor]:
        """Compute crop slices and the emitted label for a segmentation raster.

        Returns (rows, cols, label): rows/cols crop every spatial raster of the
        sample (None = no crop), and label is either the cropped raster or, with
        label_at_center_pixel, the labeled pixel's class as a scalar. The crop
        is centered on the labeled pixel (clamped to raster bounds) so the
        labeled pixel sits at the center token of the cropped window.
        """
        rows: slice | None = None
        cols: slice | None = None
        if self.window_size is not None:
            h, w = classes.shape
            ws = self.window_size
            if ws > h or ws > w:
                raise ValueError(
                    f"window_size {ws} exceeds the sample's label raster ({h}x{w})"
                )
            if self.label_at_center_pixel:
                row, col = self._locate_labeled_pixel(classes)
            else:
                row, col = h // 2, w // 2
            row0 = min(max(row - ws // 2, 0), h - ws)
            col0 = min(max(col - ws // 2, 0), w - ws)
            rows, cols = slice(row0, row0 + ws), slice(col0, col0 + ws)
            classes = classes[rows, cols]
        if self.label_at_center_pixel:
            row, col = self._locate_labeled_pixel(classes)
            return rows, cols, classes[row, col]
        return rows, cols, classes

    def _transform_sample(
        self,
        input_dict: dict,
        target: dict,
        tile: tuple[int, int] | None = None,
    ) -> tuple[MaskedOlmoEarthSample, torch.Tensor]:
        """Transform a raw rslearn sample into (MaskedOlmoEarthSample, label).

        With tile set (tile_samples mode), (tile_row, tile_col) selects which
        window_size x window_size tile of the stored sample to emit.
        """
        sample_dict: dict[str, Any] = {}
        sample_timesteps: int | None = None

        # Parse the target first: with window_size / label_at_center_pixel the
        # imagery crop is derived from the label raster (centered on the
        # labeled pixel), so the label must be known before imagery is built.
        label = self._parse_label(target)
        crop_rows: slice | None = None
        crop_cols: slice | None = None
        if tile is not None:
            assert self.window_size is not None
            if label.shape != (self.sample_size, self.sample_size):
                raise ValueError(
                    f"tile_samples expects {self.sample_size}x{self.sample_size} "
                    f"label rasters, got {tuple(label.shape)}"
                )
            tile_row, tile_col = tile
            ws = self.window_size
            crop_rows = slice(tile_row * ws, (tile_row + 1) * ws)
            crop_cols = slice(tile_col * ws, (tile_col + 1) * ws)
            label = label[crop_rows, crop_cols]
        elif self.window_size is not None or self.label_at_center_pixel:
            crop_rows, crop_cols, label = self._label_crop_slices(label)

        for modality in self.input_modalities:
            if modality not in input_dict:
                raise ValueError(f"Modality {modality} not found in dataset inputs")
            x = input_dict[modality]
            if not isinstance(x, RasterImage):
                raise TypeError(
                    f"Input modality '{modality}' must be RasterImage, got {type(x).__name__}"
                )

            img = x.image
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            x = rearrange(img, "c t h w -> h w t c")
            if crop_rows is not None and crop_cols is not None:
                x = x[crop_rows, crop_cols]

            if sample_timesteps is None:
                sample_timesteps = x.shape[2]

            if modality == Modality.SENTINEL1.name:
                x = convert_to_db(x)

            if modality in EMBEDDING_PRODUCT_MODALITIES:
                # Precomputed embedding products are consumed exactly as
                # stored; imagery normalization does not apply, and dataset
                # registries carry no norm stats for them.
                sample_dict[modality] = torch.as_tensor(x, dtype=torch.float32)
                continue

            if self.norm_stats_from_pretrained:
                x = self.normalizer_computed.normalize(Modality.get(modality), x)
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

        sample_timesteps = sample_timesteps or self.max_timesteps
        timestamps = get_timestamps(
            self.start_time, self.end_time, num_timesteps=sample_timesteps
        )
        sample_dict["timestamps"] = torch.stack(timestamps)

        olmoearth_sample = OlmoEarthSample(**sample_dict)
        masked_sample = MaskedOlmoEarthSample.from_olmoearthsample(olmoearth_sample)

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
                if masked_attr.shape[1:3] != sample_dict[modality].shape[1:3]:
                    raise ValueError(
                        f"Modality mask {mask_attr_name} and modality {modality} have different hw shapes: "
                        f"{masked_attr.shape[1:3]} != {sample_dict[modality].shape[1:3]}"
                    )

        return masked_sample, label

    def _parse_label(self, target: dict) -> torch.Tensor:
        """Parse the raw rslearn target dict into a label tensor."""
        if self.target_task_name:
            data_dict = target.get(self.target_task_name, {})
        else:
            data_dict = target

        if self.target_task_type == TaskType.SEGMENTATION:
            classes = torch.as_tensor(
                data_dict["classes"].image, dtype=torch.long
            ).squeeze()
            valid = torch.as_tensor(
                data_dict["valid"].image, dtype=torch.long
            ).squeeze()
        elif self.target_task_type == TaskType.CLASSIFICATION:
            classes = data_dict["class"]
            valid = data_dict["valid"]
        elif self.target_task_type == TaskType.PER_PIXEL_REGRESSION:
            values = torch.as_tensor(
                data_dict["values"].image, dtype=torch.float32
            ).squeeze()
            valid = torch.as_tensor(
                data_dict["valid"].image, dtype=torch.float32
            ).squeeze()
            values[valid == 0] = float("nan")
            return values
        elif self.target_task_type == TaskType.WINDOW_REGRESSION:
            # Vector RegressionTask emits a single value/valid per window.
            value = torch.as_tensor(data_dict["value"], dtype=torch.float32).squeeze()
            valid = torch.as_tensor(data_dict["valid"], dtype=torch.float32).squeeze()
            if valid == 0:
                value = torch.tensor(float("nan"), dtype=torch.float32)
            return value
        else:
            raise ValueError(
                f"Unsupported target task type: {self.target_task_type.value}"
            )

        if valid is not None:
            assert classes is not None, "valid mask present but no classes tensor"
            classes = classes.masked_fill(valid == 0, SEGMENTATION_IGNORE_LABEL)
        return classes

    def __len__(self) -> int:
        """Length of the dataset."""
        return len(self.dataset) * self._tiles_per_side**2

    def __getitem__(self, idx: int) -> tuple[MaskedOlmoEarthSample, torch.Tensor]:
        """Return a MaskedOlmoEarthSample and target tensor."""
        if self._tiles_per_side > 1:
            base_idx, tile = divmod(idx, self._tiles_per_side**2)
            if self._cached_base is None or self._cached_base[0] != base_idx:
                input_dict, target, _ = self.dataset[base_idx]
                self._cached_base = (base_idx, (input_dict, target))
            input_dict, target = self._cached_base[1]
            return self._transform_sample(
                input_dict, target, tile=divmod(tile, self._tiles_per_side)
            )
        input_dict, target, _ = self.dataset[idx]
        return self._transform_sample(input_dict, target)


class IterableRslearnToOlmoEarthDataset(IterableDataset, RslearnToOlmoEarthDataset):
    """Iterable variant so PyTorch DataLoader uses __iter__ instead of __getitem__."""

    def __iter__(self) -> Iterator[tuple[MaskedOlmoEarthSample, torch.Tensor]]:
        """Iterate over the dataset."""
        for input_dict, target, _ in self.dataset:
            if self._tiles_per_side > 1:
                for tile in range(self._tiles_per_side**2):
                    yield self._transform_sample(
                        input_dict, target, tile=divmod(tile, self._tiles_per_side)
                    )
            else:
                yield self._transform_sample(input_dict, target)


def wrap_rslearn_dataset(**kwargs: Any) -> RslearnToOlmoEarthDataset:
    """Wrap an rslearn dataset, picking map-style or iterable based on what rslearn returns."""
    if isinstance(kwargs.get("model_dataset"), IterableDataset):
        return IterableRslearnToOlmoEarthDataset(**kwargs)
    return RslearnToOlmoEarthDataset(**kwargs)


def from_registry_entry(
    entry: EvalDatasetEntry,
    split: str = "train",
    norm_method: str = NormMethod.NORM_NO_CLIP_2_STD,
    norm_stats_from_pretrained: bool | None = None,
    max_samples: int | None = None,
    input_modalities_override: list[str] | None = None,
    groups_override: list[str] | None = None,
    tags_override: dict[str, str] | None = None,
    label_fraction: float = 1.0,
    label_fraction_seed: int = 42,
    window_size: int | None = None,
    label_at_center_pixel: bool = False,
    tile_samples: bool = False,
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
        label_fraction: Fraction of train labels to use for map-style train
            datasets. Non-train splits always use the full split.
        label_fraction_seed: Seed for the deterministic label_fraction
            subsample so the same low-label subset is used across runs.
        window_size: Center-crop every sample to window_size x window_size
            (see RslearnToOlmoEarthDataset).
        label_at_center_pixel: Emit the labeled pixel's class as a scalar
            classification label (see RslearnToOlmoEarthDataset).
        tile_samples: Tile every sample into window_size x window_size windows
            instead of center-cropping; the stored sample size is taken from
            the registry entry's window_size (see RslearnToOlmoEarthDataset).

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

    # Splits are always tag-based: ingest writes split_tag_key with train/val/test values
    effective_tags = tags_override
    if effective_tags is None and entry.split_tag_key:
        effective_tags = {entry.split_tag_key: normalized_split}
        if groups_override is None:
            groups_override = []
        log.info(f"Using tag-based splits: {entry.split_tag_key}={normalized_split}")

    log.info(f"Loading model config from {model_yaml_path}")
    model_config = parse_model_config(model_yaml_path)

    if not model_config:
        raise ValueError(
            f"Failed to load model.yaml from {model_yaml_path}. "
            "Check that the file exists and is valid YAML."
        )

    log.info(f"Building dataset for {entry.name} (path: {dataset_path})")
    if not use_pretrain_norm and not entry.norm_stats:
        raise ValueError(
            f"Dataset '{entry.name}' has use_pretrain_norm=False but no norm_stats in registry."
        )
    if tile_samples and not entry.window_size:
        raise ValueError(
            f"tile_samples requires registry entry '{entry.name}' to record its "
            "stored sample size in window_size."
        )
    return RslearnToOlmoEarthDataset.from_model_config(
        model_config=model_config,
        source_path=dataset_path,
        split=normalized_split,
        input_modalities=input_modalities,
        # Per-dataset imagery time range (for timestamp synthesis); fall back
        # to the defaults when the entry does not record one.
        start_time=entry.start_time or DEFAULT_START_TIME,
        end_time=entry.end_time or DEFAULT_END_TIME,
        norm_stats_from_pretrained=use_pretrain_norm,
        norm_method=norm_method,
        ds_norm_stats_json=None,
        ds_norm_stats=entry.norm_stats if not use_pretrain_norm else None,
        max_samples=max_samples,
        groups_override=groups_override,
        tags_override=effective_tags,
        label_fraction=label_fraction,
        label_fraction_seed=label_fraction_seed,
        window_size=window_size,
        label_at_center_pixel=label_at_center_pixel,
        tile_samples=tile_samples,
        sample_size=entry.window_size if tile_samples else None,
    )

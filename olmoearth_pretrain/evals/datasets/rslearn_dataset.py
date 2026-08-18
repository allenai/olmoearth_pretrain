"""Convert rslearn dataset to OlmoEarth Pretrain evaluation dataset format."""

from __future__ import annotations

import json
import logging
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
from olmoearth_pretrain.datatypes import MaskValue
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
from olmoearth_pretrain.evals.studio_ingest.provenance import (
    log_eval_dataset_provenance_to_wandb,
    sha256_of_file,
    verify_config_json_hash,
)
from olmoearth_pretrain.evals.task_types import TaskType
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample, OlmoEarthSample

from .normalize import normalize_bands

logger = logging.getLogger(__name__)

# Fallback imagery time range for timestamp synthesis, used when the imagery
# carries no acquisition times of its own and the registry entry does not
# record the dataset's actual time range.
DEFAULT_START_TIME = "2022-09-01"
DEFAULT_END_TIME = "2023-09-01"

# model.yaml input name of the optional Sentinel-2 SCL (scene classification)
# layers written by scripts/tools/setup_extra_layers.py. Never fed to the
# model; scl_cloud_mask=True turns it into per-pixel MISSING masks on S2.
SCL_INPUT_NAME = "scl"

# SCL classes treated as cloud-contaminated: 0 nodata, 1 saturated/defective,
# 3 cloud shadow, 8 cloud medium probability, 9 cloud high probability,
# 10 thin cirrus. Everything else (vegetation, bare, water, unclassified,
# snow) is real surface signal and stays. The default for scl_cloud_mask;
# override per task with scl_cloud_classes.
SCL_CLOUD_CLASSES = (0, 1, 3, 8, 9, 10)

# The narrower "cloudless" policy: unambiguous cloud only (medium + high
# probability), leaving shadow/cirrus/nodata in place. Sits between unmasked
# and SCL_CLOUD_CLASSES on the masking-aggressiveness ladder.
SCL_CLOUDLESS_CLASSES = (8, 9)

# Sidecar mapping "group/window" -> {"mo01": scene cloud_cover, ...} for the
# Landsat monthlies, written by
# scripts/tools/build_landsat_cloud_cover_sidecar.py from the cloud_cover
# rslearn's LandsatOliTirsItem persists in items.json. Scene-level (whole
# Landsat scene metadata), unlike the pixel-level SCL mask; -1 = unknown.
LANDSAT_CLOUD_COVER_SIDECAR = "landsat_cloud_cover.json"

# Landsat QA_PIXEL (CFMask) per-pixel cloud mask, the Landsat analogue of
# "scl". The optional "landsat_qa" input carries the QA_PIXEL band of the
# SAME scenes as the Landsat imagery (setup_extra_layers.py clones the
# prepared items, layer set `landsat_qa`); l8_pixel_cloud_mask=True turns it
# into per-pixel MISSING masks on LANDSAT. Bit flags: 1 dilated cloud,
# 2 cirrus, 3 cloud, 4 cloud shadow (Collection 2 QA_PIXEL).
L8QA_INPUT_NAME = "landsat_qa"
L8QA_CLOUD_BITS_MASK = 0b0000000000011110

# The narrow "strict" policy: the cloud bit alone, leaving dilated cloud, cirrus
# and cloud shadow in place.
#
# NAME WARNING: the task variant this drives, `_l8pixstrict`, is LESS aggressive
# than `_l8pixmask`. "Strict" qualifies the CRITERION for calling a pixel cloudy
# -- only unambiguous cloud counts -- so FEWER pixels get masked. The ladder is
# unmasked < _l8pixstrict < _l8pixmask, mirroring the S2 side's
# unmasked < _cloudless < _sclmask. The Landsat analogue of SCL_CLOUDLESS_CLASSES, and
# registered for the same reason: on the S2 side the aggressive policy LOST to
# the narrow one (descals cloudless +1.4 vs sclmask +0.6 -- masking
# shadow/cirrus/nodata nets negative), and the default above is the aggressive
# analogue. Dilated cloud is a buffer that discards clean pixels by
# construction; cirrus is often benign for surface reflectance.
L8QA_CLOUD_ONLY_BITS_MASK = 0b0000000000001000


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


def timestamps_from_time_ranges(
    time_ranges: list[tuple[datetime, datetime]],
) -> torch.Tensor:
    """Convert per-timestep acquisition ranges to OlmoEarth timestamps.

    Each range's *start* is used, matching how pretraining timestamps are built
    from the stored imagery (``dataset/sample.py``: the period start of each
    monthly mosaic), so eval-time dates are on the same convention the model
    was trained with.

    Args:
        time_ranges: one (start, end) datetime tuple per timestep, in the order
            the timesteps appear on the imagery's time axis.

    Returns:
        Long tensor of shape (T, 3) holding [day, month (0-indexed), year].
    """
    return torch.stack(
        [
            torch.tensor(
                [int(start.day), int(start.month) - 1, int(start.year)],
                dtype=torch.long,
            )
            for start, _ in time_ranges
        ]
    )


def _window_key(metadata: Any) -> str | None:
    """Rslearn SampleMetadata -> the "group/name" key used by sidecars."""
    group = getattr(metadata, "window_group", None)
    name = getattr(metadata, "window_name", None)
    if name is None:
        return None
    return f"{group}/{name}" if group else str(name)


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
        # materializer (olmoearth_pretrain/evals/embedding_materializer) or a
        # dataset export/inference script (tessera_v2).
        Modality.GSE.name,
        Modality.TESSERA.name,
        Modality.TESSERA_V11.name,
        Modality.TESSERA_V2.name,
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
        declared_bands: dict[str, list[str]] | None = None,
        scl_cloud_mask: bool = False,
        scl_cloud_classes: tuple[int, ...] | list[int] | None = None,
        landsat_cloud_cover_max: float | None = None,
        landsat_cloud_cover_table: dict[str, dict[str, float]] | None = None,
        l8_pixel_cloud_mask: bool = False,
        l8_pixel_cloud_bits: int | None = None,
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
            start_time: Fallback start time for synthesized timestamps, used
                only when the imagery carries no acquisition times.
            end_time: Fallback end time for synthesized timestamps.
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
            declared_bands: modality -> the band list model.yaml declares. When a
                modality stores fewer than its canonical bands, read channels are
                scattered to their canonical positions and the rest zeroed after
                normalization (see _init_band_scatter). None assumes full bands.
            scl_cloud_mask: If set, the optional "scl" input (Sentinel-2
                scene classification, one layer per S2 monthly layer) is read
                and every pixel-timestep whose SCL class is in
                SCL_CLOUD_CLASSES gets MaskValue.MISSING in the S2 mask, so
                the encoder ignores cloud-contaminated tokens. Pixels cloudy
                at every timestep are left unmasked (never blank a pixel
                entirely; this also keeps zero-padding pixels, whose SCL is
                0 everywhere, behaving as before). Windows without the scl
                input are left unmasked with a once-per-run warning.
            scl_cloud_classes: SCL classes to treat as cloud when
                scl_cloud_mask is set. None uses SCL_CLOUD_CLASSES.
            landsat_cloud_cover_max: If set, Landsat months whose chosen
                scene's cloud_cover meets/exceeds this threshold are masked
                MISSING (scene-level, from the sidecar table). Unknown cover
                (-1) is never masked.
            landsat_cloud_cover_table: The parsed sidecar ("group/window" ->
                {"moNN": cover}); required for the threshold to do anything.
            l8_pixel_cloud_bits: QA_PIXEL bit mask counting as cloud. None
                keeps L8QA_CLOUD_BITS_MASK (dilated|cirrus|cloud|shadow); the
                narrow policy is L8QA_CLOUD_ONLY_BITS_MASK (cloud alone).
            l8_pixel_cloud_mask: If set, the optional "landsat_qa" input
                (QA_PIXEL of the same scenes as the Landsat imagery, one
                layer per Landsat monthly layer) is read and every
                pixel-timestep flagged dilated/cirrus/cloud/shadow gets
                MaskValue.MISSING in the Landsat mask -- the Landsat analogue
                of scl_cloud_mask, with the same never-blank-a-pixel guard
                and the same leave-unmasked-with-a-warning fallbacks.
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

        # Fallback temporal config, used only when the imagery carries no
        # acquisition times of its own (see _build_timestamps).
        self.start_time = start_time
        self.end_time = end_time
        self.max_timesteps = num_timesteps  # Max expected timesteps (for validation)
        self._warned_synthesized_timestamps = False

        self.scl_cloud_mask = scl_cloud_mask
        self.scl_cloud_classes = (
            tuple(scl_cloud_classes)
            if scl_cloud_classes is not None
            else SCL_CLOUD_CLASSES
        )
        self._warned_scl_mask = False
        self._warned_ragged = False

        self.landsat_cloud_cover_max = landsat_cloud_cover_max
        self.landsat_cloud_cover_table = landsat_cloud_cover_table
        self._warned_l8_mask = False

        self.l8_pixel_cloud_mask = l8_pixel_cloud_mask
        # None keeps the aggressive default, so every existing _l8pixmask task
        # scores exactly as before.
        self.l8_pixel_cloud_bits = (
            L8QA_CLOUD_BITS_MASK if l8_pixel_cloud_bits is None else l8_pixel_cloud_bits
        )
        self._warned_l8qa_mask = False

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

        self._init_band_scatter(declared_bands)

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
        scl_cloud_mask: bool = False,
        scl_cloud_classes: tuple[int, ...] | list[int] | None = None,
        landsat_cloud_cover_max: float | None = None,
        l8_pixel_cloud_mask: bool = False,
        l8_pixel_cloud_bits: int | None = None,
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
            start_time: Fallback start time for synthesized timestamps, used
                only when the imagery carries no acquisition times.
            end_time: Fallback end time for synthesized timestamps.
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
            scl_cloud_mask: Mask cloudy S2 pixel-timesteps MISSING using the
                optional "scl" input (see RslearnToOlmoEarthDataset).
            scl_cloud_classes: SCL classes to treat as cloud (None =
                SCL_CLOUD_CLASSES).
            landsat_cloud_cover_max: Scene-level Landsat cloud threshold; the
                sidecar is read from source_path (see
                RslearnToOlmoEarthDataset).
            l8_pixel_cloud_bits: QA_PIXEL bit mask counting as cloud; None
                keeps the aggressive default.
            l8_pixel_cloud_mask: Mask cloudy Landsat pixel-timesteps MISSING
                using the optional "landsat_qa" input (see
                RslearnToOlmoEarthDataset).
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
                # SCL and Landsat QA are mask inputs, not model modalities
                # (see scl_cloud_mask / l8_pixel_cloud_mask).
                if layer.startswith("sentinel2_scl"):
                    continue
                if layer.startswith("landsat_qa"):
                    continue
                resolved = resolve_rslearn_layer_name(layer)
                if resolved is not None:
                    input_modalities.append(RSLEARN_TO_OLMOEARTH[resolved].name)
                else:
                    input_modalities.append(layer)

        task_info = get_task_info(model_config)

        landsat_cloud_cover_table = None
        if landsat_cloud_cover_max is not None:
            sidecar = f"{str(source_path).rstrip('/')}/{LANDSAT_CLOUD_COVER_SIDECAR}"
            try:
                with open(sidecar) as f:
                    landsat_cloud_cover_table = json.load(f)["windows"]
            except (OSError, KeyError, json.JSONDecodeError) as e:
                logger.warning(
                    f"landsat_cloud_cover_max set but sidecar unusable at "
                    f"{sidecar} ({type(e).__name__}: {e}); Landsat cloud "
                    "masking disabled for this dataset"
                )

        # Per-input band lists as declared in model.yaml. A dataset may store a
        # subset of a modality's canonical bands; see band_scatter.
        declared_bands = {
            name: list(cfg["bands"])
            for name, cfg in (
                model_config.get("data", {})
                .get("init_args", {})
                .get("inputs", {})
                .items()
            )
            if not cfg.get("is_target") and cfg.get("bands")
        }

        return wrap_rslearn_dataset(
            model_dataset=model_dataset,
            input_modalities=input_modalities,
            declared_bands=declared_bands,
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
            scl_cloud_mask=scl_cloud_mask,
            scl_cloud_classes=scl_cloud_classes,
            landsat_cloud_cover_max=landsat_cloud_cover_max,
            landsat_cloud_cover_table=landsat_cloud_cover_table,
            l8_pixel_cloud_mask=l8_pixel_cloud_mask,
            l8_pixel_cloud_bits=l8_pixel_cloud_bits,
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
        window_key: str | None = None,
    ) -> tuple[MaskedOlmoEarthSample, torch.Tensor]:
        """Transform a raw rslearn sample into (MaskedOlmoEarthSample, label).

        With tile set (tile_samples mode), (tile_row, tile_col) selects which
        window_size x window_size tile of the stored sample to emit.
        window_key ("group/name") keys the Landsat cloud-cover sidecar.
        """
        sample_dict: dict[str, Any] = {}
        sample_timesteps: int | None = None
        # Real acquisition ranges rslearn read off the imagery, per modality.
        stored_time_ranges: dict[str, list[tuple[datetime, datetime]]] = {}

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

        # First pass: read every present modality (crop, dB-convert), keeping
        # the raw arrays so ragged imagery can be aligned onto a shared
        # temporal axis before normalization.
        raw: dict[str, np.ndarray] = {}
        absent: list[str] = []
        for modality in self.input_modalities:
            x = input_dict.get(modality)
            if x is None:
                # Optional imagery (landsat) is simply absent on windows with
                # no coverage; it is represented as all-MISSING below. A
                # precomputed embedding product keeps the loud failure -- a
                # coverage gap there must not silently become a zero vector.
                if modality in EMBEDDING_PRODUCT_MODALITIES:
                    raise ValueError(f"Modality {modality} not found in dataset inputs")
                absent.append(modality)
                continue
            if not isinstance(x, RasterImage):
                raise TypeError(
                    f"Input modality '{modality}' must be RasterImage, got {type(x).__name__}"
                )

            if x.timestamps is not None:
                stored_time_ranges[modality] = list(x.timestamps)

            img = x.image
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            arr = rearrange(img, "c t h w -> h w t c")
            if crop_rows is not None and crop_cols is not None:
                arr = arr[crop_rows, crop_cols]

            if modality == Modality.SENTINEL1.name:
                arr = convert_to_db(arr)
            if modality in self.band_scatter:
                arr = self._scatter_bands(modality, arr)
            raw[modality] = arr

        if not raw:
            raise ValueError("No input modalities present in sample")

        # Canonical temporal axis: the longest present imagery modality,
        # preferring one with stored acquisition times (the S2/S1 monthlies
        # in practice -- required inputs, so complete on surviving windows).
        # Embedding products are exempt from alignment: they are consumed
        # exactly as stored (typically T=1 annual) and bypass time encodings.
        imagery = [m for m in raw if m not in EMBEDDING_PRODUCT_MODALITIES]
        canonical = max(
            imagery or raw,
            key=lambda m: (raw[m].shape[2], m in stored_time_ranges),
        )
        canonical_t = raw[canonical].shape[2]
        sample_timesteps = canonical_t
        height, width = raw[canonical].shape[:2]

        # Slots each ragged modality is missing on the canonical axis; they
        # get MaskValue.MISSING after the masked sample is built.
        ragged_missing: dict[str, list[int]] = {}
        for modality in absent:
            n_bands = len(Modality.get(modality).band_order)
            raw[modality] = np.zeros(
                (height, width, canonical_t, n_bands), dtype=np.float32
            )
            ragged_missing[modality] = list(range(canonical_t))
            self._warn_ragged_once(f"'{modality}' absent from sample")
        for modality in imagery:
            if raw[modality].shape[2] == canonical_t:
                continue
            raw[modality], missing = self._align_to_canonical(
                modality,
                raw[modality],
                stored_time_ranges.get(modality),
                stored_time_ranges.get(canonical),
                canonical_t,
            )
            ragged_missing[modality] = missing

        # Scene-level Landsat cloud mask: months whose chosen scene is over
        # the threshold join the MISSING slots (union with coverage gaps).
        if self.landsat_cloud_cover_max is not None and Modality.LANDSAT.name in raw:
            cloudy = self._landsat_cloud_slots(window_key, canonical_t)
            if cloudy:
                merged = set(ragged_missing.get(Modality.LANDSAT.name, []))
                ragged_missing[Modality.LANDSAT.name] = sorted(merged | set(cloudy))

        for modality in self.input_modalities:
            x = raw[modality]
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
            # Post-normalization, so the value the model sees is exactly 0 --
            # raw 0 would normalize to (0 - (mean - 2*std)) / (4*std) instead.
            for band_index in self.absent_bands.get(modality, []):
                x[..., band_index] = 0.0

            sample_dict[modality] = torch.as_tensor(x, dtype=torch.float32)

        sample_timesteps = sample_timesteps or self.max_timesteps
        sample_dict["timestamps"] = self._build_timestamps(
            sample_timesteps, stored_time_ranges
        )

        olmoearth_sample = OlmoEarthSample(**sample_dict)
        masked_sample = MaskedOlmoEarthSample.from_olmoearthsample(olmoearth_sample)

        # Slots a ragged/absent modality has no observation for are MISSING,
        # so the encoder ignores them instead of reading zeros as data.
        for modality, slots in ragged_missing.items():
            if not slots:
                continue
            mask = getattr(
                masked_sample,
                MaskedOlmoEarthSample.get_masked_modality_name(modality),
            )
            mask[:, :, slots, :] = MaskValue.MISSING.value

        if self.scl_cloud_mask and Modality.SENTINEL2_L2A.name in self.input_modalities:
            self._apply_scl_cloud_mask(masked_sample, input_dict, crop_rows, crop_cols)
        if self.l8_pixel_cloud_mask and Modality.LANDSAT.name in self.input_modalities:
            self._apply_l8_pixel_cloud_mask(
                masked_sample,
                input_dict,
                crop_rows,
                crop_cols,
                stored_time_ranges.get(canonical),
                canonical_t,
            )

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

    def _warn_scl_once(self, reason: str) -> None:
        """Warn about a skipped cloud mask once per dataset instance."""
        if not self._warned_scl_mask:
            logger.warning(f"scl_cloud_mask: {reason}; leaving S2 unmasked")
            self._warned_scl_mask = True

    def _warn_l8_once(self, reason: str) -> None:
        """Warn about a skipped Landsat cloud mask once per dataset instance."""
        if not self._warned_l8_mask:
            logger.warning(f"landsat_cloud_cover_max: {reason}; not masking")
            self._warned_l8_mask = True

    def _warn_l8qa_once(self, reason: str) -> None:
        """Warn about a skipped QA_PIXEL mask once per dataset instance."""
        if not self._warned_l8qa_mask:
            logger.warning(f"l8_pixel_cloud_mask: {reason}; leaving Landsat unmasked")
            self._warned_l8qa_mask = True

    def _landsat_cloud_slots(
        self, window_key: str | None, canonical_t: int
    ) -> list[int]:
        """Canonical slots whose Landsat scene cloud_cover is over threshold.

        Scene-level: the sidecar records the cloud_cover of each month's top
        mosaic scene. moNN maps to slot NN-1 -- the aligned exports' canonical
        axis is the twelve ascending monthly layers by construction. -1
        (unknown cover) is never masked.
        """
        if self.landsat_cloud_cover_table is None:
            self._warn_l8_once("no sidecar table loaded")
            return []
        if canonical_t != 12:
            self._warn_l8_once(f"canonical axis has {canonical_t} slots, expected 12")
            return []
        months = self.landsat_cloud_cover_table.get(window_key or "")
        if months is None:
            self._warn_l8_once(f"window {window_key!r} not in sidecar")
            return []
        assert self.landsat_cloud_cover_max is not None
        return sorted(
            int(month[2:]) - 1
            for month, cover in months.items()
            if cover is not None
            and 0 <= cover
            and cover >= self.landsat_cloud_cover_max
        )

    def _warn_ragged_once(self, reason: str) -> None:
        """Warn about ragged/absent optional imagery once per dataset instance."""
        if not self._warned_ragged:
            logger.warning(
                f"ragged imagery: {reason}; representing missing timesteps as "
                "MaskValue.MISSING (expected for optional inputs with coverage "
                "gaps, e.g. landsat)"
            )
            self._warned_ragged = True

    def _align_to_canonical(
        self,
        modality: str,
        arr: np.ndarray,
        ranges: list[tuple[datetime, datetime]] | None,
        canonical_ranges: list[tuple[datetime, datetime]] | None,
        canonical_t: int,
    ) -> tuple[np.ndarray, list[int]]:
        """Scatter a ragged modality's timesteps onto the canonical time axis.

        A modality with coverage gaps (landsat at a 16-day revisit misses
        whole months) arrives with T < canonical_t, and its timestep i is NOT
        month i -- consuming it positionally would desynchronize the data
        from the shared timestamps tensor and its own mask. Each timestep is
        placed into the canonical slot whose acquisition period it belongs to
        (nearest period start, within half a period), and unfilled slots are
        reported for MISSING-masking.

        Args:
            modality: the modality name (for the warning).
            arr: (H, W, T, C) with T < canonical_t.
            ranges: the modality's stored acquisition ranges (len T), if any.
            canonical_ranges: the canonical modality's ranges (len
                canonical_t), if any.
            canonical_t: the shared temporal length.

        Returns:
            ((H, W, canonical_t, C) array, canonical slots left unfilled).
        """
        height, width, t, channels = arr.shape
        aligned = np.zeros((height, width, canonical_t, channels), dtype=arr.dtype)
        filled: set[int] = set()
        if ranges and canonical_ranges and len(canonical_ranges) == canonical_t:
            for i, (start, _) in enumerate(ranges):
                deltas = [
                    abs((start - c_start).days) for c_start, _ in canonical_ranges
                ]
                slot = min(range(canonical_t), key=deltas.__getitem__)
                if deltas[slot] <= 16 and slot not in filled:
                    aligned[:, :, slot] = arr[:, :, i]
                    filled.add(slot)
            self._warn_ragged_once(
                f"'{modality}' has {t}/{canonical_t} timesteps; aligned by "
                "acquisition date"
            )
        else:
            # No dates to align on -- place positionally and mask the tail.
            # Defensive: the monthly exports always carry acquisition times.
            aligned[:, :, :t] = arr
            filled = set(range(t))
            self._warn_ragged_once(
                f"'{modality}' has {t}/{canonical_t} timesteps and no "
                "acquisition dates; padded positionally"
            )
        return aligned, sorted(set(range(canonical_t)) - filled)

    def _apply_scl_cloud_mask(
        self,
        masked_sample: MaskedOlmoEarthSample,
        input_dict: dict[str, Any],
        crop_rows: slice | None,
        crop_cols: slice | None,
    ) -> None:
        """Set S2 mask to MISSING where SCL says cloud, in place.

        SCL rides through the same Pad transform as the imagery (the
        year-aligned model.yamls list it in image_selectors), so it arrives
        on the S2 pixel grid and takes the same crop. Any window where that
        does not hold (no scl input, unexpected shape) is left unmasked --
        the conservative direction -- with a once-per-run warning.
        """
        scl_image = input_dict.get(SCL_INPUT_NAME)
        if not isinstance(scl_image, RasterImage):
            self._warn_scl_once(f"no '{SCL_INPUT_NAME}' input in sample")
            return
        img = scl_image.image
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        scl = rearrange(img, "c t h w -> h w t c")[..., 0]

        s2 = input_dict[Modality.SENTINEL2_L2A.name].image
        if scl.shape[:2] != tuple(s2.shape[2:]):
            # SCL is stored at 20 m (zoom_offset -1); rslearn reads it back on
            # the window grid, but repair an exact half-resolution read rather
            # than silently misaligning the crop below.
            if tuple(x * 2 for x in scl.shape[:2]) == tuple(s2.shape[2:]):
                scl = scl.repeat(2, axis=0).repeat(2, axis=1)
            else:
                self._warn_scl_once(
                    f"SCL spatial shape {scl.shape[:2]} does not match S2 {tuple(s2.shape[2:])}"
                )
                return
        if crop_rows is not None and crop_cols is not None:
            scl = scl[crop_rows, crop_cols]

        mask_name = MaskedOlmoEarthSample.get_masked_modality_name(
            Modality.SENTINEL2_L2A.name
        )
        mask = getattr(masked_sample, mask_name)
        if mask is None or scl.shape != tuple(mask.shape[:3]):
            self._warn_scl_once(
                f"SCL shape {scl.shape} does not match S2 mask "
                f"{None if mask is None else tuple(mask.shape)}"
            )
            return

        cloudy = np.isin(scl, self.scl_cloud_classes)
        # Never blank a pixel entirely: a pixel cloudy at every timestep keeps
        # all of them (better a cloudy token than none, and zero-padding
        # pixels -- SCL 0 everywhere -- keep behaving exactly as unmasked).
        cloudy[cloudy.all(axis=2)] = False
        mask[torch.from_numpy(cloudy)] = MaskValue.MISSING.value

    def _apply_l8_pixel_cloud_mask(
        self,
        masked_sample: MaskedOlmoEarthSample,
        input_dict: dict[str, Any],
        crop_rows: slice | None,
        crop_cols: slice | None,
        canonical_ranges: list[tuple[datetime, datetime]] | None,
        canonical_t: int,
    ) -> None:
        """Set Landsat mask to MISSING where QA_PIXEL says cloud, in place.

        The QA_PIXEL raster describes exactly the scene each Landsat month was
        materialized from (the landsat_qa layers clone the imagery's prepared
        items), and rides through the same Pad transform, so it arrives on the
        Landsat pixel grid and takes the same crop. Mirrors
        _apply_scl_cloud_mask, including the conservative fallbacks: any
        window where the input is absent or misshapen is left unmasked with a
        once-per-run warning.

        Unlike SCL (derived from the required, always-complete S2 monthlies),
        QA inherits Landsat's ragged coverage, so it must be scattered onto the
        canonical time axis by acquisition date before it can index a mask that
        already lives on that axis -- consuming it positionally would mask the
        wrong months whenever either side has a coverage gap. Slots QA does not
        cover stay zero, which carries no cloud bits and so masks nothing (a
        Landsat gap is already MISSING from the ragged pass).
        """
        qa_image = input_dict.get(L8QA_INPUT_NAME)
        if not isinstance(qa_image, RasterImage):
            self._warn_l8qa_once(f"no '{L8QA_INPUT_NAME}' input in sample")
            return
        landsat_image = input_dict.get(Modality.LANDSAT.name)
        if not isinstance(landsat_image, RasterImage):
            # Landsat absent entirely: the ragged pass already marked every
            # slot MISSING, so there is nothing left to mask.
            self._warn_l8qa_once("Landsat imagery absent from sample")
            return
        stored = qa_image.image
        qa_raw: np.ndarray = (
            stored.numpy() if isinstance(stored, torch.Tensor) else np.asarray(stored)
        )
        # Keep the channel axis so the ragged scatter can be reused verbatim.
        qa = rearrange(qa_raw, "c t h w -> h w t c")[..., :1]

        landsat = landsat_image.image
        if qa.shape[:2] != tuple(landsat.shape[2:]):
            # QA is stored at 30 m (zoom_offset -1, like the multispectral
            # band set); repair an exact half-resolution read rather than
            # silently misaligning the crop below.
            if tuple(x * 2 for x in qa.shape[:2]) == tuple(landsat.shape[2:]):
                qa = qa.repeat(2, axis=0).repeat(2, axis=1)
            else:
                self._warn_l8qa_once(
                    f"QA spatial shape {qa.shape[:2]} does not match Landsat "
                    f"{tuple(landsat.shape[2:])}"
                )
                return
        if crop_rows is not None and crop_cols is not None:
            qa = qa[crop_rows, crop_cols]

        if qa.shape[2] != canonical_t:
            qa, _ = self._align_to_canonical(
                L8QA_INPUT_NAME,
                qa,
                list(qa_image.timestamps) if qa_image.timestamps is not None else None,
                canonical_ranges,
                canonical_t,
            )
        qa = qa[..., 0]

        mask_name = MaskedOlmoEarthSample.get_masked_modality_name(
            Modality.LANDSAT.name
        )
        mask = getattr(masked_sample, mask_name)
        if mask is None or qa.shape != tuple(mask.shape[:3]):
            self._warn_l8qa_once(
                f"QA shape {qa.shape} does not match Landsat mask "
                f"{None if mask is None else tuple(mask.shape)}"
            )
            return

        cloudy = (qa.astype(np.uint16) & self.l8_pixel_cloud_bits) != 0
        # Never blank a pixel entirely: if masking every cloudy timestep would
        # leave a pixel with no visible Landsat timestep at all, leave that
        # pixel unmasked (better a cloudy token than none). Slots already
        # MISSING from a coverage gap count against the pixel's survivors, so
        # this holds on ragged windows too.
        visible = (mask[..., 0] != MaskValue.MISSING.value).numpy()
        cloudy[~(visible & ~cloudy).any(axis=2)] = False
        mask[torch.from_numpy(cloudy)] = MaskValue.MISSING.value

    def _build_timestamps(
        self,
        num_timesteps: int,
        stored_time_ranges: dict[str, list[tuple[datetime, datetime]]],
    ) -> torch.Tensor:
        """Build the sample's (T, 3) timestamps tensor.

        Prefers the acquisition ranges rslearn read off the imagery, so every
        timestep carries its own real date. A single dataset-level
        (start_time, end_time) range cannot describe these datasets: their
        windows are dated per label (the AEF supplemental datasets span
        2016-2024), and several store their imagery item groups in *descending*
        time order, so synthesized ascending months mislabel every timestep.

        A sample carries one timestamps tensor for all modalities, so when
        several provide times the first ``input_modalities`` entry whose axis
        length matches wins (co-registered modalities share their period
        boundaries by construction; see the monthly layer scheme in
        scripts/tools/build_year_aligned_eval_configs.py).

        Falls back to synthesizing monthly timestamps over
        [start_time, end_time] only when the imagery carries no times at all
        (rslearn leaves ``RasterImage.timestamps`` unset for single-timestep
        layers whose items have no time range).

        Args:
            num_timesteps: the sample's time axis length.
            stored_time_ranges: modality -> per-timestep (start, end) ranges, as
                read from the imagery.

        Returns:
            Long tensor of shape (num_timesteps, 3): [day, month0, year].
        """
        for modality in self.input_modalities:
            time_ranges = stored_time_ranges.get(modality)
            if time_ranges is None:
                continue
            if len(time_ranges) != num_timesteps:
                logger.warning(
                    f"Modality {modality} has {len(time_ranges)} stored time "
                    f"ranges but the sample has {num_timesteps} timesteps; "
                    "ignoring them for timestamp construction."
                )
                continue
            return timestamps_from_time_ranges(time_ranges)

        if not self._warned_synthesized_timestamps:
            logger.warning(
                "No stored acquisition times on any of "
                f"{self.input_modalities}; synthesizing {num_timesteps} monthly "
                f"timestamps from {self.start_time}..{self.end_time}. These are "
                "the same for every window, so multi-year datasets get the "
                "wrong year."
            )
            self._warned_synthesized_timestamps = True
        return torch.stack(
            get_timestamps(self.start_time, self.end_time, num_timesteps=num_timesteps)
        )

    def _init_band_scatter(self, declared_bands: dict[str, list[str]] | None) -> None:
        """Record which canonical bands this dataset actually stores.

        A dataset may declare a SUBSET of a modality's bands -- e.g. an export
        built from a fetch that never pulled S2's 60 m band set. The model still
        tokenizes the full band_order (S2_SINGLE_BANDSET indexes B01/B09 at
        channels 10/11) and normalization carries one stat per canonical band, so
        read channels are scattered into full width and the rest are zeroed AFTER
        normalization -- which is what pretraining's band dropout produces.
        Full-band datasets get no entry and are untouched.

        Args:
            declared_bands: modality -> the band list model.yaml declares.

        Raises:
            ValueError: if a declared band is not in the modality's band order.
        """
        self.band_scatter: dict[str, list[int]] = {}
        self.absent_bands: dict[str, list[int]] = {}
        for modality, bands in (declared_bands or {}).items():
            if modality not in self.input_modalities:
                continue
            canonical = list(Modality.get(modality).band_order)
            unknown = [band for band in bands if band not in canonical]
            if unknown:
                raise ValueError(
                    f"modality '{modality}' declares bands {unknown} that are not "
                    f"in its canonical band order {canonical}"
                )
            indices = [canonical.index(band) for band in bands]
            if indices == list(range(len(canonical))):
                continue
            self.band_scatter[modality] = indices
            self.absent_bands[modality] = sorted(
                set(range(len(canonical))) - set(indices)
            )
            logger.info(
                f"'{modality}' stores {len(indices)}/{len(canonical)} bands; "
                f"missing {[canonical[i] for i in self.absent_bands[modality]]} "
                "will be zero after normalization"
            )

    def _scatter_bands(self, modality: str, arr: np.ndarray) -> np.ndarray:
        """Widen a subset-band array to the modality's canonical band count.

        Read channels land at their canonical positions; the rest stay 0 here and
        are re-zeroed after normalization (which is the value that matters).

        Args:
            modality: modality name.
            arr: (H, W, T, n_declared) array as read.

        Returns:
            (H, W, T, n_canonical) array.

        Raises:
            ValueError: if the array's channel count does not match the number of
                declared bands, which means the config and the rasters disagree.
        """
        indices = self.band_scatter[modality]
        if arr.shape[-1] != len(indices):
            raise ValueError(
                f"modality '{modality}' declares {len(indices)} bands but its "
                f"rasters carry {arr.shape[-1]}; config and data disagree"
            )
        widened = np.zeros(
            (*arr.shape[:-1], len(Modality.get(modality).band_order)),
            dtype=arr.dtype,
        )
        widened[..., indices] = arr
        return widened

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
                input_dict, target, metadata = self.dataset[base_idx]
                self._cached_base = (
                    base_idx,
                    (input_dict, target, _window_key(metadata)),
                )
            input_dict, target, window_key = self._cached_base[1]
            return self._transform_sample(
                input_dict,
                target,
                tile=divmod(tile, self._tiles_per_side),
                window_key=window_key,
            )
        input_dict, target, metadata = self.dataset[idx]
        return self._transform_sample(
            input_dict, target, window_key=_window_key(metadata)
        )


class IterableRslearnToOlmoEarthDataset(IterableDataset, RslearnToOlmoEarthDataset):
    """Iterable variant so PyTorch DataLoader uses __iter__ instead of __getitem__."""

    def __iter__(self) -> Iterator[tuple[MaskedOlmoEarthSample, torch.Tensor]]:
        """Iterate over the dataset."""
        for input_dict, target, metadata in self.dataset:
            window_key = _window_key(metadata)
            if self._tiles_per_side > 1:
                for tile in range(self._tiles_per_side**2):
                    yield self._transform_sample(
                        input_dict,
                        target,
                        tile=divmod(tile, self._tiles_per_side),
                        window_key=window_key,
                    )
            else:
                yield self._transform_sample(input_dict, target, window_key=window_key)


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
    scl_cloud_mask: bool = False,
    scl_cloud_classes: tuple[int, ...] | list[int] | None = None,
    landsat_cloud_cover_max: float | None = None,
    l8_pixel_cloud_mask: bool = False,
    l8_pixel_cloud_bits: int | None = None,
) -> RslearnToOlmoEarthDataset:
    """Build RslearnToOlmoEarthDataset from a registry EvalDatasetEntry.

    Uses jsonargparse to build ModelDataset directly from model.yaml. The
    model.yaml is read from the git checkout when the entry records a
    config_repo_dir (pinned by the commit being run), otherwise from the copy
    at entry.weka_path/model.yaml written during ingestion. The dataset
    folder's config.json is verified against the sha256 recorded in the
    registry before building.

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
        scl_cloud_mask: Mask cloudy S2 pixel-timesteps MISSING using the
            optional "scl" input (see RslearnToOlmoEarthDataset).
        scl_cloud_classes: SCL classes to treat as cloud (None =
            SCL_CLOUD_CLASSES).
        landsat_cloud_cover_max: Scene-level Landsat cloud threshold (see
            RslearnToOlmoEarthDataset).
        l8_pixel_cloud_bits: QA_PIXEL bit mask counting as cloud; None keeps
            the aggressive default.
        l8_pixel_cloud_mask: Mask cloudy Landsat pixel-timesteps MISSING using
            the optional "landsat_qa" input (see RslearnToOlmoEarthDataset).

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

    # Resolves to the git-tracked config when the entry records a
    # config_repo_dir; otherwise the copy in the Weka dataset folder.
    model_yaml_path = entry.model_yaml_path

    # config.json must live in the dataset folder (rslearn reads it from the
    # dataset root), so pin it by hash instead: fail loudly if it has drifted
    # from what was registered at ingest time.
    config_json_sha256 = verify_config_json_hash(
        entry.name, entry.weka_path, entry.config_json_sha256
    )
    log_eval_dataset_provenance_to_wandb(
        entry.name,
        {
            "model_yaml_path": model_yaml_path,
            "model_yaml_sha256": sha256_of_file(model_yaml_path),
            "config_json_sha256": config_json_sha256,
            "config_repo_dir": entry.config_repo_dir,
            "weka_path": entry.weka_path,
        },
    )

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
        # Per-dataset imagery time range, used only as the fallback when the
        # imagery has no acquisition times of its own (see _build_timestamps);
        # fall back further to the defaults when the entry records no range.
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
        scl_cloud_mask=scl_cloud_mask,
        scl_cloud_classes=scl_cloud_classes,
        landsat_cloud_cover_max=landsat_cloud_cover_max,
        l8_pixel_cloud_mask=l8_pixel_cloud_mask,
        l8_pixel_cloud_bits=l8_pixel_cloud_bits,
    )

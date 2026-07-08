"""GeoBench v2 datasets backed by custom tortilla-based dataloaders."""

from __future__ import annotations

import json
import logging
from enum import StrEnum
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset

import olmoearth_pretrain.evals.datasets.paths as paths
from olmoearth_pretrain.data.constants import Modality, ModalitySpec
from olmoearth_pretrain.data.dataset import OlmoEarthSample
from olmoearth_pretrain.data.normalize import Normalizer, Strategy
from olmoearth_pretrain.evals.datasets.configs import dataset_to_config
from olmoearth_pretrain.evals.datasets.geobench_v2_loaders import SLUG_TO_DATASET
from olmoearth_pretrain.evals.datasets.normalize import normalize_bands
from olmoearth_pretrain.evals.task_types import TaskType
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

logger = logging.getLogger(__name__)

_S2_ORDER = list(Modality.SENTINEL2_L2A.band_order)

# Max digital number (DN) value of the Sentinel-2 L2A dynamic range; used to
# rescale reflectance/uint8 inputs into the DN scale the OlmoEarth normalizer
# expects. Non-S2 sensors are rescaled into this range to pass through the
# S2 normalizer (ex. high-res imagery that isn't yet supported in the model).
_S2_DN_MAX = 10000.0
# Max value of a uint8 image, used when rescaling [0–255] inputs to DN scale.
_UINT8_MAX = 255.0


class Slug(StrEnum):
    """GeoBench v2 dataset slugs we branch on when mapping samples to modalities.

    Mirrors the keys of ``SLUG_TO_DATASET``.
    """

    BENV2 = "benv2"
    BIOMASSTERS = "biomassters"
    BURN_SCARS = "burn_scars"
    CAFFE = "caffe"
    CLOUDSEN12 = "cloudsen12"
    FLAIR2 = "flair2"
    FOTW = "fotw"
    KURO_SIWO = "kuro_siwo"
    SPACENET2 = "spacenet2"
    SPACENET7 = "spacenet7"
    TREESATAI = "treesatai"


def _s2_names(band_order: Any) -> list[str]:
    if isinstance(band_order, dict) and "s2" in band_order:
        return [str(b) for b in band_order["s2"]]
    return []


def _s1_names(band_order: Any) -> list[str]:
    if isinstance(band_order, dict):
        for k in ("s1", "s1_asc", "sar"):
            if k in band_order:
                return [str(b) for b in band_order[k]]
    return []


def _permute_bchw(
    x: torch.Tensor, source_names: list[str], target_names: list[str]
) -> torch.Tensor:
    idx = {n: i for i, n in enumerate(source_names)}
    out = torch.zeros(
        (x.shape[0], len(target_names), x.shape[2], x.shape[3]),
        dtype=x.dtype,
        device=x.device,
    )
    for j, name in enumerate(target_names):
        if name in idx:
            out[:, j] = x[:, idx[name]]
    return out


def _align_s2_to_sentinel2_l2a(
    x: torch.Tensor, source_names: list[str]
) -> torch.Tensor:
    """Map GeoBench S2 channels to full OlmoEarth SENTINEL2_L2A order (12 bands)."""
    if x.dim() == 3:
        return _permute_bchw(x.unsqueeze(0), source_names, _S2_ORDER)[0]
    if x.dim() == 4:
        out = torch.zeros(
            (len(_S2_ORDER), x.shape[1], x.shape[2], x.shape[3]),
            dtype=x.dtype,
            device=x.device,
        )
        idx = {n: i for i, n in enumerate(source_names)}
        for j, name in enumerate(_S2_ORDER):
            if name in idx:
                out[j] = x[idx[name]]
        return out
    raise ValueError(f"expected 3D or 4D S2 tensor, got {x.shape}")


def _bchw_to_hwtc(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        return rearrange(x, "c h w -> h w 1 c")
    if x.dim() == 4:
        return rearrange(x, "c t h w -> h w t c")
    raise ValueError(f"expected 3D or 4D image tensor, got {x.shape}")


def _timestamps(
    t: int, device: torch.device, dtype: torch.dtype = torch.long
) -> torch.Tensor:
    day = torch.full((t,), 15, device=device, dtype=dtype)
    month = torch.full((t,), 6, device=device, dtype=dtype)
    year = torch.full((t,), 2020, device=device, dtype=dtype)
    return torch.stack([day, month, year], dim=-1)


def _pad_to_t(hwtc: torch.Tensor, t: int) -> torch.Tensor:
    """Pad hwtc along the time axis with zeros to reach t timesteps."""
    gap = t - hwtc.shape[2]
    if gap <= 0:
        return hwtc
    pad = torch.zeros(
        *hwtc.shape[:2], gap, hwtc.shape[3], device=hwtc.device, dtype=hwtc.dtype
    )
    return torch.cat([hwtc, pad], dim=2)


def _to_db(x: torch.Tensor) -> torch.Tensor:
    """Convert linear power scale to dB."""
    return 10.0 * torch.log10(x.clamp(min=1e-10))


def _s2_sample(
    image: torch.Tensor, src_names: list[str], device: torch.device
) -> OlmoEarthSample:
    """Build a single-modality S2 OlmoEarthSample from an arbitrary-band image."""
    hwtc = _bchw_to_hwtc(_align_s2_to_sentinel2_l2a(image.float(), src_names))
    return OlmoEarthSample(
        sentinel2_l2a=hwtc, timestamps=_timestamps(hwtc.shape[2], device)
    )


def _sample_to_olmoearth(
    sample: dict[str, torch.Tensor],
    band_order: Any,
    slug: str,
) -> OlmoEarthSample:
    """Map a raw GeoBench sample into an OlmoEarthSample.

    Several datasets are not collected from a modality the model trains against
    (see ``EvalDatasetConfig.source_imagery``); their imagery is rescaled into
    the S2 DN range and fed through the SENTINEL2_L2A slot. The per-branch
    comments below describe the source sensor and the rescale it requires.
    """
    device = next(iter(sample.values())).device

    if slug == Slug.CAFFE:
        # CaFFe is SAR imagery (primarily Sentinel-1), single-channel amplitude in
        # [0–255]. Treat it as Sentinel-1: grayscale -> VV, with VH left as a zero
        # (dropped) channel that is re-zeroed after normalization. The 0–255 scale
        # is reconciled by normalization (OlmoEarth S1 stats, or the dataset's own
        # stats when norm_stats_from_pretrained=False).
        gray = sample["image"].float()[:1]  # (1, H, W)
        s1 = torch.cat([gray, torch.zeros_like(gray)], dim=0)  # (2, H, W) = [vv, vh]
        hwtc = _bchw_to_hwtc(s1)
        return OlmoEarthSample(
            sentinel1=hwtc, timestamps=_timestamps(hwtc.shape[2], device)
        )

    if slug in (Slug.BURN_SCARS, Slug.CLOUDSEN12, Slug.SPACENET7):
        image = sample["image"].float()
        src = _s2_names(band_order)
        if slug == Slug.BURN_SCARS:
            # GeoBench2 stores HLS reflectance in [0, 1]; OlmoEarth pretraining
            # stats are in DN scale (~0–10000), so rescale to match.
            image = image * _S2_DN_MAX
        if slug == Slug.CLOUDSEN12:
            image = image[: len(src)]  # tortilla has 14 bands; truncate to 12
        if slug == Slug.SPACENET7:
            # PlanetScope uint8 [0–255]; rescale to S2 DN range.
            image = image * (_S2_DN_MAX / _UINT8_MAX)
            rgbn_to_s2 = {"red": "B04", "green": "B03", "blue": "B02", "nir": "B08"}
            src = [rgbn_to_s2.get(s, s) for s in src]
        return _s2_sample(image, src, device)

    if slug == Slug.FOTW:
        # 4-band RGBN Sentinel-2 (uint16, already in S2 DN scale); map to S2 band
        # names and pad remaining channels to 0.
        rgbn_to_s2 = {"red": "B04", "green": "B03", "blue": "B02", "nir": "B08"}
        src = [rgbn_to_s2[b] for b in ("red", "green", "blue", "nir")]
        return _s2_sample(sample["image"].float(), src, device)

    if slug == Slug.FLAIR2:
        # 5-band aerial uint8 [0–255] (RGBN + elevation); rescale to S2 DN range
        # and pad to 12 channels, treating the result as sentinel2_l2a.
        x = sample["image"].float() * (_S2_DN_MAX / _UINT8_MAX)
        n = len(_S2_ORDER)
        if x.shape[0] < n:
            x = torch.cat(
                [
                    x,
                    torch.zeros(
                        n - x.shape[0], *x.shape[1:], device=x.device, dtype=x.dtype
                    ),
                ]
            )
        else:
            x = x[:n]
        hwtc = _bchw_to_hwtc(x)
        return OlmoEarthSample(
            sentinel2_l2a=hwtc, timestamps=_timestamps(hwtc.shape[2], device)
        )

    if slug == Slug.TREESATAI:
        return _s2_sample(sample["image_s2"].float(), _s2_names(band_order), device)

    if slug == Slug.BIOMASSTERS:
        x1 = sample["image_s1"].float()
        s1_src = _s1_names(band_order)
        # Average ascending and descending passes to get standard 2-channel (VV, VH).
        vv_idx = [i for i, n in enumerate(s1_src) if "VV" in n.upper()]
        vh_idx = [i for i, n in enumerate(s1_src) if "VH" in n.upper()]
        vv = x1[vv_idx].mean(dim=0)
        vh = x1[vh_idx].mean(dim=0)
        s1 = _bchw_to_hwtc(torch.stack([vv, vh], dim=0))
        s2 = _bchw_to_hwtc(
            _align_s2_to_sentinel2_l2a(
                sample["image_s2"].float(), _s2_names(band_order)
            )
        )
        t = max(s1.shape[2], s2.shape[2])
        return OlmoEarthSample(
            sentinel1=_pad_to_t(s1, t),
            sentinel2_l2a=_pad_to_t(s2, t),
            timestamps=_timestamps(t, device),
        )

    if slug == Slug.KURO_SIWO:
        # Stack pre_1, pre_2, post as 3 SAR timesteps; convert linear power → dB.
        s1 = _bchw_to_hwtc(
            torch.stack(
                [
                    _to_db(sample["image_pre_1"].float()),
                    _to_db(sample["image_pre_2"].float()),
                    _to_db(sample["image_post"].float()),
                ],
                dim=1,
            )
        )  # (C=2, T=3, H, W) → (H, W, 3, C)
        dem = _bchw_to_hwtc(sample["image_dem"].float())
        t = max(s1.shape[2], dem.shape[2])
        return OlmoEarthSample(
            sentinel1=_pad_to_t(s1, t),
            srtm=_pad_to_t(dem, t),
            timestamps=_timestamps(t, device),
        )

    if slug == Slug.SPACENET2:
        # spacenet2: worldview (8-band) + pan (1-band) stacked into the S2 slot.
        parts = [v.float() for k, v in sorted(sample.items()) if k.startswith("image_")]
        x = torch.cat(parts, dim=0)
        n = len(_S2_ORDER)
        if x.shape[0] < n:
            x = torch.cat(
                [
                    x,
                    torch.zeros(
                        n - x.shape[0], *x.shape[1:], device=x.device, dtype=x.dtype
                    ),
                ]
            )
        else:
            x = x[:n]
        hwtc = _bchw_to_hwtc(x)
        return OlmoEarthSample(
            sentinel2_l2a=hwtc, timestamps=_timestamps(hwtc.shape[2], device)
        )

    if slug == Slug.BENV2:
        s2 = _bchw_to_hwtc(
            _align_s2_to_sentinel2_l2a(
                sample["image_s2"].float(), _s2_names(band_order)
            )
        )
        sample_dict: dict[str, Any] = {
            "sentinel2_l2a": s2,
            "timestamps": _timestamps(s2.shape[2], device),
        }
        if "image_s1" in sample:
            s1_src = _s1_names(band_order)
            x1 = sample["image_s1"].float()
            vv_i = next((i for i, n in enumerate(s1_src) if "VV" in n.upper()), 0)
            vh_i = next(
                (i for i, n in enumerate(s1_src) if "VH" in n.upper()),
                min(1, len(s1_src) - 1),
            )
            sample_dict["sentinel1"] = _bchw_to_hwtc(
                torch.stack([x1[vv_i], x1[vh_i]], dim=0)
            )
        return OlmoEarthSample(**sample_dict)

    raise KeyError(f"cannot map GeoBench v2 sample keys={list(sample)} slug={slug}")


def _extract_label(
    sample: dict[str, torch.Tensor],
    slug: str,
    task_type: Any,
    num_classes: int,
) -> torch.Tensor:
    if task_type == TaskType.PER_PIXEL_REGRESSION and "mask" in sample:
        return sample["mask"].float()

    if "mask" in sample:
        return sample["mask"].long().squeeze()

    if "label" in sample:
        y = sample["label"]
        if not torch.is_tensor(y):
            y = torch.as_tensor(y)
        y = y.long()
        return y.squeeze() if y.numel() == 1 else y

    raise KeyError(f"no label/mask in sample keys={list(sample)} slug={slug}")


_MODALITY_NORM_MAP: dict[str, ModalitySpec] = {
    "sentinel2_l2a": Modality.SENTINEL2_L2A,
    "sentinel1": Modality.SENTINEL1,
    "landsat": Modality.LANDSAT,
    "naip": Modality.NAIP,
    "srtm": Modality.SRTM,
}


def _apply_olmoearth_normalization(
    olmo: OlmoEarthSample, normalizer: Normalizer
) -> OlmoEarthSample:
    """Return a new OlmoEarthSample with each present modality normalized to OlmoEarth stats."""
    updates: dict[str, torch.Tensor] = {}
    for field_name, modality_spec in _MODALITY_NORM_MAP.items():
        tensor = getattr(olmo, field_name, None)
        if tensor is None:
            continue
        arr = normalizer.normalize(modality_spec, tensor.numpy())
        updates[field_name] = torch.tensor(arr, dtype=torch.float32)
    return olmo._replace(**updates) if updates else olmo


@lru_cache(maxsize=1)
def _load_geobench_norm_stats() -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Load per-dataset GeoBench normalization stats, keyed by slug.

    The (optional) ``geobench2_norm_stats.json`` resource maps
    ``slug -> modality_name -> band -> {mean, std[, min, max]}``. These are the
    datasets' own published stats, used (instead of the OlmoEarth pretraining
    stats) when we want eval numbers comparable to other models. Returns an empty
    dict if the file is absent, so callers transparently fall back to OlmoEarth
    stats for datasets we don't have stats for.
    """
    resource = (
        files("olmoearth_pretrain.evals.datasets")
        / "config"
        / "geobench2_norm_stats.json"
    )
    if not resource.is_file():
        return {}
    with resource.open() as f:
        return json.load(f)


def _stats_arrays(
    modality_spec: ModalitySpec, band_stats: dict[str, dict[str, float]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Build per-band (mean, std, min, max) arrays in the modality's band order."""
    means, stds, mins, maxs = [], [], [], []
    for band in modality_spec.band_order:
        if band not in band_stats:
            raise ValueError(
                f"missing norm stats for band {band!r} in modality {modality_spec.name}"
            )
        s = band_stats[band]
        means.append(s["mean"])
        stds.append(s["std"])
        mins.append(s.get("min"))
        maxs.append(s.get("max"))
    mins_arr = np.array(mins) if all(m is not None for m in mins) else None
    maxs_arr = np.array(maxs) if all(m is not None for m in maxs) else None
    return np.array(means), np.array(stds), mins_arr, maxs_arr


def _apply_dataset_normalization(
    olmo: OlmoEarthSample,
    dataset_stats: dict[str, dict[str, dict[str, float]]],
    norm_method: str,
) -> OlmoEarthSample:
    """Normalize each present modality with the dataset's own published stats.

    Mirrors :func:`_apply_olmoearth_normalization` but uses ``normalize_bands``
    with the given ``norm_method`` so results match how other models normalize
    these datasets. Raises if a present modality lacks stats, rather than
    silently mixing normalization schemes.
    """
    updates: dict[str, torch.Tensor] = {}
    for field_name, modality_spec in _MODALITY_NORM_MAP.items():
        tensor = getattr(olmo, field_name, None)
        if tensor is None:
            continue
        band_stats = dataset_stats.get(modality_spec.name)
        if band_stats is None:
            raise ValueError(
                f"no dataset norm stats for modality {modality_spec.name}; "
                "stats must cover every modality present in the sample"
            )
        means, stds, mins, maxs = _stats_arrays(modality_spec, band_stats)
        arr = normalize_bands(tensor.numpy(), means, stds, mins, maxs, norm_method)
        updates[field_name] = torch.tensor(arr, dtype=torch.float32)
    return olmo._replace(**updates) if updates else olmo


def _rezero_padded_channels(
    normalized: OlmoEarthSample, pre_norm: OlmoEarthSample
) -> OlmoEarthSample:
    """Re-zero channels that were all-zero before normalization.

    Zero-padded missing bands become large negatives after normalization
    ((0 - mean) / std ≈ -4), but the pretrained model sees 0 for dropped
    bands (band dropout zeroes normalized values before patch embedding).
    Restore those channels to 0 in normalized space to match training.
    """
    updates: dict[str, torch.Tensor] = {}
    for field_name in _MODALITY_NORM_MAP:
        pre = getattr(pre_norm, field_name, None)
        post = getattr(normalized, field_name, None)
        if pre is None or post is None:
            continue
        # (H, W, T, C) — find channels all-zero across every pixel and timestep
        missing = (pre == 0).all(dim=(0, 1, 2))  # shape (C,)
        if missing.any():
            t = post.clone()
            t[:, :, :, missing] = 0.0
            updates[field_name] = t
    return normalized._replace(**updates) if updates else normalized


class GeobenchV2Dataset(Dataset):
    """Wraps a GeoBench v2 tortilla dataset as an OlmoEarth eval dataset."""

    def __init__(
        self,
        dataset: str,
        split: str,
        partition: str,
        norm_stats_from_pretrained: bool = False,
        norm_method: str = "norm_no_clip_2_std",
        norm_strategy: str = "computed",
        tanh_gain: float = 1.0,
    ) -> None:
        """Initialize dataset for the given gb2 slug and split.

        Args:
            dataset: ``gb2-<slug>`` dataset identifier.
            split: One of ``train``, ``valid``, ``test``.
            partition: Unused; splits are fixed per-dataset.
            norm_stats_from_pretrained: If True, always normalize with the
                OlmoEarth pretraining stats. If False, use the dataset's own
                published stats *when available* (see ``geobench2_norm_stats.json``),
                falling back to the pretraining stats otherwise.
            norm_method: Normalization method applied when using dataset stats.
            norm_strategy: Pretraining normalization strategy to use when
                norm_stats_from_pretrained is True (e.g. "computed" or "arcsinh_tanh").
            tanh_gain: The tanh gain, only used for the "arcsinh_tanh" strategy.
        """
        del partition  # splits are fixed per-dataset; partition is not applicable
        if not dataset.startswith("gb2-"):
            raise ValueError(dataset)
        slug = dataset[len("gb2-") :]
        if slug not in SLUG_TO_DATASET:
            raise ValueError(f"unknown gb2 slug: {slug}")
        if split not in ("train", "valid", "test"):
            raise ValueError(split)

        self.config = dataset_to_config(dataset)
        self._slug = slug
        self._normalizer = Normalizer(Strategy(norm_strategy), tanh_gain=tanh_gain)
        self._norm_method = norm_method
        # Use the dataset's own stats only when asked AND we actually have them;
        # otherwise leave None and fall back to OlmoEarth pretraining stats.
        self._dataset_stats: dict[str, dict[str, dict[str, float]]] | None = None
        if not norm_stats_from_pretrained:
            all_stats = _load_geobench_norm_stats()
            self._dataset_stats = all_stats.get(slug)
            # Only flag a genuine gap: stats exist for other datasets but not
            # this one. When no stats file is present at all, the feature simply
            # isn't set up, so fall back silently.
            if all_stats and self._dataset_stats is None:
                logger.warning(
                    "No dataset norm stats for gb2 slug %r; falling back to "
                    "OlmoEarth pretraining stats.",
                    slug,
                )

        loader_cls = SLUG_TO_DATASET[slug]
        root = str(Path(paths.GEOBENCH2_DIR) / slug)
        self._inner = loader_cls(root=root, split=split)
        self._band_order = loader_cls.band_order

    def __len__(self) -> int:
        """Return number of samples in the split."""
        return len(self._inner)

    def __getitem__(self, idx: int) -> tuple[MaskedOlmoEarthSample, torch.Tensor]:
        """Load, normalize, and mask sample at idx, returning (masked_sample, label)."""
        raw = self._inner[idx]
        device = next(
            (v.device for v in raw.values() if torch.is_tensor(v)), torch.device("cpu")
        )
        olmo = _sample_to_olmoearth(raw, self._band_order, self._slug)
        pre_norm = olmo
        if self._dataset_stats is not None:
            olmo = _apply_dataset_normalization(
                olmo, self._dataset_stats, self._norm_method
            )
        else:
            olmo = _apply_olmoearth_normalization(olmo, self._normalizer)
        olmo = _rezero_padded_channels(olmo, pre_norm)
        masked = MaskedOlmoEarthSample.from_olmoearthsample(olmo)
        label = _extract_label(
            raw, self._slug, self.config.task_type, self.config.num_classes
        )
        # Z-score regression targets when train-split stats are configured, so
        # reported RMSE is in standardized units (comparable to GeoBench-2).
        if (
            self.config.task_type == TaskType.PER_PIXEL_REGRESSION
            and self.config.target_mean is not None
            and self.config.target_std is not None
        ):
            label = (label - self.config.target_mean) / self.config.target_std
        if label.device != device:
            label = label.to(device)
        return masked, label

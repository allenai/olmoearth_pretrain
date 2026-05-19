"""GeoBench v2 datasets backed by custom tortilla-based dataloaders."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from einops import rearrange
from torch.utils.data import Dataset

import olmoearth_pretrain.evals.datasets.paths as paths
from olmoearth_pretrain.data.constants import Modality, ModalitySpec
from olmoearth_pretrain.data.dataset import OlmoEarthSample
from olmoearth_pretrain.data.normalize import Normalizer, Strategy
from olmoearth_pretrain.evals.datasets.configs import dataset_to_config
from olmoearth_pretrain.evals.datasets.geobench_v2_loaders import SLUG_TO_DATASET
from olmoearth_pretrain.evals.metrics import SEGMENTATION_IGNORE_LABEL
from olmoearth_pretrain.evals.task_types import TaskType
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

logger = logging.getLogger(__name__)

_S2_ORDER = list(Modality.SENTINEL2_L2A.band_order)


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
    device = next(iter(sample.values())).device

    if slug == "caffe":
        # Grayscale; broadcast to all 12 S2 channels.
        filled = sample["image"].float()[:1].repeat(len(_S2_ORDER), 1, 1)
        hwtc = _bchw_to_hwtc(filled)
        return OlmoEarthSample(
            sentinel2_l2a=hwtc, timestamps=_timestamps(hwtc.shape[2], device)
        )

    if slug in ("burn_scars", "cloudsen12", "spacenet7"):
        image = sample["image"].float()
        src = _s2_names(band_order)
        if slug == "cloudsen12":
            image = image[: len(src)]  # tortilla has 14 bands; truncate to 12
        if slug == "spacenet7":
            rgbn_to_s2 = {"red": "B04", "green": "B03", "blue": "B02", "nir": "B08"}
            src = [rgbn_to_s2.get(s, s) for s in src]
        return _s2_sample(image, src, device)

    if slug == "treesatai":
        return _s2_sample(sample["image_s2"].float(), _s2_names(band_order), device)

    if slug == "biomassters":
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

    if slug == "kuro_siwo":
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

    if slug.startswith("spacenet"):
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

    if slug == "benv2":
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
    if task_type == TaskType.REGRESSION and "mask" in sample:
        return torch.nanmean(sample["mask"].float()).unsqueeze(0)

    if "mask" in sample:
        m = sample["mask"].long().squeeze()
        if slug == "burn_scars":
            m = torch.where(m == 2, torch.full_like(m, SEGMENTATION_IGNORE_LABEL), m)
        return m

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


class GeobenchV2Dataset(Dataset):
    """Wraps a GeoBench v2 tortilla dataset as an OlmoEarth eval dataset."""

    def __init__(
        self,
        dataset: str,
        split: str,
        partition: str,
        norm_stats_from_pretrained: bool = False,
        norm_method: str = "norm_no_clip_2_std",
    ) -> None:
        """Initialize dataset for the given gb2 slug and split."""
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
        self._normalizer = Normalizer(Strategy.COMPUTED)

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
        olmo = _apply_olmoearth_normalization(olmo, self._normalizer)
        masked = MaskedOlmoEarthSample.from_olmoearthsample(olmo)
        label = _extract_label(
            raw, self._slug, self.config.task_type, self.config.num_classes
        )
        if label.device != device:
            label = label.to(device)
        return masked, label

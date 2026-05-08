"""GeoBench v2 datasets backed by custom tortilla-based dataloaders."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

import olmoearth_pretrain.evals.datasets.paths as paths
from olmoearth_pretrain.data.constants import Modality, ModalitySpec
from olmoearth_pretrain.data.dataset import OlmoEarthSample
from olmoearth_pretrain.evals.datasets.configs import dataset_to_config
from olmoearth_pretrain.evals.datasets.geobench_v2_loaders import SLUG_TO_DATASET
from olmoearth_pretrain.evals.metrics import SEGMENTATION_IGNORE_LABEL
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

logger = logging.getLogger(__name__)


def _s2_names(band_order: Any) -> list[str]:
    if isinstance(band_order, dict) and "s2" in band_order:
        return [str(b) for b in band_order["s2"]]
    if isinstance(band_order, list | tuple):
        return [str(b) for b in band_order]
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
    b, _, h, w = x.shape
    idx = {n: i for i, n in enumerate(source_names)}
    out = torch.zeros((b, len(target_names), h, w), dtype=x.dtype, device=x.device)
    for j, name in enumerate(target_names):
        if name in idx:
            out[:, j] = x[:, idx[name]]
    return out


def _align_s2_to_sentinel2_l2a(
    x: torch.Tensor, source_names: list[str]
) -> torch.Tensor:
    """Map GeoBench S2 channels to full OlmoEarth SENTINEL2_L2A order (12 bands)."""
    target_names = list(Modality.SENTINEL2_L2A.band_order)
    if x.dim() == 3:
        return _permute_bchw(x.unsqueeze(0), source_names, target_names)[0]
    if x.dim() == 4:
        c, t, h, w = x.shape
        out = torch.zeros((len(target_names), t, h, w), dtype=x.dtype, device=x.device)
        idx = {n: i for i, n in enumerate(source_names)}
        for j, name in enumerate(target_names):
            if name in idx:
                out[j] = x[idx[name]]
        return out
    raise ValueError(f"expected 3D or 4D S2 tensor, got {x.shape}")


def _bchw_to_hwtc(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        c, h, w = x.shape
        return x.permute(1, 2, 0).unsqueeze(2).contiguous()
    if x.dim() == 4:
        c, t, h, w = x.shape
        return x.permute(2, 3, 1, 0).contiguous()
    raise ValueError(f"expected 3D or 4D image tensor, got {x.shape}")


def _timestamps(
    t: int, device: torch.device, dtype: torch.dtype = torch.long
) -> torch.Tensor:
    day = torch.full((t,), 15, device=device, dtype=dtype)
    month = torch.full((t,), 6, device=device, dtype=dtype)
    year = torch.full((t,), 2020, device=device, dtype=dtype)
    return torch.stack([day, month, year], dim=-1)


def _landsat_from_list(x: torch.Tensor, geo_names: list[str]) -> torch.Tensor:
    idx = {str(n): i for i, n in enumerate(geo_names)}
    order = list(Modality.LANDSAT.band_order)
    mapping = {
        "B8": "B8A",
        "B2": "B02",
        "B3": "B03",
        "B4": "B04",
        "B11": "B11",
        "B10": "B12",
    }
    b, _, h, w = x.shape
    chans: list[torch.Tensor] = []
    for ob in order:
        geo = mapping.get(ob)
        if geo is not None and geo in idx:
            chans.append(x[:, idx[geo] : idx[geo] + 1])
        else:
            chans.append(torch.zeros(b, 1, h, w, device=x.device, dtype=x.dtype))
    return torch.cat(chans, dim=1)


def _stack_image_keys(
    sample: dict[str, torch.Tensor], prefix: str = "image_"
) -> torch.Tensor:
    keys = sorted(
        k for k in sample if k.startswith(prefix) and torch.is_tensor(sample[k])
    )
    parts = []
    for k in keys:
        t = sample[k]
        if t.dim() == 3:
            parts.append(t)
        elif t.dim() == 4:
            c, ti, h, w = t.shape
            parts.append(t.reshape(c * ti, h, w))
        else:
            raise ValueError(f"unexpected shape for {k}: {t.shape}")
    return torch.cat(parts, dim=0)


def _sample_to_olmoearth(
    sample: dict[str, torch.Tensor],
    band_order: Any,
    slug: str,
) -> OlmoEarthSample:
    device = next(iter(sample.values())).device
    sample_dict: dict[str, Any] = {}

    if slug in ("burn_scars", "caffe") and "image" in sample:
        g = sample["image"].float()
        if g.dim() == 3:
            g = g.unsqueeze(0)
        g13 = g[0, :1].repeat(13, 1, 1)
        hwtc = _bchw_to_hwtc(g13)
        t = hwtc.shape[2]
        sample_dict["sentinel2_l2a"] = hwtc
        sample_dict["timestamps"] = _timestamps(t, device)
        return OlmoEarthSample(**sample_dict)

    # GeoBench single-modality S2 uses key "image" (see _rearrange_bands_single_modality).
    if slug == "cloudsen12" and "image" in sample:
        x = sample["image"].float()
        xb = x.unsqueeze(0) if x.dim() == 3 else x
        if isinstance(band_order, list | tuple):
            src = [str(b) for b in band_order]
        else:
            src = _s2_names(band_order) or [str(i) for i in range(xb.shape[1])]
        if len(src) != xb.shape[1]:
            raise ValueError(
                f"cloudsen12: band_order length {len(src)} != image channels {xb.shape[1]}"
            )
        s2_order = list(Modality.SENTINEL2_L2A.band_order)
        x_perm = _permute_bchw(xb, src, s2_order)
        hwtc = _bchw_to_hwtc(x_perm[0])
        t = hwtc.shape[2]
        sample_dict["sentinel2_l2a"] = hwtc
        sample_dict["timestamps"] = _timestamps(t, device)
        return OlmoEarthSample(**sample_dict)

    # Substation: S2 stack under "image" plus detection fields (bbox, label, mask).
    if slug == "substation" and "image" in sample:
        x = sample["image"].float()
        xb = x.unsqueeze(0) if x.dim() == 3 else x
        if isinstance(band_order, list | tuple):
            src = [str(b) for b in band_order]
        else:
            src = _s2_names(band_order) or [str(i) for i in range(xb.shape[1])]
        if len(src) != xb.shape[1]:
            raise ValueError(
                f"substation: band_order length {len(src)} != image channels {xb.shape[1]}"
            )
        s2_order = list(Modality.SENTINEL2_L2A.band_order)
        x_perm = _permute_bchw(xb, src, s2_order)
        hwtc = _bchw_to_hwtc(x_perm[0])
        t = hwtc.shape[2]
        sample_dict["sentinel2_l2a"] = hwtc
        sample_dict["timestamps"] = _timestamps(t, device)
        return OlmoEarthSample(**sample_dict)

    # SpaceNet7 returns a single C×H×W "image" (PlanetScope), not image_* keys. The generic
    # 3-channel "image" branch below maps to naip, which then mismatches gb2_spacenet7's
    # SENTINEL2_L2A input_modalities and yields an empty modality list in the ViT.
    if slug == "spacenet7" and "image" in sample:
        x = sample["image"].float()
        xb = x.unsqueeze(0) if x.dim() == 3 else x
        if xb.dim() == 3:
            xb = xb.unsqueeze(0)
        if isinstance(band_order, list | tuple):
            src = [str(b) for b in band_order]
        else:
            src = [str(i) for i in range(xb.shape[1])]
        if len(src) != xb.shape[1]:
            raise ValueError(
                f"spacenet7: band_order length {len(src)} != image channels {xb.shape[1]}"
            )
        rgbn_to_s2 = {"red": "B04", "green": "B03", "blue": "B02", "nir": "B08"}
        src_s2 = [rgbn_to_s2.get(s, s) for s in src]
        s2_order = list(Modality.SENTINEL2_L2A.band_order)
        x_perm = _permute_bchw(xb, src_s2, s2_order)
        hwtc = _bchw_to_hwtc(x_perm[0])
        t = hwtc.shape[2]
        sample_dict["sentinel2_l2a"] = hwtc
        sample_dict["timestamps"] = _timestamps(t, device)
        return OlmoEarthSample(**sample_dict)

    if slug == "biomassters" and "image_s1" in sample and "image_s2" in sample:
        s1 = sample["image_s1"].float()
        s2 = sample["image_s2"].float()
        s2_src = _s2_names(band_order)
        if not s2_src:
            raise ValueError("biomassters requires band_order with s2 bands")
        s2 = _align_s2_to_sentinel2_l2a(s2, s2_src)
        s1_hwtc = _bchw_to_hwtc(s1)
        s2_hwtc = _bchw_to_hwtc(s2)
        t = max(s1_hwtc.shape[2], s2_hwtc.shape[2])
        if s1_hwtc.shape[2] < t:
            s1_hwtc = torch.cat(
                [
                    s1_hwtc,
                    torch.zeros(
                        *s1_hwtc.shape[:2],
                        t - s1_hwtc.shape[2],
                        s1_hwtc.shape[3],
                        device=device,
                        dtype=s1_hwtc.dtype,
                    ),
                ],
                dim=2,
            )
        if s2_hwtc.shape[2] < t:
            s2_hwtc = torch.cat(
                [
                    s2_hwtc,
                    torch.zeros(
                        *s2_hwtc.shape[:2],
                        t - s2_hwtc.shape[2],
                        s2_hwtc.shape[3],
                        device=device,
                        dtype=s2_hwtc.dtype,
                    ),
                ],
                dim=2,
            )
        sample_dict["sentinel1"] = s1_hwtc
        sample_dict["sentinel2_l2a"] = s2_hwtc
        sample_dict["timestamps"] = _timestamps(t, device)
        return OlmoEarthSample(**sample_dict)

    if slug == "kuro_siwo" and "image_post" in sample:
        x1 = sample["image_post"].float().unsqueeze(0)
        s1_hwtc = _bchw_to_hwtc(x1[0])
        dem = sample["image_dem"].float()
        if dem.dim() == 3:
            dem = dem.unsqueeze(1)
        dem_hwtc = _bchw_to_hwtc(dem)
        t = max(s1_hwtc.shape[2], dem_hwtc.shape[2])
        if s1_hwtc.shape[2] < t:
            s1_hwtc = torch.cat(
                [
                    s1_hwtc,
                    torch.zeros(
                        *s1_hwtc.shape[:2],
                        t - s1_hwtc.shape[2],
                        s1_hwtc.shape[3],
                        device=device,
                        dtype=s1_hwtc.dtype,
                    ),
                ],
                dim=2,
            )
        if dem_hwtc.shape[2] < t:
            dem_hwtc = torch.cat(
                [
                    dem_hwtc,
                    torch.zeros(
                        *dem_hwtc.shape[:2],
                        t - dem_hwtc.shape[2],
                        dem_hwtc.shape[3],
                        device=device,
                        dtype=dem_hwtc.dtype,
                    ),
                ],
                dim=2,
            )
        sample_dict["sentinel1"] = s1_hwtc
        sample_dict["srtm"] = dem_hwtc
        sample_dict["timestamps"] = _timestamps(t, device)
        return OlmoEarthSample(**sample_dict)

    # TreeSatAI exposes aerial + S2 + S1; the generic image_aerial branch only fills naip and
    # returns early, but gb2_treesatai uses SENTINEL2_L2A — route S2 explicitly first.
    if slug == "treesatai" and "image_s2" in sample:
        s2_order = list(Modality.SENTINEL2_L2A.band_order)
        x = sample["image_s2"].float()
        if x.dim() == 4:
            xb = x.unsqueeze(0)
        elif x.dim() == 3:
            xb = x.unsqueeze(0)
        else:
            raise ValueError(f"treesatai image_s2 shape {x.shape}")
        src = _s2_names(band_order)
        if not src:
            raise ValueError("treesatai requires band_order dict with an 's2' key")
        c_in = xb.shape[1]
        if len(src) != c_in:
            raise ValueError(
                f"treesatai s2 band_order length {len(src)} != channels {c_in}"
            )
        x_perm = _permute_bchw(xb, src, s2_order)
        hwtc = _bchw_to_hwtc(x_perm[0])
        t = hwtc.shape[2]
        sample_dict["sentinel2_l2a"] = hwtc
        sample_dict["timestamps"] = _timestamps(t, device)
        return OlmoEarthSample(**sample_dict)

    if "image_aerial" in sample:
        ae = sample["image_aerial"].float()
        naip_hwtc = _bchw_to_hwtc(ae)
        t = naip_hwtc.shape[2]
        sample_dict["naip"] = naip_hwtc
        sample_dict["timestamps"] = _timestamps(t, device)
        if "image_elevation" in sample:
            el = sample["image_elevation"].float()
            if el.dim() == 3:
                el = el.unsqueeze(1)
            dem_hwtc = _bchw_to_hwtc(el)
            if dem_hwtc.shape[2] != t:
                dem_hwtc = dem_hwtc[:, :, :t, :]
            sample_dict["srtm"] = dem_hwtc
        return OlmoEarthSample(**sample_dict)

    if slug.startswith("spacenet") and any(k.startswith("image_") for k in sample):
        x = _stack_image_keys(sample).unsqueeze(0)
        s2_order = list(Modality.SENTINEL2_L2A.band_order)
        n_out = len(s2_order)
        if x.shape[1] > n_out:
            x = x[:, :n_out]
        else:
            pad = torch.zeros(
                1,
                n_out - x.shape[1],
                x.shape[2],
                x.shape[3],
                device=x.device,
                dtype=x.dtype,
            )
            x = torch.cat([x, pad], dim=1)
        hwtc = _bchw_to_hwtc(x[0])
        t = hwtc.shape[2]
        sample_dict["sentinel2_l2a"] = hwtc
        sample_dict["timestamps"] = _timestamps(t, device)
        return OlmoEarthSample(**sample_dict)

    if (
        "image" in sample
        and torch.is_tensor(sample["image"])
        and sample["image"].shape[0] == 3
    ):
        x = sample["image"].float()
        x = torch.cat(
            [x, torch.zeros(1, x.shape[1], x.shape[2], device=x.device, dtype=x.dtype)],
            dim=0,
        )
        hwtc = _bchw_to_hwtc(x)
        t = hwtc.shape[2]
        sample_dict["naip"] = hwtc
        sample_dict["timestamps"] = _timestamps(t, device)
        return OlmoEarthSample(**sample_dict)

    if "image_rgbn" in sample or "image_sar" in sample:
        parts = []
        if "image_rgbn" in sample:
            parts.append(sample["image_rgbn"].float())
        if "image_sar" in sample:
            sar = sample["image_sar"].float()
            if sar.shape[0] >= 2:
                parts.append(sar[:2])
        x = torch.cat(parts, dim=0).unsqueeze(0)
        s2_order = list(Modality.SENTINEL2_L2A.band_order)
        if x.shape[1] > len(s2_order):
            x = x[:, : len(s2_order)]
        else:
            pad = torch.zeros(
                1,
                len(s2_order) - x.shape[1],
                x.shape[2],
                x.shape[3],
                device=x.device,
                dtype=x.dtype,
            )
            x = torch.cat([x, pad], dim=1)
        hwtc = _bchw_to_hwtc(x[0])
        t = hwtc.shape[2]
        sample_dict["sentinel2_l2a"] = hwtc
        sample_dict["timestamps"] = _timestamps(t, device)
        return OlmoEarthSample(**sample_dict)

    if "image_s2" in sample:
        s2_order = list(Modality.SENTINEL2_L2A.band_order)
        x = sample["image_s2"].float()
        if x.dim() == 4:
            xb = x.unsqueeze(0)
        elif x.dim() == 3:
            xb = x.unsqueeze(0)
        else:
            raise ValueError(f"image_s2 shape {x.shape}")
        src = _s2_names(band_order)
        if not src:
            raise ValueError("image_s2 requires band_order with s2 key")
        c_in = xb.shape[1]
        if len(src) != c_in:
            raise ValueError(f"s2 band_order length {len(src)} != channels {c_in}")
        x_perm = _permute_bchw(xb, src, s2_order)
        hwtc = _bchw_to_hwtc(x_perm[0])
        t = hwtc.shape[2]
        sample_dict["sentinel2_l2a"] = hwtc
        sample_dict["timestamps"] = _timestamps(t, device)
        if "image_s1" in sample:
            x1 = sample["image_s1"].float()
            x1b = x1.unsqueeze(0) if x1.dim() == 3 else x1
            if x1b.dim() == 3:
                x1b = x1b.unsqueeze(0)
            s1_src = _s1_names(band_order)
            if not s1_src:
                s1_src = [str(i) for i in range(x1b.shape[1])]
            vv_i = next((s1_src.index(n) for n in s1_src if "VV" in n.upper()), 0)
            vh_i = next(
                (s1_src.index(n) for n in s1_src if "VH" in n.upper()),
                min(1, len(s1_src) - 1),
            )
            x1p = torch.stack([x1b[0, vv_i], x1b[0, vh_i]], dim=0).unsqueeze(0)
            s1_hwtc = _bchw_to_hwtc(x1p[0])
            sample_dict["sentinel1"] = s1_hwtc
        return OlmoEarthSample(**sample_dict)

    keys = [k for k in sample if k.startswith("image_") and torch.is_tensor(sample[k])]
    if keys:
        x = _stack_image_keys(sample).unsqueeze(0)
        s2_order = list(Modality.SENTINEL2_L2A.band_order)
        if x.shape[1] > len(s2_order):
            x = x[:, : len(s2_order)]
        else:
            pad = torch.zeros(
                1,
                len(s2_order) - x.shape[1],
                x.shape[2],
                x.shape[3],
                device=x.device,
                dtype=x.dtype,
            )
            x = torch.cat([x, pad], dim=1)
        hwtc = _bchw_to_hwtc(x[0])
        t = hwtc.shape[2]
        sample_dict["sentinel2_l2a"] = hwtc
        sample_dict["timestamps"] = _timestamps(t, device)
        return OlmoEarthSample(**sample_dict)

    raise KeyError(f"cannot map GeoBench v2 sample keys={list(sample)} slug={slug}")


def _extract_label(
    sample: dict[str, torch.Tensor],
    slug: str,
    task_type: Any,
    num_classes: int,
) -> torch.Tensor:
    from olmoearth_pretrain.evals.task_types import TaskType

    if task_type == TaskType.REGRESSION and "mask" in sample:
        m = sample["mask"].float()
        v = torch.nanmean(m)
        return v.unsqueeze(0)

    # Must run before the generic "mask" segmentation branch: substation carries instance masks.
    if slug == "substation" and "label" in sample:
        labs = sample["label"]
        if not torch.is_tensor(labs):
            labs = torch.as_tensor(labs, dtype=torch.long)
        if labs.numel() == 0:
            return torch.zeros(1, dtype=torch.long)
        # COCO-style category id for the single foreground class (power_station).
        return torch.tensor(int((labs == 1).any()), dtype=torch.long).unsqueeze(0)

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
        if y.numel() == 1:
            return y.squeeze()  # 0D scalar — batches to (N,) as cross_entropy expects
        return y

    raise KeyError(f"no label/mask in sample keys={list(sample)} slug={slug}")


# Mapping from OlmoEarthSample field names to ModalitySpec for normalization.
_MODALITY_NORM_MAP: dict[str, ModalitySpec] = {
    "sentinel2_l2a": Modality.SENTINEL2_L2A,
    "sentinel1": Modality.SENTINEL1,
    "landsat": Modality.LANDSAT,
    "naip": Modality.NAIP,
    "srtm": Modality.SRTM,
}


def _normalize_tensor_olmoearth(
    tensor: torch.Tensor,
    modality_spec: ModalitySpec,
    norm_config: dict,
    std_mult: float = 2.0,
) -> torch.Tensor:
    """Normalize a [H, W, T, C] tensor using OlmoEarth pretraining (NORM_NO_CLIP_2_STD) stats.

    Only the first C band stats are used when C < len(modality_spec.band_order),
    which handles datasets that expose fewer channels than the full modality (e.g.
    aerial RGB with 3 channels in a 4-channel NAIP slot).
    """
    arr = tensor.numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
    n_channels = arr.shape[-1]
    band_order = list(modality_spec.band_order)
    modality_values = norm_config.get(modality_spec.name, {})

    means: list[float] = []
    stds: list[float] = []
    for i in range(n_channels):
        band = band_order[i] if i < len(band_order) else None
        stats = modality_values.get(str(band)) if band is not None else None
        if stats is None:
            means.append(0.0)
            stds.append(1.0)
        else:
            means.append(float(stats["mean"]))
            stds.append(float(stats["std"]))

    means_arr = np.array(means, dtype=np.float32)
    stds_arr = np.array(stds, dtype=np.float32)
    lo = means_arr - std_mult * stds_arr
    hi = means_arr + std_mult * stds_arr
    denom = hi - lo
    denom[denom == 0.0] = 1.0  # avoid div-by-zero for constant bands
    return torch.tensor((arr.astype(np.float32) - lo) / denom, dtype=torch.float32)


def _apply_olmoearth_normalization(
    olmo: OlmoEarthSample, norm_config: dict
) -> OlmoEarthSample:
    """Return a new OlmoEarthSample with each present modality normalized to OlmoEarth stats."""
    updates: dict[str, torch.Tensor] = {}
    for field_name, modality_spec in _MODALITY_NORM_MAP.items():
        tensor = getattr(olmo, field_name, None)
        if tensor is None:
            continue
        updates[field_name] = _normalize_tensor_olmoearth(
            tensor, modality_spec, norm_config
        )
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
        """Initialize the dataset loader and normalization config."""
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

        from olmoearth_pretrain.data.normalize import load_computed_config

        self._olmoearth_norm_config = load_computed_config()

        loader_cls = SLUG_TO_DATASET[slug]
        root = str(Path(paths.GEOBENCH2_DIR) / slug)
        self._inner = loader_cls(root=root, split=split)
        self._band_order = loader_cls.band_order

    def __len__(self) -> int:  # noqa: D105
        return len(self._inner)

    def __getitem__(self, idx: int) -> tuple[MaskedOlmoEarthSample, torch.Tensor]:  # noqa: D105
        raw = self._inner[idx]
        device = next(
            (v.device for v in raw.values() if torch.is_tensor(v)), torch.device("cpu")
        )

        olmo = _sample_to_olmoearth(raw, self._band_order, self._slug)
        olmo = _apply_olmoearth_normalization(olmo, self._olmoearth_norm_config)

        masked = MaskedOlmoEarthSample.from_olmoearthsample(olmo)
        label = _extract_label(
            raw, self._slug, self.config.task_type, self.config.num_classes
        )
        if label.device != device:
            label = label.to(device)
        return masked, label

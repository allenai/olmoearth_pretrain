"""GeoBench v2 datasets via official Lightning DataModules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

import olmoearth_pretrain.evals.datasets.paths as paths
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataset import OlmoEarthSample
from olmoearth_pretrain.evals.datasets.configs import dataset_to_config
from olmoearth_pretrain.evals.metrics import SEGMENTATION_IGNORE_LABEL
from olmoearth_pretrain.train.masking import MaskValue, MaskedOlmoEarthSample

_SLUG_TO_DM: dict[str, type] = {}


def _load_datamodule_classes() -> None:
    if _SLUG_TO_DM:
        return
    from geobench_v2.datamodules import (
        GeoBenchBENV2DataModule,
        GeoBenchBioMasstersDataModule,
        GeoBenchBurnScarsDataModule,
        GeoBenchCaFFeDataModule,
        GeoBenchCloudSen12DataModule,
        GeoBenchDynamicEarthNetDataModule,
        GeoBenchEverWatchDataModule,
        GeoBenchFLAIR2DataModule,
        GeoBenchFieldsOfTheWorldDataModule,
        GeoBenchForestnetDataModule,
        GeoBenchKuroSiwoDataModule,
        GeoBenchNZCattleDataModule,
        GeoBenchPASTISDataModule,
        GeoBenchSo2SatDataModule,
        GeoBenchSpaceNet2DataModule,
        GeoBenchSpaceNet7DataModule,
        GeoBenchSubstationDataModule,
        GeoBenchTreeSatAIDataModule,
    )

    _SLUG_TO_DM.update(
        {
            "benv2": GeoBenchBENV2DataModule,
            "biomassters": GeoBenchBioMasstersDataModule,
            "burn_scars": GeoBenchBurnScarsDataModule,
            "caffe": GeoBenchCaFFeDataModule,
            "cloudsen12": GeoBenchCloudSen12DataModule,
            "dynamic_earthnet": GeoBenchDynamicEarthNetDataModule,
            "everwatch": GeoBenchEverWatchDataModule,
            "flair2": GeoBenchFLAIR2DataModule,
            "forestnet": GeoBenchForestnetDataModule,
            "fotw": GeoBenchFieldsOfTheWorldDataModule,
            "kuro_siwo": GeoBenchKuroSiwoDataModule,
            "nzcattle": GeoBenchNZCattleDataModule,
            "pastis": GeoBenchPASTISDataModule,
            "so2sat": GeoBenchSo2SatDataModule,
            "spacenet2": GeoBenchSpaceNet2DataModule,
            "spacenet7": GeoBenchSpaceNet7DataModule,
            "substation": GeoBenchSubstationDataModule,
            "treesatai": GeoBenchTreeSatAIDataModule,
        }
    )


_SLUG_EXTRA_DM_KWARGS: dict[str, dict[str, Any]] = {
    "kuro_siwo": {"time_step": ["post"]},
    "pastis": {"num_time_steps": 10},
}


def _s2_names(band_order: Any) -> list[str]:
    if isinstance(band_order, dict) and "s2" in band_order:
        return [str(b) for b in band_order["s2"]]
    if isinstance(band_order, (list, tuple)):
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


def _align_s2_to_sentinel2_l2a(x: torch.Tensor, source_names: list[str]) -> torch.Tensor:
    """Map GeoBench S2 channels to full OlmoEarth SENTINEL2_L2A order (12 bands)."""
    target_names = list(Modality.SENTINEL2_L2A.band_order)
    if x.dim() == 3:
        return _permute_bchw(x.unsqueeze(0), source_names, target_names)[0]
    if x.dim() == 4:
        c, t, h, w = x.shape
        out = torch.zeros(
            (len(target_names), t, h, w), dtype=x.dtype, device=x.device
        )
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


def _timestamps(t: int, device: torch.device, dtype: torch.dtype = torch.long) -> torch.Tensor:
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


def _stack_image_keys(sample: dict[str, torch.Tensor], prefix: str = "image_") -> torch.Tensor:
    keys = sorted(k for k in sample if k.startswith(prefix) and torch.is_tensor(sample[k]))
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

    if slug == "fotw" and ("image_a" in sample or "image_b" in sample):
        xa = sample.get("image_a")
        xb = sample.get("image_b")
        if xa is not None and xb is not None:
            x = (xa.float() + xb.float()) * 0.5
        elif xa is not None:
            x = xa.float()
        else:
            x = xb.float()
        hwtc = _bchw_to_hwtc(x)
        t = hwtc.shape[2]
        sample_dict["naip"] = hwtc
        sample_dict["timestamps"] = _timestamps(t, device)
        return OlmoEarthSample(**sample_dict)

    if slug == "forestnet" and "image" in sample:
        x = sample["image"].float()
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if isinstance(band_order, dict):
            geo_names = [str(n) for n in next(iter(band_order.values()))]
        elif isinstance(band_order, (list, tuple)):
            geo_names = [str(n) for n in band_order]
        else:
            raise TypeError(f"forestnet unexpected band_order type: {type(band_order)}")
        ls = _landsat_from_list(x, geo_names)
        ls = ls.squeeze(0)
        hwtc = _bchw_to_hwtc(ls)
        t = hwtc.shape[2]
        sample_dict["landsat"] = hwtc
        sample_dict["timestamps"] = _timestamps(t, device)
        return OlmoEarthSample(**sample_dict)

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
        if isinstance(band_order, (list, tuple)):
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
        if isinstance(band_order, (list, tuple)):
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
        if isinstance(band_order, (list, tuple)):
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

    if slug == "pastis" and "image_s2" in sample:
        s2 = sample["image_s2"].float()
        s2_src = _s2_names(band_order)
        if not s2_src:
            s2_src = [str(i) for i in range(s2.shape[0])]
        s2 = _align_s2_to_sentinel2_l2a(s2, s2_src)
        s2_hwtc = _bchw_to_hwtc(s2)
        asc = sample["image_s1_asc"].float()
        # GeoBench PASTIS is C,T,H,W for num_time_steps>1. Use VV/VH (first two channels), full time.
        if asc.dim() == 4:
            s1_bt = asc[:2]
        elif asc.dim() == 3:
            s1_bt = asc[:2]
        else:
            raise ValueError(f"pastis unexpected image_s1_asc shape {asc.shape}")
        if s1_bt.shape[0] < 2:
            pad = torch.zeros(2 - s1_bt.shape[0], *s1_bt.shape[1:], device=device, dtype=s1_bt.dtype)
            s1_bt = torch.cat([s1_bt, pad], dim=0)
        s1_hwtc = _bchw_to_hwtc(s1_bt)
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
        sample_dict["sentinel2_l2a"] = s2_hwtc
        sample_dict["sentinel1"] = s1_hwtc
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

    if "image_planet" in sample and "image_s2" in sample:
        pl = sample["image_planet"].float()
        if pl.dim() == 4:
            pl = pl[:, 0]
        naip_hwtc = _bchw_to_hwtc(pl)
        s2 = sample["image_s2"].float()
        s2b = s2.unsqueeze(0) if s2.dim() == 3 else s2
        if s2b.dim() == 3:
            s2b = s2b.unsqueeze(0)
        src = _s2_names(band_order) or [str(i) for i in range(s2b.shape[1])]
        s2_perm = _permute_bchw(s2b, src, list(Modality.SENTINEL2_L2A.band_order))
        s2_hwtc = _bchw_to_hwtc(s2_perm[0])
        t = max(naip_hwtc.shape[2], s2_hwtc.shape[2])
        sample_dict["naip"] = naip_hwtc
        sample_dict["sentinel2_l2a"] = s2_hwtc
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
            raise ValueError(f"treesatai s2 band_order length {len(src)} != channels {c_in}")
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
            pad = torch.zeros(1, n_out - x.shape[1], x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        hwtc = _bchw_to_hwtc(x[0])
        t = hwtc.shape[2]
        sample_dict["sentinel2_l2a"] = hwtc
        sample_dict["timestamps"] = _timestamps(t, device)
        return OlmoEarthSample(**sample_dict)

    if "image" in sample and torch.is_tensor(sample["image"]) and sample["image"].shape[0] == 3:
        x = sample["image"].float()
        x = torch.cat(
            [x, torch.zeros(1, x.shape[1], x.shape[2], device=x.device, dtype=x.dtype)], dim=0
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
                1, len(s2_order) - x.shape[1], x.shape[2], x.shape[3], device=x.device, dtype=x.dtype
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
            vh_i = next((s1_src.index(n) for n in s1_src if "VH" in n.upper()), min(1, len(s1_src) - 1))
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
                1, len(s2_order) - x.shape[1], x.shape[2], x.shape[3], device=x.device, dtype=x.dtype
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

    if slug == "everwatch" and "label" in sample:
        labs = sample["label"].long()
        out = torch.zeros(num_classes, dtype=torch.long)
        for c in labs.tolist():
            if 1 <= c <= num_classes:
                out[c - 1] = 1
        return out

    if slug == "nzcattle" and "label" in sample:
        labs = sample["label"].long()
        cattle = int((labs == 1).any())
        return torch.tensor(cattle, dtype=torch.long)

    if slug == "so2sat" and "label" in sample:
        y = sample["label"]
        if not torch.is_tensor(y):
            y = torch.as_tensor(y)
        # so2sat labels are one-hot encoded; convert to class index
        if y.dim() == 1:
            return y.argmax().long().unsqueeze(0)
        return y.long()

    if "label" in sample:
        y = sample["label"]
        if not torch.is_tensor(y):
            y = torch.as_tensor(y)
        if y.dim() == 0:
            return y.long().unsqueeze(0)
        return y.long()

    raise KeyError(f"no label/mask in sample keys={list(sample)} slug={slug}")


class GeobenchV2Dataset(Dataset):
    def __init__(
        self,
        dataset: str,
        split: str,
        partition: str,
        norm_stats_from_pretrained: bool = False,
        norm_method: str = "norm_no_clip_2_std",
    ) -> None:
        del partition, norm_stats_from_pretrained, norm_method
        _load_datamodule_classes()
        if not dataset.startswith("gb2-"):
            raise ValueError(dataset)
        slug = dataset[len("gb2-") :]
        if slug not in _SLUG_TO_DM:
            raise ValueError(f"unknown gb2 slug: {slug}")
        if split not in ("train", "valid", "test"):
            raise ValueError(split)

        self.config = dataset_to_config(dataset)
        self._slug = slug
        root = Path(paths.GEOBENCH2_DIR) / slug
        dm_cls = _SLUG_TO_DM[slug]
        extra = dict(_SLUG_EXTRA_DM_KWARGS.get(slug, {}))
        self._dm = dm_cls(root=str(root), download=False, **extra)
        self._dm.setup("fit")
        self._dm.setup("test")
        if split == "train":
            self._inner = self._dm.train_dataset
        elif split == "valid":
            self._inner = self._dm.val_dataset
        else:
            self._inner = self._dm.test_dataset
        self._band_order = self._dm.band_order

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> tuple[MaskedOlmoEarthSample, torch.Tensor]:
        raw = {}
        for k, v in self._inner[idx].items():
            raw[k] = v.clone() if torch.is_tensor(v) else v
        device = next(v.device for v in raw.values() if torch.is_tensor(v))
        olmo = _sample_to_olmoearth(raw, self._band_order, self._slug)
        masked = MaskedOlmoEarthSample.from_olmoearthsample(olmo)
        label = _extract_label(raw, self._slug, self.config.task_type, self.config.num_classes)
        if label.device != device:
            label = label.to(device)
        return masked, label

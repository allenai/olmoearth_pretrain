"""Set-Latent Perceiver (SLP) self-supervised encoder.

A self-contained SSL model (ported from the ``earthy`` project's
``SetLatentSSLModel``; see ``docs/perceiver_encoder_spec.md``). Every
modality/band-set is tokenized at its stored resolution into a single token set
carrying two-scale metadata (droppable global GPS / absolute time; always-kept
local Fourier geometry / relative time). The set is compressed through a latent
funnel (weight-shared input reads, nested-K capacity) and anything is read back
out with metadata-only Perceiver-IO queries. Masked tokens are dropped from the
encoder input (MAE-style) and reconstructed against frozen random-projection
targets with a global-pool soft-label InfoNCE objective.

Unlike the repo's ``FlexiVitBase`` encoders, the SLP does not emit
``TokensAndMasks`` and does not reuse ``LatentMIM``: its masking,
metadata-only readout, and soft-InfoNCE-against-frozen-random-targets loss are
integral to the architecture. It consumes the repo's ``MaskedOlmoEarthSample`` /
``OlmoEarthSample`` directly and masks internally (a deliberate divergence from
``MASKING_STRATEGY_REGISTRY``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed import DeviceMesh

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.data.constants import (
    BASE_RESOLUTION,
    IMAGE_TILE_SIZE,
    MISSING_VALUE,
    Modality,
    ModalitySpec,
)
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, OlmoEarthSample
from olmoearth_pretrain.nn.encodings import timestamps_to_days
from olmoearth_pretrain.nn.utils import DistributedMixins

# Reference epoch for absolute time (years-since-2020), matching the earthy SLP.
REFERENCE_YEAR = 2020
MEAN_YEAR_DAYS = 365.2425
# Output grid GSD used only to derive dense-readout query positions.
TOKEN_GSD_M = 60.0
# Local-geometry Fourier band: 40 m (adjacent-token resolution) to 4 km (window).
LOCAL_WAVELENGTHS_M = (40.0, 4000.0)
# Relative-time Fourier band: 5 d to 4000 d (covers multi-year samples w/o alias).
REL_TIME_WAVELENGTHS_D = (5.0, 4000.0)

# Default modalities tokenized by the SLP. Each band set of each modality becomes
# one group; band sets are co-registered on the modality's stored grid.
DEFAULT_MODALITY_NAMES: tuple[str, ...] = (
    "sentinel2_l2a",
    "sentinel1",
    "landsat",
    "era5_10",
)


def fourier_features(
    x: torch.Tensor, n_freqs: int, wavelengths: tuple[float, float]
) -> torch.Tensor:
    """Deterministic sin/cos features of ``x`` at log-spaced frequencies.

    Not learned: the high-frequency components are what an MLP on raw coordinates
    cannot recover (spectral bias). Returns ``(*x.shape, 2 * n_freqs)``.

    Args:
        x: Input tensor (any shape), in the unit implied by ``wavelengths``.
        n_freqs: Number of log-spaced frequencies.
        wavelengths: ``(min, max)`` wavelength in the same unit as ``x``.
    """
    lo, hi = wavelengths
    freqs = (
        2.0
        * math.pi
        / torch.logspace(
            math.log10(lo),
            math.log10(hi),
            n_freqs,
            device=x.device,
            dtype=torch.float32,
        )
    )
    ang = x.float()[..., None] * freqs
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


def _time_features(years: torch.Tensor) -> torch.Tensor:
    """Absolute-time features: (years-since-2020, annual sin, annual cos)."""
    return torch.stack(
        [years, torch.sin(years * 2.0 * math.pi), torch.cos(years * 2.0 * math.pi)],
        dim=-1,
    )


@dataclass(frozen=True)
class SLPGroupSpec:
    """One tokenization group: a single band set of a repo modality.

    Args:
        name: Unique group name (``"{modality}__bs{idx}"``).
        modality: Repo modality name.
        band_indices: Channel indices of this band set in the modality tensor.
        num_bands: Number of channels in the band set.
        stored_gsd_m: Ground sample distance of the modality's stored grid.
        native_gsd_m: Native GSD of the band set (for the record; extents use the
            stored GSD, which is the true footprint of a token).
        patch_px: Conv patch size in stored pixels (spatial groups only).
        is_spatial: Whether the modality has a spatial grid.
        is_multitemporal: Whether the modality varies in time.
    """

    name: str
    modality: str
    band_indices: tuple[int, ...]
    num_bands: int
    stored_gsd_m: float
    native_gsd_m: float
    patch_px: int
    is_spatial: bool
    is_multitemporal: bool

    @property
    def extent_m(self) -> float:
        """Ground extent (metres) of one token."""
        return self.patch_px * self.stored_gsd_m


def _modality_stored_gsd(spec: ModalitySpec) -> float:
    """Stored GSD of a modality's grid in metres.

    Tile ground extent divided by stored pixels per side; uses the registry
    helpers so the negative ``image_tile_size_factor`` convention (coarser
    stored grids, e.g. era5_10 at -256 -> 2560 m) is honoured.
    """
    return spec.get_tile_resolution() * IMAGE_TILE_SIZE / spec.get_expected_tile_size()


def build_groups(
    modality_names: tuple[str, ...] | list[str], token_extent_m: float
) -> tuple[SLPGroupSpec, ...]:
    """Build SLP groups (one per band set) from repo modality specs.

    Spatial groups get a conv patch size targeting ``token_extent_m`` ground
    extent at the modality's stored GSD; non-spatial modalities (e.g. ERA5) are
    one token per timestep.
    """
    groups: list[SLPGroupSpec] = []
    for modality in modality_names:
        spec = Modality.get(modality)
        stored_gsd = _modality_stored_gsd(spec)
        offset = 0
        for idx, band_set in enumerate(spec.band_sets):
            num_bands = len(band_set.bands)
            band_indices = tuple(range(offset, offset + num_bands))
            offset += num_bands
            native_gsd = BASE_RESOLUTION * band_set.resolution_factor
            if spec.is_spatial:
                patch_px = max(1, round(token_extent_m / stored_gsd))
            else:
                patch_px = 1
            groups.append(
                SLPGroupSpec(
                    name=f"{modality}__bs{idx}",
                    modality=modality,
                    band_indices=band_indices,
                    num_bands=num_bands,
                    stored_gsd_m=stored_gsd,
                    native_gsd_m=native_gsd,
                    patch_px=patch_px,
                    is_spatial=spec.is_spatial,
                    is_multitemporal=spec.is_multitemporal,
                )
            )
    return tuple(groups)


class PatchTokenizer(nn.Module):
    """Per-group conv patchifier over the stored grid.

    Non-spatial groups (1x1 grid) fall back to a 1x1 conv, so the same module
    handles ERA5 (one token per timestep) uniformly.
    """

    def __init__(
        self, in_channels: int, dim: int, patch_px: int, valid_threshold: float
    ) -> None:
        """Initialize the patch tokenizer."""
        super().__init__()
        self.patch_px = patch_px
        self.valid_threshold = valid_threshold
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_px, stride=patch_px)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Patchify ``(b, t, c, h, w)`` data into ``(b, t, gh, gw, d)`` tokens."""
        b, t, c, h, w = data.shape
        data = data.to(self.proj.weight.dtype)  # e.g. bf16 under FSDP
        x = self.proj(data.reshape(b * t, c, h, w))
        return x.reshape(b, t, x.shape[1], x.shape[2], x.shape[3]).permute(
            0, 1, 3, 4, 2
        )

    def valid_tokens(self, valid: torch.Tensor) -> torch.Tensor:
        """A token is valid if >= ``valid_threshold`` of its pixels are valid."""
        b, t, h, w = valid.shape
        x = valid.reshape(b * t, 1, h, w).float()
        if self.patch_px > 1:
            x = F.avg_pool2d(x, kernel_size=self.patch_px, stride=self.patch_px)
        return x.reshape(b, t, x.shape[2], x.shape[3]) >= self.valid_threshold - 1e-3


class TwoLayerEncoding(nn.Module):
    """Small MLP encoder for metadata channels."""

    def __init__(self, in_dim: int, token_dim: int) -> None:
        """Initialize the two-layer encoding."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``x`` (cast to the layer's compute dtype, e.g. bf16 FSDP)."""
        return self.net(x.to(self.net[0].weight.dtype))


class CrossBlock(nn.Module):
    """Pre-norm cross-attention + MLP. Queries ``x`` attend to context ``kv``."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0) -> None:
        """Initialize the cross block."""
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(
        self, x: torch.Tensor, kv: torch.Tensor, kv_padding: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Cross-attend ``x`` to ``kv`` (``kv_padding`` True = ignore key)."""
        kv_n = self.norm_kv(kv)
        attended, _ = self.attn(
            self.norm_q(x), kv_n, kv_n, key_padding_mask=kv_padding, need_weights=False
        )
        x = x + attended
        return x + self.mlp(self.norm2(x))


class SelfBlock(nn.Module):
    """Pre-norm self-attention + MLP over the latent set (positionless)."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0) -> None:
        """Initialize the self block."""
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Self-attend over the latent set."""
        h = self.norm1(x)
        attended, _ = self.attn(h, h, h, need_weights=False)
        x = x + attended
        return x + self.mlp(self.norm2(x))


class SetLatentPerceiver(nn.Module, DistributedMixins):
    """Set-Latent Perceiver: one token soup -> latent funnel -> query readout.

    See module docstring and ``docs/perceiver_encoder_spec.md``. Consumes the
    repo's ``MaskedOlmoEarthSample`` / ``OlmoEarthSample`` and is trained with an
    internal masking + soft-InfoNCE objective against frozen random targets.
    """

    supports_multiple_modalities_at_once = True

    def __init__(
        self,
        *,
        supported_modality_names: tuple[str, ...] | list[str] = DEFAULT_MODALITY_NAMES,
        token_extent_m: float = 80.0,
        dim: int = 768,
        heads: int = 12,
        mlp_ratio: float = 4.0,
        latents: int = 1024,
        nested_latents: tuple[int, ...] = (128, 256, 512, 1024),
        self_depth_per_read: int = 4,
        num_input_reads: int = 2,
        downsample_factor: int = 4,
        level2_depth: int = 2,
        decoder_depth: int = 2,
        target_dim: int = 256,
        mask_family_probs: tuple[float, float, float] = (0.3, 0.4, 0.3),
        token_mask_prob: float = 0.3,
        modality_mask_prob: float = 0.35,
        timestep_mask_prob: float = 0.15,
        spatial_block_frac: tuple[float, float] = (0.25, 0.5),
        n_local_freqs: int = 16,
        n_rel_time_freqs: int = 8,
        temperature: float = 0.1,
        max_contrastive_samples: int = 4096,
        loss_mode: str = "infonce",
        contrast_scope: str = "global",
        soft_targets: bool = True,
        label_temperature: float = 0.05,
        valid_token_threshold: float = 0.5,
        cond_dropout: float = 0.5,
        trained_years: tuple[float, float] | None = None,
    ) -> None:
        """Initialize the Set-Latent Perceiver. See ``SetLatentPerceiverConfig``."""
        super().__init__()
        if contrast_scope not in ("global", "temporal"):
            raise ValueError("contrast_scope must be 'global' or 'temporal'")
        if loss_mode not in ("infonce", "cosine", "smooth_l1"):
            raise ValueError("loss_mode must be 'infonce', 'cosine', or 'smooth_l1'")

        self.supported_modality_names = list(supported_modality_names)
        self.token_extent_m = token_extent_m
        self.groups = build_groups(supported_modality_names, token_extent_m)
        self.group_index = {g.name: idx for idx, g in enumerate(self.groups)}

        self.dim = dim
        self.feature_dim = dim
        self.latent_count = latents
        self.nested_latents = tuple(k for k in nested_latents if k <= latents) or (
            latents,
        )
        self.num_input_reads = num_input_reads
        self.downsample_factor = downsample_factor
        self.k2 = max(1, latents // downsample_factor)
        self.mask_family_probs = mask_family_probs
        self.token_mask_prob = token_mask_prob
        self.modality_mask_prob = modality_mask_prob
        self.timestep_mask_prob = timestep_mask_prob
        self.spatial_block_frac = spatial_block_frac
        self.n_local_freqs = n_local_freqs
        self.n_rel_time_freqs = n_rel_time_freqs
        self.temperature = temperature
        self.max_contrastive_samples = max_contrastive_samples
        self.loss_mode = loss_mode
        self.contrast_scope = contrast_scope
        self.soft_targets = soft_targets
        self.label_temperature = label_temperature
        self.valid_token_threshold = valid_token_threshold
        self.cond_dropout = cond_dropout
        self.trained_years = trained_years

        # Online + frozen random-projection tokenizers (one per band-set group).
        self.tokenizers = nn.ModuleDict(
            {
                g.name: PatchTokenizer(
                    g.num_bands, dim, g.patch_px, valid_token_threshold
                )
                for g in self.groups
            }
        )
        self.target_tokenizers = nn.ModuleDict(
            {
                g.name: PatchTokenizer(
                    g.num_bands, target_dim, g.patch_px, valid_token_threshold
                )
                for g in self.groups
            }
        )
        for param in self.target_tokenizers.parameters():
            param.requires_grad = False

        # Metadata encoders. Global GPS + absolute time are droppable; local /
        # relative Fourier channels are always kept.
        self.gps_encoding = TwoLayerEncoding(3, dim)
        self.time_encoding = TwoLayerEncoding(3, dim)
        self.local_pos_proj = nn.Linear(4 * n_local_freqs, dim)  # 2 axes x sin/cos
        self.rel_time_proj = nn.Linear(2 * n_rel_time_freqs, dim)
        self.extent_encoding = TwoLayerEncoding(1, dim)
        self.group_tokens = nn.Embedding(len(self.groups), dim)

        self.latent_pool = nn.Parameter(torch.zeros(latents, dim))
        self.latent_pool2 = nn.Parameter(torch.zeros(self.k2, dim))
        nn.init.normal_(self.latent_pool, std=0.02)
        nn.init.normal_(self.latent_pool2, std=0.02)

        self.read_block = CrossBlock(dim, heads, mlp_ratio)  # weight-shared reads
        self.self_depth_per_read = self_depth_per_read
        self.self_blocks = nn.ModuleList(
            [
                SelfBlock(dim, heads, mlp_ratio)
                for _ in range(self_depth_per_read * num_input_reads)
            ]
        )
        self.down_block = CrossBlock(dim, heads, mlp_ratio)
        self.level2_blocks = nn.ModuleList(
            [SelfBlock(dim, heads, mlp_ratio) for _ in range(level2_depth)]
        )
        self.enc_norm = nn.LayerNorm(dim)
        self.decoder_blocks = nn.ModuleList(
            [CrossBlock(dim, heads, mlp_ratio) for _ in range(decoder_depth)]
        )
        self.head = nn.Linear(dim, target_dim)

    # ------------------------------------------------------------- properties

    @property
    def supported_modalities(self) -> list[str]:
        """Modality names this model tokenizes.

        The eval callback's modality gate compares these against task
        ``input_modalities`` strings.
        """
        return list(self.supported_modality_names)

    @property
    def device(self) -> torch.device:
        """Device of the model parameters."""
        return next(self.parameters()).device

    # ------------------------------------------------------------- tokens

    def _group_data(
        self, sample: OlmoEarthSample | MaskedOlmoEarthSample, g: SLPGroupSpec
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Extract ``(data, valid)`` for a group as ``(b, t, c, h, w)`` tensors.

        Returns None if the modality is absent or has zero timesteps.
        """
        raw = getattr(sample, g.modality, None)
        if raw is None:
            return None
        device = self.device
        raw = raw.to(device=device, dtype=torch.float32)
        # Normalize to (b, h, w, t, c_full).
        if g.is_spatial:
            if raw.ndim != 5:
                raise ValueError(
                    f"expected 5D [B,H,W,T,C] for spatial modality {g.modality}, "
                    f"got shape {tuple(raw.shape)}"
                )
            b, h, w, t, _ = raw.shape
            band = raw[..., list(g.band_indices)]  # (b,h,w,t,c)
            data = band.permute(0, 3, 4, 1, 2).contiguous()  # (b,t,c,h,w)
        else:
            # Time-only (e.g. ERA5): [B, T, C] -> one 1x1 token per timestep.
            if raw.ndim != 3:
                raise ValueError(
                    f"expected 3D [B,T,C] for non-spatial modality {g.modality}, "
                    f"got shape {tuple(raw.shape)}"
                )
            b, t, _ = raw.shape
            band = raw[..., list(g.band_indices)]  # (b,t,c)
            data = band.permute(0, 2, 1).reshape(b, len(g.band_indices), t, 1, 1)
            data = data.permute(0, 2, 1, 3, 4).contiguous()  # (b,t,c,1,1)
        if data.shape[1] == 0:
            return None
        # Validity: robust to float16 (MISSING_VALUE -> -inf) via a threshold.
        valid = torch.isfinite(data) & (data > MISSING_VALUE / 2)
        data = torch.nan_to_num(data.where(valid, torch.zeros_like(data)))
        return data, valid

    def _pad_to_multiple(
        self, data: torch.Tensor, valid: torch.Tensor, patch_px: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad H,W up to a multiple of ``patch_px`` (padding marked invalid)."""
        if patch_px == 1:
            return data, valid
        b, t, c, h, w = data.shape
        pad_h = (-h) % patch_px
        pad_w = (-w) % patch_px
        if pad_h == 0 and pad_w == 0:
            return data, valid
        data = F.pad(data, (0, pad_w, 0, pad_h))
        valid = F.pad(valid, (0, pad_w, 0, pad_h), value=False)
        return data, valid

    def _timestamps(
        self, sample: OlmoEarthSample | MaskedOlmoEarthSample
    ) -> torch.Tensor | None:
        ts = getattr(sample, "timestamps", None)
        if ts is None:
            return None
        return ts.to(device=self.device)

    def _years_since_reference(self, timestamps: torch.Tensor, t: int) -> torch.Tensor:
        """Compute (b, t) years-since-2020 from ``[B, T, 3]`` timestamps."""
        ts = timestamps[:, :t].long()
        days = timestamps_to_days(ts, anchor_year=REFERENCE_YEAR)
        return days / MEAN_YEAR_DAYS

    def _tokenize(
        self,
        sample: OlmoEarthSample | MaskedOlmoEarthSample,
        *,
        generator: torch.Generator | None,
    ) -> tuple[dict[str, dict[str, Any]], int, torch.device]:
        """Build per-group dense token layouts ``(b, t, gh, gw, d)``.

        ``content`` and ``meta`` are kept separate: metadata alone is the decoder
        query (feeding content would leak the reconstruction answer).
        """
        device = self.device
        timestamps = self._timestamps(sample)

        available = [
            g for g in self.groups if getattr(sample, g.modality, None) is not None
        ]
        if not available:
            raise ValueError("sample does not contain any configured modality groups")

        b = None
        drop_gps = drop_time = None
        latlon = getattr(sample, "latlon", None)

        per_group: dict[str, dict[str, Any]] = {}
        for g in available:
            got = self._group_data(sample, g)
            if got is None:
                continue
            data, valid = got
            data, valid = self._pad_to_multiple(data, valid, g.patch_px)
            bg = data.shape[0]
            if b is None:
                b = bg
                # cond-dropout draws are per-sample and shared across groups.
                if self.training and self.cond_dropout > 0.0:
                    drop_gps = (
                        torch.rand(b, device=device, generator=generator)
                        < self.cond_dropout
                    )
                    drop_time = (
                        torch.rand(b, device=device, generator=generator)
                        < self.cond_dropout
                    )

            valid_pixels = valid.all(dim=2)  # (b,t,h,w): all bands valid
            tok = self.tokenizers[g.name]
            content = tok(data)  # (b,t,gh,gw,d)
            with torch.no_grad():
                target = self.target_tokenizers[g.name](data)  # (b,t,gh,gw,d_t)
            valid_tok = tok.valid_tokens(valid_pixels)
            time_mask = valid_pixels.flatten(2).any(dim=2)  # (b,t)
            valid_tok = valid_tok & time_mask[:, :, None, None]

            _, t, gh, gw, _ = content.shape
            extent_m = g.extent_m

            # --- Local geometry (always kept): token-center offsets from origin.
            east = (
                torch.arange(gw, device=device, dtype=torch.float32) + 0.5
            ) * extent_m
            north = (
                torch.arange(gh, device=device, dtype=torch.float32) + 0.5
            ) * extent_m
            local = torch.cat(
                [
                    fourier_features(east, self.n_local_freqs, LOCAL_WAVELENGTHS_M)[
                        None, :, :
                    ].expand(gh, gw, -1),
                    fourier_features(north, self.n_local_freqs, LOCAL_WAVELENGTHS_M)[
                        :, None, :
                    ].expand(gh, gw, -1),
                ],
                dim=-1,
            )
            local_enc = self.local_pos_proj(local.to(self.local_pos_proj.weight.dtype))
            meta = local_enc[None, None] + torch.zeros(
                b, t, 1, 1, self.dim, device=device, dtype=local_enc.dtype
            )

            # --- Global GPS (droppable): sample-level unit-sphere of latlon.
            coords = self._gps_coords(latlon, b, device)  # (b,3)
            if drop_gps is not None and drop_gps.any():
                coords = coords.clone()
                coords[drop_gps] = 0.0
            meta = meta + self.gps_encoding(coords)[:, None, None, None]

            # --- Absolute time (droppable) + relative time within sample (kept).
            if g.is_multitemporal and timestamps is not None:
                years = self._years_since_reference(timestamps, t)  # (b,t)
            else:
                # Static / timeless group: no absolute date.
                years = torch.zeros(b, t, device=device)
                time_mask = torch.zeros(b, t, dtype=torch.bool, device=device)
            feats = _time_features(years)
            timeless = ~time_mask
            if self.trained_years is not None:
                lo, hi = self.trained_years
                out_of_range = (years < lo - 1e-6) | (years > hi + 1e-6)
                timeless = timeless | out_of_range
            if drop_time is not None:
                timeless = timeless | drop_time[:, None]
            if timeless.any():
                feats = feats.clone()
                feats[timeless] = 0.0
            abs_time = self.time_encoding(feats)

            masked_years = years.where(time_mask, torch.full_like(years, float("nan")))
            median = torch.nan_to_num(
                masked_years.nanmedian(dim=1, keepdim=True).values
            )
            rel_days = (years - median) * MEAN_YEAR_DAYS
            rel_days = torch.where(time_mask, rel_days, torch.zeros_like(rel_days))
            rel_time = self.rel_time_proj(
                fourier_features(
                    rel_days, self.n_rel_time_freqs, REL_TIME_WAVELENGTHS_D
                ).to(self.rel_time_proj.weight.dtype)
            )
            meta = meta + (abs_time + rel_time)[:, :, None, None]
            meta = meta + self.group_tokens.weight[self.group_index[g.name]]
            meta = meta + self.extent_encoding(
                torch.log(torch.tensor([[extent_m]], device=device))
            ).view(1, 1, 1, 1, self.dim)

            per_group[g.name] = {
                "content": content,
                "meta": meta,
                "target": target,
                "valid": valid_tok,
                "shape": (t, gh, gw),
            }
        if not per_group or b is None:
            raise ValueError("every configured modality group is empty in this sample")
        return per_group, b, device

    def _gps_coords(
        self, latlon: torch.Tensor | None, b: int, device: torch.device
    ) -> torch.Tensor:
        """Unit-sphere GPS ``(b, 3)`` from ``latlon`` (lat, lon in degrees).

        Never fabricates GPS: a missing latlon becomes the trained null (zero)
        vector, an off-sphere "location unknown" state.
        """
        if latlon is None:
            return torch.zeros(b, 3, device=device)
        latlon = latlon.to(device=device, dtype=torch.float32)
        lat = latlon[:, 0] * math.pi / 180.0
        lon = latlon[:, 1] * math.pi / 180.0
        coords = torch.stack(
            [
                torch.cos(lat) * torch.cos(lon),
                torch.cos(lat) * torch.sin(lon),
                torch.sin(lat),
            ],
            dim=-1,
        )
        # Rows with non-finite georef (placeholder) -> trained null.
        bad = ~torch.isfinite(coords).all(dim=-1)
        if bad.any():
            coords = coords.clone()
            coords[bad] = 0.0
        return coords

    # ------------------------------------------------------------- masking

    def _sample_masks(
        self,
        per_group: dict[str, dict[str, Any]],
        b: int,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> dict[str, torch.Tensor]:
        """Per-sample mask family: random tokens / temporal / spatial block."""
        family = torch.multinomial(
            torch.tensor(self.mask_family_probs, device=device),
            b,
            replacement=True,
            generator=generator,
        )
        max_t = max(g["shape"][0] for g in per_group.values())
        timestep_mask = (
            torch.rand((b, max_t), device=device, generator=generator)
            < self.timestep_mask_prob
        )
        block = torch.rand((b, 4), device=device, generator=generator)

        masks = {}
        for name, g in per_group.items():
            t, gh, gw = g["shape"]
            rand_tok = (
                torch.rand((b, t, gh, gw), device=device, generator=generator)
                < self.token_mask_prob
            )
            modality = (
                torch.rand((b, t), device=device, generator=generator)
                < self.modality_mask_prob
            )
            temporal = (timestep_mask[:, :t] | modality)[:, :, None, None].expand(
                b, t, gh, gw
            )

            lo, hi = self.spatial_block_frac
            frac_h = (lo + (hi - lo) * block[:, 2]).sqrt()
            frac_w = (lo + (hi - lo) * block[:, 3]).sqrt()
            ys = (torch.arange(gh, device=device, dtype=torch.float32) + 0.5)[
                None, :
            ] / max(gh, 1)
            xs = (torch.arange(gw, device=device, dtype=torch.float32) + 0.5)[
                None, :
            ] / max(gw, 1)
            cy = block[:, 0:1] * (1.0 - frac_h[:, None])
            cx = block[:, 1:2] * (1.0 - frac_w[:, None])
            in_y = (ys >= cy) & (ys < cy + frac_h[:, None])
            in_x = (xs >= cx) & (xs < cx + frac_w[:, None])
            spatial = (in_y[:, None, :, None] & in_x[:, None, None, :]).expand(
                b, t, gh, gw
            )

            chosen = torch.where(
                (family == 0)[:, None, None, None],
                rand_tok,
                torch.where((family == 1)[:, None, None, None], temporal, spatial),
            )
            masks[name] = g["valid"] & chosen
        return masks

    def _apply_cloud_mask(
        self,
        masks: dict[str, torch.Tensor],
        per_group: dict[str, dict[str, Any]],
        cloud: dict[str, torch.Tensor] | None,
    ) -> dict[str, torch.Tensor]:
        """Drop cloudy tokens from targets (kept encoder-visible).

        The repo corpus does not expose a per-token cloud modality, so ``cloud``
        is None during ``forward`` and this is a no-op. Retained (and tested
        directly) for parity: ``cloud[modality]`` is a ``(b, t, H, W)`` bool grid
        on the modality's stored grid; a token is cloudy if any stored pixel in
        its patch footprint is cloudy.
        """
        if not cloud:
            return masks
        for name, g in per_group.items():
            spec = self.groups[self.group_index[name]]
            cm = cloud.get(spec.modality)
            if cm is None:
                continue
            t, gh, gw = g["shape"]
            cm = cm.to(device=masks[name].device).bool()
            if cm.shape[1] < t:
                continue
            cm = cm[:, :t].float()
            if spec.patch_px > 1:
                # Pad up like _pad_to_multiple (pad pixels clear) so the pooled
                # grid matches the token grid for non-multiple H/W.
                pad_h = (-cm.shape[2]) % spec.patch_px
                pad_w = (-cm.shape[3]) % spec.patch_px
                if pad_h or pad_w:
                    cm = F.pad(cm, (0, pad_w, 0, pad_h))
                cloudy = F.max_pool2d(
                    cm.reshape(-1, 1, cm.shape[2], cm.shape[3]),
                    kernel_size=spec.patch_px,
                    stride=spec.patch_px,
                ).reshape(cm.shape[0], t, -1, cm.shape[3] // spec.patch_px)
            else:
                cloudy = cm
            cloudy = cloudy[:, :, :gh, :gw] > 0
            masks[name] = masks[name] & ~cloudy
        return masks

    # ------------------------------------------------------------- encoder

    def _sample_k(self, generator: torch.Generator | None, device: torch.device) -> int:
        if not self.training or len(self.nested_latents) == 1:
            return self.latent_count
        idx = int(
            torch.randint(
                len(self.nested_latents), (1,), device=device, generator=generator
            ).item()
        )
        return self.nested_latents[idx]

    def _encode_set(
        self, tokens: torch.Tensor, key_padding: torch.Tensor, k: int
    ) -> torch.Tensor:
        """Compress ``(b, n, d)`` tokens (``key_padding`` True=ignore) to latents."""
        b = tokens.shape[0]
        latents = self.latent_pool[:k][None].expand(b, k, self.dim)
        for read in range(self.num_input_reads):
            latents = self.read_block(latents, tokens, key_padding)
            start = read * self.self_depth_per_read
            for blk in self.self_blocks[start : start + self.self_depth_per_read]:
                latents = blk(latents)
        k2 = max(1, k // self.downsample_factor)
        level2 = self.latent_pool2[:k2][None].expand(b, k2, self.dim)
        level2 = self.down_block(level2, latents)
        for blk in self.level2_blocks:
            level2 = blk(level2)
        return self.enc_norm(level2)

    def _decode(self, queries: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """Metadata-only queries cross-attend the latents; returns ``(..., dim)``."""
        x = queries
        for blk in self.decoder_blocks:
            x = blk(x, latents)
        return x

    # ------------------------------------------------------------- forward

    def forward(
        self,
        sample: OlmoEarthSample | MaskedOlmoEarthSample,
        patch_size: int | None = None,
        *,
        mask_seed: int | None = None,
        k_seed: int | None = None,
        cloud: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run one SSL step, returning ``(loss, metrics)``.

        Args:
            sample: The input sample (masks, if present, are ignored: the SLP
                masks internally from data validity).
            patch_size: Unused; accepted for a uniform training-loop call.
            mask_seed: Per-rank seed for masking (distinct data -> distinct masks).
            k_seed: Rank-free seed for nested-K (must match across DDP ranks).
            cloud: Optional per-modality cloud grids for target exclusion.
        """
        del patch_size
        device = self.device
        generator = None
        if mask_seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(mask_seed)
        # K must match across DDP ranks in a step (rank-free seed); falls back to
        # the mask generator when not provided.
        k_generator = generator
        if k_seed is not None:
            k_generator = torch.Generator(device=device)
            k_generator.manual_seed(k_seed)

        per_group, b, device = self._tokenize(sample, generator=generator)
        target_masks = self._sample_masks(per_group, b, device, generator)
        target_masks = self._apply_cloud_mask(target_masks, per_group, cloud)
        visible = {
            name: g["valid"] & ~target_masks[name] for name, g in per_group.items()
        }
        _ensure_nonempty(target_masks, visible, per_group)

        flat_tokens = torch.cat(
            [(g["content"] + g["meta"]).flatten(1, 3) for g in per_group.values()],
            dim=1,
        )
        flat_visible = torch.cat(
            [visible[name].flatten(1, 3) for name in per_group], dim=1
        )
        k = self._sample_k(k_generator, device)
        latents = self._encode_set(flat_tokens, ~flat_visible, k)

        group_losses: dict[str, torch.Tensor] = {}
        group_correct: dict[str, int] = {}
        group_total: dict[str, int] = {}
        target_count = 0

        for name, g in per_group.items():
            mask = target_masks[name]
            if not mask.any():
                continue
            t, gh, gw = g["shape"]
            queries = g["meta"].reshape(b, t * gh * gw, self.dim)
            pred = self.head(self._decode(queries, latents)).reshape(b, t, gh, gw, -1)
            target = g["target"]
            target_count += int(mask.sum().item())

            parts: list[torch.Tensor] = []
            correct = total = 0
            if self.contrast_scope == "temporal" and self.loss_mode == "infonce":
                multi = mask & (mask.sum(dim=1, keepdim=True) >= 2)
                single = mask & ~multi
                if multi.any():
                    result = temporal_contrastive_loss(
                        pred, target, multi, self.temperature
                    )
                    if result is not None:
                        loss_t, c, n = result
                        parts.append(loss_t)
                        correct += c
                        total += n
                if single.any():
                    loss_g, c, n = prediction_loss(
                        pred[single],
                        target[single],
                        mode=self.loss_mode,
                        temperature=self.temperature,
                        max_samples=self.max_contrastive_samples,
                    )
                    parts.append(loss_g)
                    correct += c
                    total += n
            elif self.loss_mode == "infonce" and self.soft_targets:
                loss_g, correct, total = soft_target_contrastive_loss(
                    pred[mask],
                    target[mask],
                    temperature=self.temperature,
                    label_temperature=self.label_temperature,
                    max_samples=self.max_contrastive_samples,
                )
                parts.append(loss_g)
            else:
                loss_g, correct, total = prediction_loss(
                    pred[mask],
                    target[mask],
                    mode=self.loss_mode,
                    temperature=self.temperature,
                    max_samples=self.max_contrastive_samples,
                )
                parts.append(loss_g)
            if parts:
                group_losses[name] = torch.stack(parts).mean()
                group_correct[name] = correct
                group_total[name] = total

        loss = _mean_or_zero(group_losses, device)
        # Unused params (tokenizers of modalities absent from a batch) are handled
        # by the wrapper: DDP runs with find_unused_parameters=True and FSDP2
        # tolerates them natively. A 0*sum(p.sum()) grad anchor would turn the
        # loss into a DTensor under FSDP2 and crash backward.

        total_correct = sum(group_correct.values())
        total_total = sum(group_total.values())
        metrics: dict[str, Any] = {
            "loss": float(loss.detach().cpu()),
            "target_count": target_count,
            "top1": (total_correct / total_total) if total_total > 0 else 0.0,
            "num_groups": len(per_group),
            "k": k,
            "group_losses": {
                n: float(v.detach().cpu()) for n, v in group_losses.items()
            },
            "group_correct": dict(group_correct),
            "group_total": dict(group_total),
            "group_valid_frac": {
                name: float(g["valid"].float().mean().item())
                for name, g in per_group.items()
            },
        }
        return loss, metrics

    # ------------------------------------------------------------- eval

    @torch.no_grad()
    def update_target(self) -> None:
        """No-op: targets are a fixed frozen random projection (no EMA)."""
        return None

    def _feature_grid_shape(
        self, per_group: dict[str, dict[str, Any]]
    ) -> tuple[str, int, int]:
        """Anchor group name + (gh, gw) of the finest spatial group present."""
        best = None
        for g in self.groups:
            if g.name in per_group and g.is_spatial:
                _, gh, gw = per_group[g.name]["shape"]
                area = gh * gw
                if best is None or area > best[1]:
                    best = (g.name, area, gh, gw)
        if best is None:
            # Only non-spatial groups (e.g. ERA5): degenerate 1x1 grid.
            name = next(iter(per_group))
            _, gh, gw = per_group[name]["shape"]
            return name, gh, gw
        return best[0], best[2], best[3]

    def _encode_latents(
        self, sample: OlmoEarthSample | MaskedOlmoEarthSample
    ) -> tuple[dict[str, dict[str, Any]], torch.Tensor, int, torch.device]:
        per_group, b, device = self._tokenize(sample, generator=None)
        flat_tokens = torch.cat(
            [(g["content"] + g["meta"]).flatten(1, 3) for g in per_group.values()],
            dim=1,
        )
        flat_valid = torch.cat(
            [g["valid"].flatten(1, 3) for g in per_group.values()], dim=1
        )
        latents = self._encode_set(flat_tokens, ~flat_valid, self.latent_count)
        return per_group, latents, b, device

    def encode(self, sample: OlmoEarthSample | MaskedOlmoEarthSample) -> torch.Tensor:
        """Dense features on the anchor group's token grid: ``(b, gh, gw, dim)``.

        Read out with metadata-only grid queries mirroring a TRAINED query
        configuration exactly: the anchor group's identity + extent + gps (real
        or trained null), null absolute time, and rel_time(0).
        """
        per_group, latents, b, device = self._encode_latents(sample)
        anchor, gh, gw = self._feature_grid_shape(per_group)
        spec = self.groups[self.group_index[anchor]]
        extent_m = spec.extent_m

        east = (torch.arange(gw, device=device, dtype=torch.float32) + 0.5) * extent_m
        north = (torch.arange(gh, device=device, dtype=torch.float32) + 0.5) * extent_m
        local = torch.cat(
            [
                fourier_features(east, self.n_local_freqs, LOCAL_WAVELENGTHS_M)[
                    None, :, :
                ].expand(gh, gw, -1),
                fourier_features(north, self.n_local_freqs, LOCAL_WAVELENGTHS_M)[
                    :, None, :
                ].expand(gh, gw, -1),
            ],
            dim=-1,
        )
        local_enc = self.local_pos_proj(local.to(self.local_pos_proj.weight.dtype))
        queries = local_enc[None] + torch.zeros(
            b, 1, 1, self.dim, device=device, dtype=local_enc.dtype
        )
        coords = self._gps_coords(getattr(sample, "latlon", None), b, device)
        queries = queries + self.gps_encoding(coords)[:, None, None]
        queries = queries + self.group_tokens.weight[self.group_index[anchor]]
        queries = queries + self.time_encoding(torch.zeros(1, 3, device=device)).view(
            1, 1, 1, self.dim
        )
        queries = queries + self.rel_time_proj(
            fourier_features(
                torch.zeros(1, device=device),
                self.n_rel_time_freqs,
                REL_TIME_WAVELENGTHS_D,
            ).to(self.rel_time_proj.weight.dtype)
        ).view(1, 1, 1, self.dim)
        queries = queries + self.extent_encoding(
            torch.log(torch.tensor([[extent_m]], device=device))
        ).view(1, 1, 1, self.dim)
        features = self._decode(queries.reshape(b, gh * gw, self.dim), latents)
        return features.reshape(b, gh, gw, self.dim).float()

    def encode_global(
        self, sample: OlmoEarthSample | MaskedOlmoEarthSample
    ) -> torch.Tensor:
        """Global pooled feature ``(b, dim)``: mean over the final latents.

        A fully-trained global-feature baseline (no decoder).
        """
        _, latents, _, _ = self._encode_latents(sample)
        return latents.float().mean(dim=1)

    # ------------------------------------------------------------- distributed

    def apply_fsdp(
        self,
        dp_mesh: DeviceMesh | None = None,
        param_dtype: torch.dtype | None = None,
        reduce_dtype: torch.dtype = torch.float32,
        prefetch_factor: int = 0,
    ) -> None:
        """Apply FSDP to the model."""
        from torch.distributed.fsdp import (
            MixedPrecisionPolicy,
            fully_shard,
            register_fsdp_forward_method,
        )

        # cast_forward_inputs=False: the model casts data at each parameterized
        # boundary itself; a blanket input cast would crush timestamps/coords
        # to bf16 (e.g. year 2021 is not representable) before the fp32
        # encoding math runs.
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            cast_forward_inputs=False,
        )
        fsdp_config = dict(mesh=dp_mesh, mp_policy=mp_policy)
        for blk in (
            list(self.self_blocks)
            + list(self.level2_blocks)
            + list(self.decoder_blocks)
        ):
            fully_shard(blk, **fsdp_config)
        fully_shard(self.read_block, **fsdp_config)
        fully_shard(self.down_block, **fsdp_config)
        fully_shard(self, **fsdp_config)
        # Eval enters through these methods rather than forward(); register them
        # so FSDP unshards/reshards around them instead of crashing on DTensors.
        register_fsdp_forward_method(self, "encode")
        register_fsdp_forward_method(self, "encode_global")

    def apply_compile(self) -> None:
        """Apply torch.compile to the transformer blocks."""
        for blk in (
            list(self.self_blocks)
            + list(self.level2_blocks)
            + list(self.decoder_blocks)
        ):
            blk.compile()
        self.read_block.compile()
        self.down_block.compile()


def _ensure_nonempty(
    target_masks: dict[str, torch.Tensor],
    visible: dict[str, torch.Tensor],
    per_group: dict[str, dict[str, Any]],
) -> None:
    """Guarantee >=1 visible token per sample, and >=1 target when possible.

    Searches ALL groups (the first group may be fully invalid), and recomputes
    visibility AFTER the target-fallback: with exactly one valid token the
    target-fallback would otherwise consume the only visible token, recreating
    the fully-key-masked state this guard exists to prevent. Predicates are
    computed batched to avoid per-sample device syncs.
    """
    names = list(per_group)
    total_targets = sum(target_masks[n].flatten(1).sum(1) for n in names)
    total_visible = sum(visible[n].flatten(1).sum(1) for n in names)
    need_target = (total_targets == 0).nonzero().flatten().tolist()
    for i in need_target:
        for name in names:
            nz = per_group[name]["valid"][i].nonzero()
            if len(nz) > 0:
                t, y, x = nz[0]
                target_masks[name][i, t, y, x] = True
                if visible[name][i, t, y, x]:
                    visible[name][i, t, y, x] = False
                    total_visible[i] -= 1
                break
    need_visible = (total_visible == 0).nonzero().flatten().tolist()
    for i in need_visible:
        for name in names:
            nz = target_masks[name][i].nonzero()
            if len(nz) > 0:
                t, y, x = nz[0]
                target_masks[name][i, t, y, x] = False
                visible[name][i, t, y, x] = True
                break


# ===================================================================== losses


def _mean_or_zero(
    losses: dict[str, torch.Tensor], device: torch.device
) -> torch.Tensor:
    if losses:
        return torch.stack(list(losses.values())).mean()
    return torch.zeros((), device=device, requires_grad=True)


def prediction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    mode: str,
    temperature: float,
    max_samples: int,
) -> tuple[torch.Tensor, int, int]:
    """Loss between predicted and (detached) target tokens + top-1 counts."""
    # Compute losses in fp32 for numerical stability under bf16 training.
    pred = pred.float()
    target = target.detach().float()
    if mode == "infonce":
        return contrastive_loss(pred, target, temperature, max_samples)
    if pred.shape[0] > max_samples:
        idx = torch.randperm(pred.shape[0], device=pred.device)[:max_samples]
        pred = pred[idx]
        target = target[idx]
    if mode == "cosine":
        return (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean(), 0, 0
    if mode == "smooth_l1":
        return (
            F.smooth_l1_loss(F.normalize(pred, dim=-1), F.normalize(target, dim=-1)),
            0,
            0,
        )
    raise ValueError(f"unknown loss mode {mode!r}")


def contrastive_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    temperature: float,
    max_samples: int,
) -> tuple[torch.Tensor, int, int]:
    """Hard InfoNCE with exact-duplicate dedup."""
    pred = pred.float()
    target = target.float()
    if pred.shape[0] > max_samples:
        idx = torch.randperm(pred.shape[0], device=pred.device)[:max_samples]
        pred = pred[idx]
        target = target[idx]
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    keys, labels = torch.unique(target, dim=0, return_inverse=True)
    logits = pred @ keys.T / temperature
    correct = int((logits.argmax(dim=1) == labels).sum().item())
    return F.cross_entropy(logits, labels), correct, int(labels.shape[0])


def soft_target_contrastive_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    temperature: float,
    label_temperature: float,
    max_samples: int,
) -> tuple[torch.Tensor, int, int]:
    """InfoNCE with target-similarity soft labels + exact-duplicate dedup.

    Each anchor's label distribution is ``softmax(sim(target_i, target_j) /
    label_temperature)`` so near-duplicate targets (uniform fields, static
    scenes) share label mass instead of acting as false negatives, while
    distinct tokens remain hard negatives. ``correct`` counts hard top-1
    agreement with the anchor's own target.
    """
    pred = pred.float()
    target = target.detach().float()
    if pred.shape[0] > max_samples:
        idx = torch.randperm(pred.shape[0], device=pred.device)[:max_samples]
        pred = pred[idx]
        target = target[idx]
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    keys, own = torch.unique(target, dim=0, return_inverse=True)
    logits = pred @ keys.T / temperature
    labels = (target @ keys.T / label_temperature).softmax(dim=-1)
    loss = -(labels * logits.log_softmax(dim=-1)).sum(dim=-1).mean()
    correct = int((logits.argmax(dim=1) == own).sum().item())
    return loss, correct, int(pred.shape[0])


def temporal_contrastive_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    masked: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, int, int] | None:
    """InfoNCE where negatives are the OTHER masked timesteps at each location.

    ``pred``/``target``: ``(b, T, h, w, d)``; ``masked``: ``(b, T, h, w)`` bool.
    Returns ``(loss, correct, total)`` or None if no location has >=2 masked
    timesteps (no negatives).
    """
    b, t, h, w, d = pred.shape
    pred = pred.float()
    P = F.normalize(pred.permute(0, 2, 3, 1, 4).reshape(b * h * w, t, d), dim=-1)
    Tt = F.normalize(
        target.detach().float().permute(0, 2, 3, 1, 4).reshape(b * h * w, t, d),
        dim=-1,
    )
    vm = masked.permute(0, 2, 3, 1).reshape(b * h * w, t)  # (L, T) bool
    logits = torch.bmm(P, Tt.transpose(1, 2)) / temperature  # (L, T, T)
    logits = logits.masked_fill(~vm[:, None, :], float("-inf"))
    valid_query = vm & (vm.sum(dim=1, keepdim=True) >= 2)
    if not valid_query.any():
        return None
    labels = torch.arange(t, device=pred.device).expand(b * h * w, t)
    sel_logits = logits[valid_query]
    sel_labels = labels[valid_query]
    correct = int((sel_logits.argmax(dim=1) == sel_labels).sum().item())
    return F.cross_entropy(sel_logits, sel_labels), correct, int(sel_labels.shape[0])


# ===================================================================== config


@dataclass
class SetLatentPerceiverConfig(Config):
    """Configuration for :class:`SetLatentPerceiver`.

    All fields have defaults so old checkpoints deserialize. ``build`` mirrors
    ``EncoderConfig.build`` (``as_dict(exclude_none=True, recurse=False)`` auto-
    forwards fields whose names match constructor kwargs).
    """

    supported_modality_names: list[str] = field(
        default_factory=lambda: list(DEFAULT_MODALITY_NAMES)
    )
    token_extent_m: float = 80.0
    dim: int = 768
    heads: int = 12
    mlp_ratio: float = 4.0
    latents: int = 1024
    nested_latents: tuple[int, ...] = (128, 256, 512, 1024)
    self_depth_per_read: int = 4
    num_input_reads: int = 2
    downsample_factor: int = 4
    level2_depth: int = 2
    decoder_depth: int = 2
    target_dim: int = 256
    mask_family_probs: tuple[float, float, float] = (0.3, 0.4, 0.3)
    token_mask_prob: float = 0.3
    modality_mask_prob: float = 0.35
    timestep_mask_prob: float = 0.15
    spatial_block_frac: tuple[float, float] = (0.25, 0.5)
    n_local_freqs: int = 16
    n_rel_time_freqs: int = 8
    temperature: float = 0.1
    max_contrastive_samples: int = 4096
    loss_mode: str = "infonce"
    contrast_scope: str = "global"
    soft_targets: bool = True
    label_temperature: float = 0.05
    valid_token_threshold: float = 0.5
    cond_dropout: float = 0.5
    trained_years: tuple[float, float] | None = None

    def validate(self) -> None:
        """Validate the configuration."""
        if len(self.supported_modality_names) == 0:
            raise ValueError("At least one modality must be supported")
        for modality in self.supported_modality_names:
            Modality.get(modality)  # raises if unknown
        if self.contrast_scope not in ("global", "temporal"):
            raise ValueError("contrast_scope must be 'global' or 'temporal'")
        if self.loss_mode not in ("infonce", "cosine", "smooth_l1"):
            raise ValueError("loss_mode must be 'infonce', 'cosine', or 'smooth_l1'")
        if self.dim % self.heads != 0:
            raise ValueError("dim must be divisible by heads")

    def build(self) -> SetLatentPerceiver:
        """Build the Set-Latent Perceiver."""
        self.validate()
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        return SetLatentPerceiver(**kwargs)

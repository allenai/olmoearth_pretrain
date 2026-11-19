"""Static-shape encoder wrapper for TensorRT export.

OlmoEarth's FlexiViT encoder has dynamic shapes (boolean indexing, data-dependent
slicing, ndim branching) that prevent torch.export() and TensorRT compilation.

This module provides a StaticOlmoEarthEncoder that replays the encoder's forward
pass with all shapes baked in at construction time. It references the same trained
weights — no copying, no retraining.

Usage:
    from olmoearth_pretrain.export import StaticOlmoEarthEncoder, verify_export

    # Build static wrapper (reuses trained weights)
    static_enc = StaticOlmoEarthEncoder(encoder, patch_size=2, spatial_size=64)

    # Verify output matches original
    cosine_sim = verify_export(encoder, static_enc)
    assert cosine_sim > 0.9999

    # Export to TensorRT
    exported = torch.export.export(static_enc, (dummy_x, dummy_ts))
"""
from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from olmoearth_pretrain.data.constants import BASE_GSD, Modality
from olmoearth_pretrain.nn.encodings import (
    get_2d_sincos_pos_encoding_with_resolution,
)

logger = logging.getLogger(__name__)

# Sentinel-2 L2A band structure
S2_MODALITY = Modality.SENTINEL2_L2A
S2_BANDSET_CHANNELS = [len(bs.bands) for bs in S2_MODALITY.band_sets]  # [4, 6, 2]
S2_NUM_BANDSETS = S2_MODALITY.num_band_sets  # 3


def _get_bandset_channel_indices() -> list[list[int]]:
    """Get channel indices for each band set in the 12-band input.

    OlmoEarth S2 L2A band order: B02,B03,B04,B08, B05,B06,B07,B8A,B11,B12, B01,B09
    Band set 0 (10m): B02,B03,B04,B08 → indices [0,1,2,3]
    Band set 1 (20m): B05,B06,B07,B8A,B11,B12 → indices [4,5,6,7,8,9]
    Band set 2 (60m): B01,B09 → indices [10,11]
    """
    idx = 0
    result = []
    for n_chans in S2_BANDSET_CHANNELS:
        result.append(list(range(idx, idx + n_chans)))
        idx += n_chans
    return result


class StaticOlmoEarthEncoder(nn.Module):
    """Static-shape wrapper around OlmoEarth Encoder for TensorRT export.

    References the same trained weights as the original encoder but performs
    the forward pass with all shapes determined at construction time.
    Only supports Sentinel-2 L2A single-modality inference.

    Args:
        encoder: Trained OlmoEarth Encoder instance.
        patch_size: Patch size for inference (compile-time constant).
        spatial_size: Input spatial dimension H=W (must be square).
        num_timesteps: Number of timesteps (default 1).
        input_res: Ground sample distance in meters (default BASE_GSD=10).
    """

    def __init__(
        self,
        encoder: nn.Module,
        patch_size: int = 2,
        spatial_size: int = 64,
        num_timesteps: int = 1,
        input_res: int = BASE_GSD,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.spatial_size = spatial_size
        self.num_timesteps = num_timesteps

        # Extract embedding size from encoder
        self.embedding_size = encoder.embedding_size
        self.n = self.embedding_size // 4  # encoding dim per type

        # Pre-compute patch embedding shapes
        modality_name = "sentinel2_l2a"
        patch_embed_dict = encoder.patch_embeddings.per_modality_embeddings[modality_name]
        tile_factor = S2_MODALITY.image_tile_size_factor  # 1 for S2

        # The stored patch size in the FlexiPatchEmbed modules
        first_key = list(patch_embed_dict.keys())[0]
        first_embed = patch_embed_dict[first_key]
        self.stored_patch_size = first_embed.patch_size  # e.g. (16, 16)
        self.effective_patch_size = (patch_size * tile_factor, patch_size * tile_factor)
        self.use_linear = first_embed.use_linear_patch_embed

        # Compute output spatial dims after patching
        p_h, p_w = self.stored_patch_size
        if self.effective_patch_size != self.stored_patch_size:
            # Need interpolation: compute intermediate size
            scale_h = spatial_size * p_h // self.effective_patch_size[0]
            scale_w = spatial_size * p_w // self.effective_patch_size[1]
            self.interp_size = (scale_h, scale_w)
        else:
            self.interp_size = (spatial_size, spatial_size)

        self.h_patches = self.interp_size[0] // p_h
        self.w_patches = self.interp_size[1] // p_w
        self.seq_len = self.h_patches * self.w_patches * num_timesteps * S2_NUM_BANDSETS

        logger.info(
            f"Static encoder: patch_size={patch_size}, spatial={spatial_size}, "
            f"stored_patch={self.stored_patch_size}, interp={self.interp_size}, "
            f"h_patches={self.h_patches}, w_patches={self.w_patches}, "
            f"seq_len={self.seq_len}, embed_dim={self.embedding_size}"
        )

        # Reference patch embedding modules (per band set)
        bandset_indices = _get_bandset_channel_indices()
        self.bandset_channel_indices = bandset_indices

        # Register channel index buffers for torch.export compatibility
        for i, indices in enumerate(bandset_indices):
            self.register_buffer(f"band_idx_{i}", torch.tensor(indices, dtype=torch.long))

        # Store references to patch projection and norm modules
        self.patch_projs = nn.ModuleList()
        self.patch_norms = nn.ModuleList()
        for i in range(S2_NUM_BANDSETS):
            key = f"{modality_name}__{i}"
            embed_module = patch_embed_dict[key]
            self.patch_projs.append(embed_module.proj)
            self.patch_norms.append(embed_module.norm)

        # Reference encoding parameters
        encodings = encoder.composite_encodings
        self.channel_embed = encodings.per_modality_channel_embeddings[modality_name]
        self.pos_embed = encodings.pos_embed  # [max_seq_len, n]
        self.month_embed = encodings.month_embed  # nn.Embedding(13, n)

        # Pre-compute spatial encoding as buffer (fixed for given patch_size + spatial_size)
        gsd_ratio = input_res * patch_size / BASE_GSD
        spatial_enc = get_2d_sincos_pos_encoding_with_resolution(
            grid_size=(self.h_patches, self.w_patches),
            res=torch.ones(1) * gsd_ratio,
            encoding_dim=self.n,
            device=torch.device("cpu"),
        )  # [1, h*w, n]
        spatial_enc = spatial_enc.reshape(1, self.h_patches, self.w_patches, self.n)
        self.register_buffer("spatial_encoding", spatial_enc)

        # Reference register tokens
        self.num_register_tokens = 0
        if hasattr(encoder, "register_tokens") and encoder.register_tokens is not None:
            self.register_tokens_param = encoder.register_tokens
            self.num_register_tokens = encoder.register_tokens.shape[0]
        else:
            self.register_tokens_param = None

        # Reference transformer blocks
        self.blocks = encoder.blocks

        # Reference final norm
        self.norm = encoder.norm

        # Reference embedding projector (optional)
        self.embedding_projector = getattr(encoder, "embedding_projector", None)

        # Reference project_and_aggregate for final projection
        self.aggregate_then_project = encoder.project_and_aggregate.aggregate_then_project
        self.final_projection = encoder.project_and_aggregate.projection

    def _patch_embed_bandset(self, x_bands: torch.Tensor, bandset_idx: int) -> torch.Tensor:
        """Apply patch embedding for a single band set.

        Args:
            x_bands: [B*T, C_bandset, H, W] input for this band set.
            bandset_idx: Index of the band set (0, 1, or 2).

        Returns:
            [B*T, h_patches, w_patches, D] patch tokens.
        """
        proj = self.patch_projs[bandset_idx]
        norm = self.patch_norms[bandset_idx]
        p_h, p_w = self.stored_patch_size

        # Interpolate if needed (static sizes)
        if self.effective_patch_size != self.stored_patch_size:
            x_bands = F.interpolate(
                x_bands, size=self.interp_size, mode="bicubic", antialias=True,
            )

        if self.use_linear:
            # Reshape to patches: [BT, C, H, W] -> [BT, h*w, p*p*C]
            bt, c, h, w = x_bands.shape
            hp, wp = h // p_h, w // p_w
            # [BT, C, hp, p_h, wp, p_w] -> [BT, hp, wp, p_h, p_w, C] -> [BT, hp*wp, p_h*p_w*C]
            x_patches = x_bands.reshape(bt, c, hp, p_h, wp, p_w)
            x_patches = x_patches.permute(0, 2, 4, 3, 5, 1).reshape(bt, hp * wp, p_h * p_w * c)
            tokens = proj(x_patches)  # [BT, hp*wp, D]
            # Reshape back to spatial: [BT, hp, wp, D]
            tokens = tokens.reshape(bt, hp, wp, self.embedding_size)
        else:
            # Conv2d path
            tokens = proj(x_bands)  # [BT, D, hp, wp]
            tokens = tokens.permute(0, 2, 3, 1)  # [BT, hp, wp, D]

        tokens = norm(tokens)
        return tokens

    def forward(self, x: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        """Static forward pass.

        Args:
            x: [B, H, W, T, 12] float tensor — Sentinel-2 L2A data.
            timestamps: [B, T, 3] long tensor — (day, month, year).

        Returns:
            [B, D] embedding tensor.
        """
        B = x.shape[0]
        T = self.num_timesteps
        D = self.embedding_size
        n = self.n
        hp, wp = self.h_patches, self.w_patches
        n_bs = S2_NUM_BANDSETS

        # Phase 1: Patch embedding per band set
        all_tokens = []
        for i in range(n_bs):
            # Select bands for this band set
            band_idx = getattr(self, f"band_idx_{i}")
            x_bs = x[:, :, :, :, band_idx]  # [B, H, W, T, C_i]

            # Permute to [B, T, C_i, H, W] then merge B*T
            x_bs = x_bs.permute(0, 3, 4, 1, 2)  # [B, T, C_i, H, W]
            x_bs = x_bs.reshape(B * T, x_bs.shape[2], self.spatial_size, self.spatial_size)

            # Patch embed: [BT, hp, wp, D]
            tokens_i = self._patch_embed_bandset(x_bs, i)

            # Reshape to [B, hp, wp, T, D]
            tokens_i = tokens_i.reshape(B, T, hp, wp, D).permute(0, 2, 3, 1, 4)

            all_tokens.append(tokens_i)

        # Stack band sets: [B, hp, wp, T, n_bs, D]
        tokens = torch.stack(all_tokens, dim=4)

        # Phase 2: Apply encodings (all with static shapes, no einops)
        encoding = torch.zeros_like(tokens)

        # Channel embeddings: [n_bs, n] -> expand to [B, hp, wp, T, n_bs, n]
        ch_embed = self.channel_embed  # [3, n]
        ch_embed = ch_embed.reshape(1, 1, 1, 1, n_bs, n).expand(B, hp, wp, T, n_bs, n)
        encoding[..., :n] = encoding[..., :n] + ch_embed

        # Time position encoding: [T, n] -> expand to [B, hp, wp, T, n_bs, n]
        time_embed = self.pos_embed[:T]  # [T, n]
        time_embed = time_embed.reshape(1, 1, 1, T, 1, n).expand(B, hp, wp, T, n_bs, n)
        encoding[..., n:2*n] = encoding[..., n:2*n] + time_embed

        # Month encoding: timestamps[:, :, 1] -> [B, T] -> embed -> [B, T, n]
        months = timestamps[:, :, 1]  # [B, T]
        month_embed = self.month_embed(months)  # [B, T, n]
        month_embed = month_embed.reshape(B, 1, 1, T, 1, n).expand(B, hp, wp, T, n_bs, n)
        encoding[..., 2*n:3*n] = encoding[..., 2*n:3*n] + month_embed

        # Spatial encoding: pre-computed [1, hp, wp, n] -> expand
        spatial_embed = self.spatial_encoding  # [1, hp, wp, n]
        spatial_embed = spatial_embed.reshape(1, hp, wp, 1, 1, n).expand(B, hp, wp, T, n_bs, n)
        encoding[..., 3*n:4*n] = encoding[..., 3*n:4*n] + spatial_embed

        tokens = tokens + encoding

        # Phase 3: Flatten to sequence [B, seq_len, D]
        tokens = tokens.reshape(B, self.seq_len, D)

        # Phase 4: Skip masking — all tokens visible, no boolean indexing

        # Phase 5: Register tokens
        if self.register_tokens_param is not None and self.num_register_tokens > 0:
            reg = self.register_tokens_param.unsqueeze(0).expand(B, -1, -1)
            tokens = torch.cat([reg, tokens], dim=1)

        # Phase 6: Transformer blocks
        for blk in self.blocks:
            tokens = blk(x=tokens)

        # Phase 7: Pop register tokens
        if self.num_register_tokens > 0:
            tokens = tokens[:, self.num_register_tokens:, :]

        # Phase 8: LayerNorm
        tokens = self.norm(tokens)

        # Phase 9: Embedding projector (optional)
        if self.embedding_projector is not None:
            tokens = self.embedding_projector.projection(tokens)

        # Phase 10: Mean pool + project
        pooled = tokens.mean(dim=1)  # [B, D]
        output = self.final_projection(pooled)  # [B, out_D]

        return output


def verify_export(
    encoder: nn.Module,
    static_encoder: StaticOlmoEarthEncoder,
    num_samples: int = 10,
    device: str = "cuda",
) -> float:
    """Verify static encoder output matches original encoder.

    Args:
        encoder: Original OlmoEarth encoder.
        static_encoder: StaticOlmoEarthEncoder wrapper.
        num_samples: Number of random samples to compare.
        device: Device to run on.

    Returns:
        Mean cosine similarity (expect > 0.9999 for FP32).
    """
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue

    encoder.eval()
    static_encoder.eval()
    patch_size = static_encoder.patch_size
    spatial = static_encoder.spatial_size
    T = static_encoder.num_timesteps

    sims = []
    for i in range(num_samples):
        torch.manual_seed(i + 42)
        # Create input tensor
        x = torch.randn(1, spatial, spatial, T, S2_MODALITY.num_bands, device=device)
        ts = torch.tensor([[1, 6, 2020]], dtype=torch.long, device=device).unsqueeze(0).expand(1, T, 3)

        # Static encoder forward
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                static_out = static_encoder(x, ts)

        # Original encoder forward (needs MaskedOlmoEarthSample)
        mask = torch.full(
            (1, spatial, spatial, T, S2_MODALITY.num_band_sets),
            MaskValue.ONLINE_ENCODER.value,
            dtype=torch.long,
            device=device,
        )
        sample = MaskedOlmoEarthSample(
            sentinel2_l2a=x, sentinel2_l2a_mask=mask, timestamps=ts,
        )
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                orig_out = encoder(sample, patch_size=patch_size)
        orig_emb = orig_out["project_aggregated"]

        cos_sim = F.cosine_similarity(
            static_out.float().flatten(), orig_emb.float().flatten(), dim=0,
        ).item()
        sims.append(cos_sim)

    mean_sim = sum(sims) / len(sims)
    logger.info(
        f"Verify export: mean cosine sim = {mean_sim:.6f} "
        f"(min={min(sims):.6f}, max={max(sims):.6f})"
    )
    return mean_sim


def export_to_tensorrt(
    encoder: nn.Module,
    precision: str = "fp32",
    patch_size: int = 2,
    spatial_size: int = 64,
    num_timesteps: int = 1,
    batch_size: int = 1,
    device: str = "cuda",
):
    """End-to-end export: quantize → wrap → torch.export → torch_tensorrt.compile.

    Args:
        encoder: Trained OlmoEarth encoder.
        precision: "fp32", "fp8", or "fp4".
        patch_size: Patch size for inference.
        spatial_size: Input spatial dimension.
        num_timesteps: Number of timesteps.
        batch_size: Batch size for TRT optimization.
        device: Target device.

    Returns:
        Compiled TensorRT model.
    """
    import torch_tensorrt

    encoder = encoder.to(device).eval()

    # Step 1: Quantize if needed
    if precision in ("fp8", "fp4"):
        from olmoearth_pretrain.quantization import quantize_model
        from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue

        def calib_fn(m):
            m.eval()
            s2_spec = Modality.SENTINEL2_L2A
            for _ in range(16):
                s2 = torch.randn(4, spatial_size, spatial_size, num_timesteps,
                                 s2_spec.num_bands, device=device)
                mask = torch.full(
                    (4, spatial_size, spatial_size, num_timesteps, s2_spec.num_band_sets),
                    MaskValue.ONLINE_ENCODER.value, dtype=torch.long, device=device,
                )
                ts = torch.zeros(4, num_timesteps, 3, dtype=torch.long, device=device)
                sample = MaskedOlmoEarthSample(
                    sentinel2_l2a=s2, sentinel2_l2a_mask=mask, timestamps=ts,
                )
                with torch.no_grad():
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        m(sample, patch_size=patch_size)

        encoder = quantize_model(encoder, calib_fn, precision=precision)

    # Step 2: Build static wrapper
    static_enc = StaticOlmoEarthEncoder(
        encoder, patch_size=patch_size, spatial_size=spatial_size,
        num_timesteps=num_timesteps,
    ).to(device).eval()

    # Step 3: Verify
    sim = verify_export(encoder, static_enc, device=device)
    if sim < 0.99:
        raise RuntimeError(f"Static encoder output diverges: cosine sim = {sim:.4f}")

    # Step 4: torch.export
    dummy_x = torch.randn(batch_size, spatial_size, spatial_size, num_timesteps,
                          S2_MODALITY.num_bands, device=device)
    dummy_ts = torch.zeros(batch_size, num_timesteps, 3, dtype=torch.long, device=device)

    exported = torch.export.export(static_enc, (dummy_x, dummy_ts))

    # Step 5: TensorRT compile
    enabled_precisions = {torch.float32}
    if precision == "fp8":
        enabled_precisions.add(torch.float8_e4m3fn)
    elif precision == "fp4":
        enabled_precisions.add(torch.float8_e4m3fn)  # TRT uses FP8 as closest

    compiled = torch_tensorrt.dynamo.compile(
        exported,
        inputs=[
            torch_tensorrt.Input(shape=(batch_size, spatial_size, spatial_size,
                                        num_timesteps, S2_MODALITY.num_bands)),
            torch_tensorrt.Input(shape=(batch_size, num_timesteps, 3), dtype=torch.long),
        ],
        enabled_precisions=enabled_precisions,
    )

    logger.info("TensorRT compilation successful")
    return compiled

"""Spectral mixer module for cross-band interaction before patch embedding."""

import math

import torch
import torch.nn as nn

_DEFAULT_CHUNK_SIZE = 65536


class SpectralMixer(nn.Module):
    """Lightweight cross-band MLP mixer applied pixel-wise before patch embedding.

    Learns non-linear spectral combinations (e.g., NDVI-like ratios) across all
    bands before spatial aggregation by the patch embedding Conv2d. This restores
    cross-spectral learning that is otherwise lost when using a single flat
    bandset, since the Conv2d can only learn linear band combinations.

    Applied after band dropout (if any) so the mixer also learns to be robust
    to partial band observations.

    Initialized as identity (zero-init on the output projection) so training
    starts from the same point as a model without the mixer.

    Args:
        num_bands: Number of spectral bands in this bandset.
        expansion: Hidden dim multiplier for the inner MLP. Default: 4.
    """

    def __init__(self, num_bands: int, expansion: int = 4) -> None:
        """Initialize SpectralMixer."""
        super().__init__()
        hidden = num_bands * expansion
        self.norm = nn.LayerNorm(num_bands)
        self.fc1 = nn.Linear(num_bands, hidden)
        self.fc2 = nn.Linear(hidden, num_bands)
        self.act = nn.GELU()
        # Zero-init so the mixer starts as a pure residual (identity transform).
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply cross-band mixing.

        Args:
            x: Any-shape tensor with bands in the last dimension [..., num_bands].

        Returns:
            Spectrally mixed tensor of the same shape.
        """
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


class SpectralAttention(nn.Module):
    """Content-dependent cross-band self-attention applied pixel-wise before patch embedding.

    Unlike SpectralMixer (fixed MLP), the mixing weights here depend on the actual
    band values — a vegetation pixel gets different spectral mixing than a water pixel.
    Each scalar band value is projected to a d_model-dim embedding, augmented with a
    learnable band identity embedding, then self-attention across bands produces
    content-dependent corrections.

    Initialized as identity (zero-init on the output projection) so training
    starts from the same point as a model without the attention.

    Processes pixels in chunks to bound memory — each pixel's band attention is
    independent so chunking is exact with no approximation.

    Args:
        num_bands: Number of spectral bands in this bandset.
        d_model: Embedding dimension for band tokens. Default: 64.
        num_heads: Number of attention heads. Default: 2.
        chunk_size: Max pixels to process at once. Default: 65536.
    """

    def __init__(
        self,
        num_bands: int,
        d_model: int = 64,
        num_heads: int = 2,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
    ) -> None:
        """Initialize SpectralAttention."""
        super().__init__()
        self.num_bands = num_bands
        self.d_model = d_model
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        assert d_model % num_heads == 0

        self.band_embed = nn.Linear(1, d_model)
        self.band_pos = nn.Parameter(torch.randn(num_bands, d_model) * 0.02)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, 1)

        # Zero-init so the attention starts as identity.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _compute_delta(
        self, x_chunk: torch.Tensor, mask_chunk: torch.Tensor | None
    ) -> torch.Tensor:
        """Compute spectral attention delta for a chunk of pixels.

        Args:
            x_chunk: [C, B] pixel values.
            mask_chunk: Optional [C, B] boolean mask.

        Returns:
            [C, B] delta to add to x_chunk.
        """
        tokens = self.band_embed(x_chunk.unsqueeze(-1)) + self.band_pos  # [C, B, d]
        C, B, d = tokens.shape
        head_dim = d // self.num_heads

        Q = self.W_q(tokens).view(C, B, self.num_heads, head_dim).transpose(1, 2)
        K = self.W_k(tokens).view(C, B, self.num_heads, head_dim).transpose(1, 2)
        V = self.W_v(tokens).view(C, B, self.num_heads, head_dim).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))

        if mask_chunk is not None:
            attn = attn.masked_fill(~mask_chunk[:, None, None, :], float("-inf"))

        attn = torch.softmax(attn, dim=-1).nan_to_num(0.0)
        out = (attn @ V).transpose(1, 2).reshape(C, B, d)

        delta = self.out_proj(out).squeeze(-1)  # [C, B]
        if mask_chunk is not None:
            delta = delta * mask_chunk.to(delta.dtype)
        return delta

    def _flatten_mask(
        self, band_mask: torch.Tensor | None, N: int, B: int
    ) -> torch.Tensor | None:
        """Flatten band_mask to [N, B], handling full, per-sample, and broadcast shapes."""
        if band_mask is None:
            return None
        if band_mask.numel() == N * B:
            return band_mask.reshape(N, B)
        bm = band_mask.reshape(band_mask.shape[0], band_mask.shape[-1])
        spatial = N // bm.shape[0]
        return bm.unsqueeze(1).expand(-1, spatial, -1).reshape(N, B)

    def forward(
        self, x: torch.Tensor, band_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply content-dependent cross-band mixing.

        Args:
            x: Any-shape tensor with bands in the last dimension [..., num_bands].
            band_mask: Optional boolean mask [..., num_bands] where True = band is
                present. When provided, dropped bands are excluded from attention
                keys so present bands only attend to other present bands. The delta
                for dropped bands is zeroed so the residual preserves the dropout.

        Returns:
            Spectrally mixed tensor of the same shape.
        """
        shape = x.shape
        x_flat = x.reshape(-1, self.num_bands)  # [N, B]
        N, B = x_flat.shape

        flat_mask = self._flatten_mask(band_mask, N, B)

        if N <= self.chunk_size:
            delta = self._compute_delta(x_flat, flat_mask)
        else:
            chunks = []
            for i in range(0, N, self.chunk_size):
                end = min(i + self.chunk_size, N)
                m = flat_mask[i:end] if flat_mask is not None else None
                chunks.append(self._compute_delta(x_flat[i:end], m))
            delta = torch.cat(chunks, dim=0)

        return (x_flat + delta).reshape(shape)

"""Spectral mixer module for cross-band interaction before patch embedding."""

import math

import torch
import torch.nn as nn


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

    Args:
        num_bands: Number of spectral bands in this bandset.
        d_model: Embedding dimension for band tokens. Default: 64.
        num_heads: Number of attention heads. Default: 2.
    """

    def __init__(self, num_bands: int, d_model: int = 64, num_heads: int = 2) -> None:
        """Initialize SpectralAttention."""
        super().__init__()
        self.num_bands = num_bands
        self.d_model = d_model
        self.num_heads = num_heads
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

        # Each band scalar → d_model embedding + learnable band identity
        tokens = self.band_embed(x_flat.unsqueeze(-1)) + self.band_pos  # [N, B, d]

        # Multi-head self-attention across bands
        N, B, d = tokens.shape
        head_dim = d // self.num_heads

        Q = (
            self.W_q(tokens).view(N, B, self.num_heads, head_dim).transpose(1, 2)
        )  # [N, nh, B, hd]
        K = self.W_k(tokens).view(N, B, self.num_heads, head_dim).transpose(1, 2)
        V = self.W_v(tokens).view(N, B, self.num_heads, head_dim).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))  # [N, nh, B, B]

        # band_mask may be full-shape [..., B] or per-sample [batch, B].
        # Flatten to [N, B] to match x_flat, expanding per-sample masks as needed.
        flat_mask = None
        if band_mask is not None:
            if band_mask.numel() == N * B:
                flat_mask = band_mask.reshape(N, B)
            else:
                spatial = N // band_mask.shape[0]
                flat_mask = band_mask.unsqueeze(1).expand(-1, spatial, -1).reshape(N, B)
            attn = attn.masked_fill(~flat_mask[:, None, None, :], float("-inf"))

        attn = torch.softmax(attn, dim=-1)
        # NaN guard: if a query band has all -inf keys (dropped band querying
        # when all others are also dropped), softmax produces NaN. Replace with 0.
        attn = attn.nan_to_num(0.0)
        out = (attn @ V).transpose(1, 2).reshape(N, B, d)  # [N, B, d]

        delta = self.out_proj(out).squeeze(-1)  # [N, B]

        if flat_mask is not None:
            delta = delta * flat_mask.to(delta.dtype)

        return (x_flat + delta).reshape(shape)

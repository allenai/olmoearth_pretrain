"""Spectral attention module for content-dependent cross-band interaction before patch embedding."""

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class SpectralAttention(nn.Module):
    """Content-dependent cross-band self-attention applied pixel-wise before patch embedding.

    Each scalar band value is projected to a d_model-dim embedding, augmented with a
    learnable band identity embedding, then self-attention across bands produces
    content-dependent corrections. Unlike a fixed MLP mixer, the mixing weights
    depend on actual band values — a vegetation pixel gets different spectral
    mixing than a water pixel.

    Uses chunked processing with gradient checkpointing to bound peak memory,
    since this operates on raw pixels (before patchification) which can be very
    numerous.

    Initialized as identity (zero-init on the output projection) so training
    starts from the same point as a model without the attention.

    Args:
        num_bands: Number of spectral bands in this bandset.
        d_model: Embedding dimension for band tokens. Default: 128.
        num_heads: Number of attention heads. Default: 2.
        chunk_size: Max pixels per chunk for memory-bounded processing.
            Set 0 to disable chunking. Default: 8192 (~100MB peak per chunk).
    """

    def __init__(
        self,
        num_bands: int,
        d_model: int = 128,
        num_heads: int = 2,
        chunk_size: int = 8192,
    ) -> None:
        """Initialize SpectralAttention."""
        super().__init__()
        self.num_bands = num_bands
        self.d_model = d_model
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.band_embed = nn.Linear(1, d_model)
        self.band_pos = nn.Parameter(torch.randn(num_bands, d_model) * 0.02)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, 1)

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _forward_chunk(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Process a chunk of pixels through spectral attention.

        Args:
            x_flat: Tensor of shape [N, num_bands].

        Returns:
            Tensor of shape [N, num_bands] with spectral corrections added.
        """
        tokens = self.band_embed(x_flat.unsqueeze(-1)) + self.band_pos  # [N, B, d]

        N, B, d = tokens.shape
        head_dim = d // self.num_heads

        Q = self.W_q(tokens).view(N, B, self.num_heads, head_dim).transpose(1, 2)
        K = self.W_k(tokens).view(N, B, self.num_heads, head_dim).transpose(1, 2)
        V = self.W_v(tokens).view(N, B, self.num_heads, head_dim).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(N, B, d)

        delta = self.out_proj(out).squeeze(-1)  # [N, B]
        return x_flat + delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply content-dependent cross-band mixing.

        Args:
            x: Any-shape tensor with bands in the last dimension [..., num_bands].

        Returns:
            Spectrally mixed tensor of the same shape.
        """
        shape = x.shape
        x_flat = x.reshape(-1, self.num_bands)  # [N, B]

        if self.training and self.chunk_size > 0 and x_flat.shape[0] > self.chunk_size:
            outputs = []
            for i in range(0, x_flat.shape[0], self.chunk_size):
                chunk = x_flat[i : i + self.chunk_size]
                outputs.append(
                    checkpoint(self._forward_chunk, chunk, use_reentrant=False)
                )
            return torch.cat(outputs, dim=0).reshape(shape)
        else:
            return self._forward_chunk(x_flat).reshape(shape)

"""Window retriever: an annual student embedding plus a time window -> that window's embedding.

The released store holds one student embedding per pixel and calendar year. The
retriever maps such an embedding ``z_T`` (computed from an input spanning ``T``)
and a contiguous sub-range ``t`` of ``T`` to ``ẑ_t``, trained to match ``z_t``, the
embedding fresh inference on ``t`` alone gives. It acts per location token and
returns ``z_T`` unchanged when ``t`` is all of ``T``.

The window is described by month-of-year fields, so what is learned about a
season is shared across inputs that start in different months:

- the first month-of-year and the length (in steps) of the input ``T``;
- the first and last month-of-year and the length of the window ``t``.

Two forms (``arch``), both starting at the identity:

- ``lowrank``: ``ẑ = z + U·diag(f(c))·Vᵀz + b(c)``, with ``U, V`` shared ``d×r``;
- ``film``: an MLP whose layers are scaled and shifted by the window code, with
  a zero-initialized output gate (AdaLN-Zero).
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from olmoearth_pretrain.config import Config

MONTHS = 12


@dataclass
class WindowFields:
    """Per-sample description of the input range and the requested window.

    All fields are ``[B]`` long tensors; months are zero-indexed months of year.
    """

    input_first_month: Tensor
    input_length: Tensor
    window_first_month: Tensor
    window_last_month: Tensor
    window_length: Tensor


@dataclass
class WindowRetrieverConfig(Config):
    """Configuration for :class:`WindowRetriever`.

    Args:
        dim: Width of the embedding it maps (the full student width).
        arch: ``lowrank`` or ``film``.
        rank: Rank of the shared subspace for ``lowrank``.
        code_dim: Width of the window code.
        width: Hidden width for ``film``.
        depth: Number of modulated blocks for ``film``.
        max_steps: Largest input length in steps (the embedding table size).
    """

    dim: int = 128
    arch: str = "lowrank"
    rank: int = 16
    code_dim: int = 64
    width: int = 512
    depth: int = 2
    max_steps: int = MONTHS

    def validate(self) -> None:
        """Validate the configuration."""
        if self.arch not in ("lowrank", "film"):
            raise ValueError(f"arch must be 'lowrank' or 'film', got {self.arch!r}")

    def build(self) -> "WindowRetriever":
        """Build the retriever."""
        self.validate()
        return WindowRetriever(self)


class WindowCode(nn.Module):
    """Summed embeddings of the window fields, then an MLP."""

    def __init__(self, code_dim: int, max_steps: int) -> None:
        """One embedding table per field."""
        super().__init__()
        self.input_first_month = nn.Embedding(MONTHS, code_dim)
        self.input_length = nn.Embedding(max_steps + 1, code_dim)
        self.window_first_month = nn.Embedding(MONTHS, code_dim)
        self.window_last_month = nn.Embedding(MONTHS, code_dim)
        self.window_length = nn.Embedding(max_steps + 1, code_dim)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(code_dim, code_dim),
            nn.SiLU(),
            nn.Linear(code_dim, code_dim),
        )

    def forward(self, w: WindowFields) -> Tensor:
        """``[B, code_dim]``."""
        c = (
            self.input_first_month(w.input_first_month)
            + self.input_length(w.input_length)
            + self.window_first_month(w.window_first_month)
            + self.window_last_month(w.window_last_month)
            + self.window_length(w.window_length)
        )
        return self.mlp(c)


class FiLMBlock(nn.Module):
    """Pre-LN residual MLP block, modulated by the window code (AdaLN-Zero)."""

    def __init__(self, width: int, code_dim: int) -> None:
        """Modulation starts at zero, so the block starts as the identity."""
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width)
        )
        self.mod = nn.Linear(code_dim, 3 * width)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, h: Tensor, c: Tensor) -> Tensor:
        """``h + gate · MLP((1 + scale) · LN(h) + shift)``; c broadcasts over tokens."""
        shift, scale, gate = self.mod(c).chunk(3, dim=-1)
        return h + gate * self.mlp(self.norm(h) * (1 + scale) + shift)


class WindowRetriever(nn.Module):
    """Maps ``[B, N, dim]`` embeddings and per-sample windows to window embeddings."""

    def __init__(self, config: WindowRetrieverConfig) -> None:
        """Initialize at the identity."""
        super().__init__()
        self.arch = config.arch
        self.code = WindowCode(config.code_dim, config.max_steps)
        dim = config.dim
        if config.arch == "lowrank":
            self.U = nn.Parameter(torch.randn(dim, config.rank) / math.sqrt(dim))
            self.V = nn.Parameter(torch.randn(dim, config.rank) / math.sqrt(dim))
            self.scale = nn.Linear(config.code_dim, config.rank)
            self.shift = nn.Linear(config.code_dim, dim)
            for layer in (self.scale, self.shift):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
        else:
            self.inp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, config.width))
            self.blocks = nn.ModuleList(
                FiLMBlock(config.width, config.code_dim) for _ in range(config.depth)
            )
            self.out = nn.Sequential(
                nn.LayerNorm(config.width), nn.Linear(config.width, dim)
            )
            self.alpha = nn.Linear(config.code_dim, dim)
            nn.init.zeros_(self.alpha.weight)
            nn.init.zeros_(self.alpha.bias)

    def forward(self, z: Tensor, w: WindowFields) -> Tensor:
        """``[B, N, dim]`` -> ``[B, N, dim]``; the full input range returns z exactly."""
        c = self.code(w)[:, None, :]  # [B, 1, code_dim], broadcast over tokens
        if self.arch == "lowrank":
            out = z + ((z @ self.V) * self.scale(c)) @ self.U.T + self.shift(c)
        else:
            h = self.inp(z)
            for block in self.blocks:
                h = block(h, c)
            out = z + self.alpha(c) * self.out(h)
        full = (w.window_length == w.input_length)[:, None, None]
        return torch.where(full, z, out)

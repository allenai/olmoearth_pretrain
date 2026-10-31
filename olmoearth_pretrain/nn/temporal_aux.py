"""Model side of the time-hidden temporal auxiliary loss (multi-year samples).

The train module (``olmoearth_pretrain.train.temporal_aux``) splits every multi-year
sample of a microbatch into its year chunks and encodes each chunk as its own batch
element, so the encoder never sees a cross-year time gap (3D RoPE is relative, and the
month embedding is seasonal only). This module holds the trainable parts that read
those chunk encodings:

* :func:`pool_chunk_embeddings`: one embedding per (chunk, spatial location), from the
  AttentionPooler location tokens (one location query per token) or, without a pooler,
  the mean of the visible encoder tokens over time / band sets / modalities.
* :class:`TemporalAuxHead`: a small projector MLP and, for the ``ordering`` variant, an
  antisymmetric pairwise ordering head ``s(a, b) = g([h(z_a), h(z_b)]) - g([h(z_b),
  h(z_a)])``. It also owns the change-gate EMA statistics of the ``gated_contrast``
  variant as (checkpointed) buffers.

The head lives on :class:`~olmoearth_pretrain.nn.latent_mim.LatentMIM` and is run
inside ``LatentMIM.forward`` (see its ``temporal_aux`` argument) so that the aux pass
and the head are part of the model's forward (data parallel / FSDP friendly).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple

import torch
import torch.nn as nn
from torch import Tensor

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue

EMBEDDING_SOURCES = ("location_tokens", "mean_tokens")

# Key under which LatentMIM.forward returns the aux outputs inside ``extra_metrics``.
# The train module pops it before the metrics are accumulated.
TEMPORAL_AUX_OUTPUT_KEY = "_temporal_aux_outputs"


class TemporalAuxInput(NamedTuple):
    """What the train module hands to ``LatentMIM.forward`` for the aux pass.

    Attributes:
        batch: The aux batch (one element per (view, year chunk)), or None when the
            microbatch has no eligible multi-year sample. With None only the zero
            parameter anchor is computed (every head parameter still gets a gradient).
        pair_a: ``[P]`` element indices of the earlier chunk of each ordered pair.
        pair_b: ``[P]`` element indices of the later chunk (same sample as ``pair_a``).
    """

    batch: MaskedOlmoEarthSample | None
    pair_a: Tensor | None = None
    pair_b: Tensor | None = None


def _mlp(dims: list[int]) -> nn.Sequential:
    """Linear layers with GELU in between (no activation after the last)."""
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class TemporalAuxHead(nn.Module):
    """Projector (+ optional antisymmetric ordering head) for the temporal aux loss."""

    def __init__(
        self,
        in_dim: int,
        projector_dims: list[int],
        ordering_dims: list[int] | None = None,
        embedding_source: str = "location_tokens",
    ) -> None:
        """Build the head.

        Args:
            in_dim: Width of the pooled chunk embedding (pooler_dim for location
                tokens, the encoder output width for mean-pooled tokens).
            projector_dims: Hidden and output widths of the projector MLP.
            ordering_dims: ``[h_hidden, h_out, g_hidden]`` of the ordering head, or
                None for no ordering head. ``h`` maps a projected embedding to
                ``h_out``; ``g`` maps the concatenation ``[h(z_a), h(z_b)]`` to a scalar.
            embedding_source: ``"location_tokens"`` or ``"mean_tokens"``.
        """
        super().__init__()
        if embedding_source not in EMBEDDING_SOURCES:
            raise ValueError(
                f"embedding_source must be one of {EMBEDDING_SOURCES}, "
                f"got {embedding_source}"
            )
        self.embedding_source = embedding_source
        self.input_norm = nn.LayerNorm(in_dim)
        self.projector = _mlp([in_dim, *projector_dims])
        self.out_dim = projector_dims[-1]
        self.order_h: nn.Sequential | None = None
        self.order_g: nn.Sequential | None = None
        if ordering_dims is not None:
            h_hidden, h_out, g_hidden = ordering_dims
            self.order_h = _mlp([self.out_dim, h_hidden, h_out])
            self.order_g = _mlp([2 * h_out, g_hidden, 1])
        # Change-gate EMA statistics (gated_contrast); checkpointed with the model.
        self.register_buffer("gate_log_ratio_mean", torch.zeros(()))
        self.register_buffer("gate_log_ratio_std", torch.ones(()))
        self.register_buffer("gate_stats_initialized", torch.zeros(()))

    @property
    def has_ordering_head(self) -> bool:
        """Whether the antisymmetric ordering head exists."""
        return self.order_g is not None

    def project(self, z: Tensor) -> Tensor:
        """Projector output (not normalized) for pooled chunk embeddings ``[..., in_dim]``."""
        return self.projector(self.input_norm(z))

    def order_logits(self, z_a: Tensor, z_b: Tensor) -> Tensor:
        """Antisymmetric "b is later than a" logit; ``order_logits(a, b) == -order_logits(b, a)``.

        Args:
            z_a: ``[..., out_dim]`` projected embeddings of the first chunk.
            z_b: ``[..., out_dim]`` projected embeddings of the second chunk.
        """
        assert self.order_h is not None and self.order_g is not None
        h_a = self.order_h(z_a)
        h_b = self.order_h(z_b)
        g_ab = self.order_g(torch.cat([h_a, h_b], dim=-1)).squeeze(-1)
        g_ba = self.order_g(torch.cat([h_b, h_a], dim=-1)).squeeze(-1)
        return g_ab - g_ba

    def param_anchor(self) -> Tensor:
        """``0 * sum(params)``: keeps every head parameter in the autograd graph."""
        return sum(p.sum() for p in self.parameters()) * 0.0  # type: ignore[return-value]

    def forward(
        self,
        pooled: Tensor | None,
        pair_a: Tensor | None = None,
        pair_b: Tensor | None = None,
    ) -> dict[str, Any]:
        """Project pooled chunk embeddings and score the ordered pairs.

        Args:
            pooled: ``[N, P, in_dim]`` (N chunk elements, P spatial locations) or None.
            pair_a: ``[Q]`` element indices (earlier chunk) or None.
            pair_b: ``[Q]`` element indices (later chunk) or None.

        Returns:
            ``{"proj": [N, P, out_dim] | None, "order_logits": [Q, P] | None,
            "param_anchor": scalar 0 with a graph to every head parameter}``.
        """
        out: dict[str, Any] = {
            "proj": None,
            "order_logits": None,
            "param_anchor": self.param_anchor(),
        }
        if pooled is None:
            return out
        proj = self.project(pooled)
        out["proj"] = proj
        if (
            self.has_ordering_head
            and pair_a is not None
            and pair_b is not None
            and pair_a.numel() > 0
        ):
            out["order_logits"] = self.order_logits(proj[pair_a], proj[pair_b])
        return out


def pool_chunk_embeddings(
    encoder_output: dict[str, Any],
    embedding_source: str,
) -> tuple[Tensor, Tensor]:
    """One embedding per (chunk element, spatial location) from an encoder output.

    Args:
        encoder_output: ``Encoder.forward`` output of the aux batch.
        embedding_source: ``"location_tokens"`` (AttentionPooler grid, one query per
            token) or ``"mean_tokens"`` (mean of the ONLINE_ENCODER patch tokens over
            time, band sets and spacetime modalities at each location).

    Returns:
        ``pooled [N, P, D]`` and ``valid [N, P]`` (bool; a location is invalid when no
        visible token of any modality sits there, which only happens for mean_tokens).
    """
    if embedding_source == "location_tokens":
        location_tokens = encoder_output.get("location_tokens")
        if location_tokens is None:
            raise ValueError(
                "embedding_source='location_tokens' needs an encoder AttentionPooler"
            )
        n, h, w, d = location_tokens.shape
        pooled = location_tokens.reshape(n, h * w, d)
        valid = torch.ones(n, h * w, dtype=torch.bool, device=pooled.device)
        return pooled, valid

    latent = encoder_output["tokens_and_masks"]
    total: Tensor | None = None
    count: Tensor | None = None
    for name in latent.modalities:
        if not Modality.get(name).is_spacetime_varying:
            continue
        tokens = getattr(latent, name)  # [N, h, w, T, BS, D]
        mask = getattr(latent, latent.get_masked_modality_name(name))
        visible = (mask == MaskValue.ONLINE_ENCODER.value).to(tokens.dtype)
        s = (tokens * visible.unsqueeze(-1)).sum(dim=(3, 4))  # [N, h, w, D]
        c = visible.sum(dim=(3, 4))  # [N, h, w]
        total = s if total is None else total + s
        count = c if count is None else count + c
    if total is None or count is None:
        raise ValueError("the aux batch has no spacetime modality tokens")
    n, h, w, d = total.shape
    pooled = (total / count.clamp(min=1).unsqueeze(-1)).reshape(n, h * w, d)
    valid = (count > 0).reshape(n, h * w)
    return pooled, valid


@dataclass
class TemporalAuxHeadConfig(Config):
    """Configuration of :class:`TemporalAuxHead` (set on ``LatentMIMConfig``).

    Args:
        embedding_source: ``"location_tokens"`` (AttentionPooler location tokens, one
            query per token; requires an encoder AttentionPooler) or ``"mean_tokens"``.
        projector_dims: Hidden and output widths of the projector MLP.
        ordering_dims: ``[h_hidden, h_out, g_hidden]`` of the antisymmetric ordering
            head (``ordering`` variant), or None (``gated_contrast``).
    """

    embedding_source: str = "location_tokens"
    projector_dims: list[int] = field(default_factory=lambda: [512, 128])
    ordering_dims: list[int] | None = None

    def validate(self) -> None:
        """Validate the configuration."""
        if self.embedding_source not in EMBEDDING_SOURCES:
            raise ValueError(
                f"embedding_source must be one of {EMBEDDING_SOURCES}, "
                f"got {self.embedding_source}"
            )
        if len(self.projector_dims) < 1:
            raise ValueError("projector_dims needs at least one width")
        if self.ordering_dims is not None and len(self.ordering_dims) != 3:
            raise ValueError("ordering_dims must be [h_hidden, h_out, g_hidden]")

    def build(self, in_dim: int) -> TemporalAuxHead:
        """Build the head for pooled embeddings of width ``in_dim``."""
        self.validate()
        return TemporalAuxHead(
            in_dim=in_dim,
            projector_dims=list(self.projector_dims),
            ordering_dims=(
                list(self.ordering_dims) if self.ordering_dims is not None else None
            ),
            embedding_source=self.embedding_source,
        )

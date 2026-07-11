"""Set-latent (Perceiver) drop-in encoder for the LatentMIM pipeline.

A token-in/token-out variant of the FlexiViT ``Encoder``: the quadratic
self-attention trunk is replaced by a set-latent bottleneck — a fixed pool of
learned latent queries cross-attends the visible token set (read-in), is
refined by latent self-attention, and per-token outputs are produced by
cross-attending the original tokens back over the latents (read-out).

Everything else — patch embeddings, band dropout, masking semantics, RoPE
positions, token-exit, projection heads, FSDP/compile hooks, eval pooling —
is inherited from ``Encoder`` unchanged, so this slots directly into
``LatentMIMConfig.encoder_config`` and the standard train/eval plumbing
(including the frozen random target encoder, which with an all-zero
``token_exit_cfg`` never runs the trunk at all).

Compute: read-in/read-out are O(N*K) and latent self-attention is O(K^2),
versus the trunk's O(N^2); latents are positionless (zero RoPE coordinates,
the same convention register tokens use).
"""

from typing import Any

import torch
from torch import nn

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.nn.attention import Block
from olmoearth_pretrain.nn.flexi_vit import Encoder, EncoderConfig

__all__ = ["PerceiverEncoder", "PerceiverEncoderConfig", "PerceiverTrunk"]

# olmo-core Config requires a dataclass decorator import at use site.
from dataclasses import dataclass  # noqa: E402


class PerceiverTrunk(nn.Module):
    """The set-latent replacement for the self-attention trunk.

    Presents the same call signature as a single ``Block`` so
    ``FlexiVitBase.apply_attn`` can drive it unmodified — it is installed as
    the only element of ``self.blocks``. ``num_input_reads`` repeated reads
    share one cross-attention block (weight-tied), each followed by
    ``depth / num_input_reads`` latent self-attention blocks.
    """

    def __init__(
        self,
        embedding_size: int,
        num_heads: int,
        mlp_ratio: float,
        depth: int,
        num_latents: int,
        num_input_reads: int,
        qk_norm: bool,
        drop_path: float,
        position_encoding: str,
        rope_base: float,
        rope_mixed_base: float,
        temporal_rope_dim_frac: float,
        rope_temporal_base: float | None,
    ) -> None:
        """Initialize the set-latent trunk."""
        super().__init__()
        if depth % num_input_reads != 0:
            raise ValueError(
                f"depth ({depth}) must be divisible by num_input_reads "
                f"({num_input_reads})"
            )
        self.num_input_reads = num_input_reads
        self.latents = nn.Parameter(torch.zeros(num_latents, embedding_size))
        nn.init.normal_(self.latents, std=0.02)
        common = dict(
            qkv_bias=True,
            qk_norm=qk_norm,
            norm_layer=nn.LayerNorm,
            drop_path=drop_path,
            use_flash_attn=False,
            position_encoding=position_encoding,
            rope_base=rope_base,
            rope_mixed_base=rope_mixed_base,
            temporal_rope_dim_frac=temporal_rope_dim_frac,
            rope_temporal_base=rope_temporal_base,
        )
        self.read_block = Block(
            embedding_size, num_heads, mlp_ratio, cross_attn=True, **common
        )
        self.self_blocks = nn.ModuleList(
            [
                Block(embedding_size, num_heads, mlp_ratio, cross_attn=False, **common)
                for _ in range(depth)
            ]
        )
        self.read_out_block = Block(
            embedding_size, num_heads, mlp_ratio, cross_attn=True, **common
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        attn_mask: torch.Tensor | None = None,
        rope_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compress tokens through the latent set and read back per-token.

        Args:
            x: Token embeddings ``(B, N, D)``.
            cu_seqlens: Unsupported (flash-attn packing); must be None.
            max_seqlen: Unused (flash-attn packing).
            attn_mask: Key-padding mask over tokens ``(B, N)``, True = attend.
            rope_positions: RoPE coordinates ``(B, N, 3)`` for the tokens;
                latents use zero coordinates (positionless).
        """
        if cu_seqlens is not None:
            raise NotImplementedError(
                "PerceiverTrunk does not support flash-attn packed sequences"
            )
        b = x.shape[0]
        latents = self.latents.unsqueeze(0).expand(b, -1, -1)
        latent_positions = None
        if rope_positions is not None:
            latent_positions = torch.zeros(
                b,
                self.latents.shape[0],
                rope_positions.shape[-1],
                device=x.device,
                dtype=rope_positions.dtype,
            )
        per_read = len(self.self_blocks) // self.num_input_reads
        for read in range(self.num_input_reads):
            latents = self.read_block(
                x=latents,
                y=x,
                attn_mask=attn_mask,
                rope_positions=latent_positions,
                rope_positions_y=rope_positions,
            )
            for blk in self.self_blocks[read * per_read : (read + 1) * per_read]:
                latents = blk(x=latents, rope_positions=latent_positions)
        # Per-token readout: original tokens query the latent set (residual
        # inside Block preserves token identity/position).
        return self.read_out_block(
            x=x,
            y=latents,
            rope_positions=rope_positions,
            rope_positions_y=latent_positions,
        )

    def apply_fsdp(self, **fsdp_kwargs: Any) -> None:
        """Apply FSDP per sub-block (the hook FlexiVitBase calls per 'block')."""
        self.read_block.apply_fsdp(**fsdp_kwargs)
        for blk in self.self_blocks:
            blk.apply_fsdp(**fsdp_kwargs)
        self.read_out_block.apply_fsdp(**fsdp_kwargs)

    def apply_compile(self) -> None:
        """Apply torch.compile per sub-block."""
        self.read_block.apply_compile()
        for blk in self.self_blocks:
            blk.apply_compile()
        self.read_out_block.apply_compile()


class PerceiverEncoder(Encoder):
    """FlexiViT ``Encoder`` with the trunk swapped for a set-latent bottleneck.

    All non-trunk behavior (patchification, masking, positions, projections,
    token-exit, eval pooling via ``OlmoEarthEvalWrapper``) is inherited.
    """

    def __init__(
        self,
        *,
        num_latents: int = 512,
        num_input_reads: int = 2,
        embedding_size: int = 16,
        num_heads: int = 2,
        mlp_ratio: float = 1.0,
        depth: int = 2,
        drop_path: float = 0.1,
        qk_norm: bool = False,
        use_flash_attn: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the encoder. See ``Encoder`` for inherited args."""
        if use_flash_attn:
            raise ValueError(
                "PerceiverEncoder requires the key-padding attn-mask path; "
                "use_flash_attn is unsupported"
            )
        # depth=0: skip building the (discarded) self-attention trunk.
        super().__init__(
            embedding_size=embedding_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            depth=0,
            drop_path=drop_path,
            qk_norm=qk_norm,
            use_flash_attn=False,
            **kwargs,
        )
        self.blocks = nn.ModuleList(
            [
                PerceiverTrunk(
                    embedding_size=embedding_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    depth=depth,
                    num_latents=num_latents,
                    num_input_reads=num_input_reads,
                    qk_norm=qk_norm,
                    drop_path=drop_path,
                    position_encoding=self.position_encoding,
                    rope_base=self.rope_base,
                    rope_mixed_base=self.rope_mixed_base,
                    temporal_rope_dim_frac=self.temporal_rope_dim_frac,
                    rope_temporal_base=self.rope_temporal_base,
                )
            ]
        )
        self.blocks.apply(self._init_weights)


@dataclass
class PerceiverEncoderConfig(EncoderConfig):
    """``EncoderConfig`` building a :class:`PerceiverEncoder`."""

    num_latents: int = 512
    num_input_reads: int = 2

    def build(self) -> "PerceiverEncoder":  # type: ignore[override]
        """Build the perceiver encoder."""
        self.validate()
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs.pop("supported_modality_names")
        kwargs["supported_modalities"] = self.supported_modalities
        return PerceiverEncoder(**kwargs)


# Keep Config import referenced (as_dict comes from the olmo-core Config base).
_ = Config

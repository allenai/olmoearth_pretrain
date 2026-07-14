"""Perceiver-style latent-bottleneck encoder for OlmoEarth.

Architecture (v0, designed against the v1.2 baseline):

1. Tokens are produced exactly as in the v1.2 ``Encoder`` (FlexiViT patch
   embeddings + composite channel/month encodings). Visible (ONLINE_ENCODER)
   tokens are used only as cross-attention keys/values — they never
   self-attend.
2. A coarse grid of *anchored latents* (stride ``latent_stride_hw`` in space,
   ``latent_stride_t`` in time) reads from the tokens via cross-attention.
   Each latent carries a real (t, row, col) coordinate in the same
   GSD-scaled/rope-scaled frame as the tokens, so RoPE applies to the
   cross-attention (Q rotated by anchor coords, K by token coords) and to the
   latent self-attention trunk.
3. The latent trunk is the standard ``Block`` stack (``depth`` blocks);
   cross-attention reads are interleaved at ``num_reads`` evenly spaced
   layers.
4. A dense read-out queries the latents once per (h, w, t) grid position and
   produces a fused multi-modal dense map. This is the encoder's output
   interface: it is broadcast to each modality's token layout so the rest of
   the pipeline (loss, pooling, evals) is unchanged.

Strict bottleneck: the dense read-out attends only to latents, never to raw
tokens. Static (T=1) modalities are decoded from the first temporal plane,
matching ``build_rope_positions``'s convention that a ``[B,H,W,1,bs,D]``
modality sits at the day of timestep 0.

The matching ``PerceiverPredictor`` is a per-location head: it takes the
fused dense map (already broadcast per modality by the encoder), adds the
modality/bandset channel and month encodings, and applies a small MLP stack —
no attention. All information for decoding a masked token must therefore flow
through the latent bottleneck into the dense map.
"""

import logging
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from einops import repeat

from olmoearth_pretrain.datatypes import TokensAndMasks
from olmoearth_pretrain.nn.attention import Block, Mlp
from olmoearth_pretrain.nn.encodings import (
    PositionEncoding,
    get_month_encoding_table,
    timestamps_to_days,
)
from olmoearth_pretrain.nn.flexi_vit import (
    BASE_GSD,
    CompositeEncodings,
    Encoder,
    EncoderConfig,
    PredictorBase,
    PredictorConfig,
    get_modalities_to_process,
)

logger = logging.getLogger(__name__)


class PerceiverEncoder(Encoder):
    """Latent-bottleneck encoder: tokens are K/V only, compute runs on latents.

    Subclasses ``Encoder`` so patch embedding, composite encodings, masking
    removal, the frozen-target exit-at-0 path, and the ``forward`` contract
    are all inherited unchanged; only ``apply_attn`` is replaced.
    """

    # The inherited ``self.blocks`` are the latent self-attention trunk.
    cross_attn: bool = False

    def __init__(
        self,
        *args: Any,
        latent_stride_hw: int = 2,
        latent_stride_t: int = 2,
        num_reads: int = 2,
        readout_depth: int = 2,
        readout_skip_tokens: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the PerceiverEncoder.

        Args:
            args: Positional args forwarded to ``Encoder``.
            latent_stride_hw: Spatial stride of the latent anchor grid, in
                patch-grid units. One latent per ``stride x stride`` spatial
                cell.
            latent_stride_t: Temporal stride of the latent anchor grid, in
                timestep slots.
            num_reads: Number of cross-attention reads from tokens into the
                latents, spread evenly through the latent trunk.
            readout_depth: Number of cross-attention blocks in the dense
                read-out (queries -> latents).
            readout_skip_tokens: If True, dense read-out queries attend to
                the visible tokens in addition to the latents (a skip path
                that relaxes the strict bottleneck to help dense
                prediction). Masked tokens remain excluded, so there is no
                leakage.
            kwargs: Keyword args forwarded to ``Encoder``.
        """
        super().__init__(*args, **kwargs)
        if not PositionEncoding.is_rope(self.position_encoding):
            raise ValueError(
                "PerceiverEncoder requires a RoPE position encoding: latent "
                "anchors carry coordinates only through RoPE. Got "
                f"{self.position_encoding!r}."
            )
        if self.use_flash_attn:
            raise NotImplementedError(
                "PerceiverEncoder currently supports only the SDPA path; "
                "packed varlen flash-attention is incompatible with the dense "
                "latent set."
            )
        if latent_stride_hw < 1 or latent_stride_t < 1:
            raise ValueError("latent strides must be >= 1")
        if num_reads < 1:
            raise ValueError("num_reads must be >= 1")
        if readout_depth < 1:
            raise ValueError("readout_depth must be >= 1")

        self.latent_stride_hw = latent_stride_hw
        self.latent_stride_t = latent_stride_t
        self.num_reads = num_reads
        self.readout_depth = readout_depth
        self.readout_skip_tokens = readout_skip_tokens

        depth = len(self.blocks)
        if num_reads > depth:
            raise ValueError(
                f"num_reads ({num_reads}) cannot exceed encoder depth ({depth})"
            )
        # Read i happens just before latent block read_layers[i].
        self.read_layers = [round(i * depth / num_reads) for i in range(num_reads)]

        block_kwargs = dict(
            dim=self.embedding_size,
            num_heads=self.blocks[0].attn.num_heads,
            mlp_ratio=self.blocks[0].mlp.fc1.out_features / self.embedding_size,
            qkv_bias=True,
            qk_norm=not isinstance(self.blocks[0].attn.q_norm, nn.Identity),
            norm_layer=nn.LayerNorm,
            cross_attn=True,
            drop_path=0.0,
            use_flash_attn=self.use_flash_attn,
            position_encoding=self.position_encoding,
            rope_base=self.rope_base,
            rope_mixed_base=self.rope_mixed_base,
            temporal_rope_dim_frac=self.temporal_rope_dim_frac,
            rope_temporal_base=self.rope_temporal_base,
        )
        self.read_blocks = nn.ModuleList(
            [Block(**block_kwargs) for _ in range(num_reads)]
        )
        self.readout_blocks = nn.ModuleList(
            [Block(**block_kwargs) for _ in range(readout_depth)]
        )
        # Norm applied to latents before they serve as read-out K/V (the
        # inherited ``self.norm`` is reserved for the encoder's final output,
        # i.e. the dense map, matching the baseline's output-norm placement).
        self.latent_norm = nn.LayerNorm(self.embedding_size)

        # Month embedding for latent/query content, mirroring
        # CompositeEncodings: a frozen sinusoid table whose output is added
        # into the third quarter ([2n:3n]) of the embedding.
        n_quarter = int(self.embedding_size * 0.25)
        self._month_quarter = n_quarter
        month_tab = get_month_encoding_table(n_quarter)
        self.anchor_month_embed = nn.Embedding.from_pretrained(month_tab, freeze=True)

        self.latent_token = nn.Parameter(torch.zeros(self.embedding_size))
        self.readout_query_token = nn.Parameter(torch.zeros(self.embedding_size))

        # Re-run the shared linear init over the newly added blocks, then the
        # perceiver-style trunc-normal init for the learned token seeds.
        self.read_blocks.apply(self._init_weights)
        self.readout_blocks.apply(self._init_weights)
        nn.init.trunc_normal_(self.latent_token, std=0.02, a=-2.0, b=2.0)
        nn.init.trunc_normal_(self.readout_query_token, std=0.02, a=-2.0, b=2.0)

    def _grid_dims(
        self, modalities_to_dims_dict: dict[str, tuple]
    ) -> tuple[int, int, int, int]:
        """Infer (B, H, W, T) of the sample's patch grid from modality dims."""
        batch_size: int | None = None
        height: int | None = None
        width: int | None = None
        timesteps = 1
        for modality, dims in modalities_to_dims_dict.items():
            if len(dims) == 6:
                b, h, w, t = dims[0], dims[1], dims[2], dims[3]
            elif len(dims) == 5:
                b, h, w, t = dims[0], dims[1], dims[2], 1
            else:
                raise NotImplementedError(
                    f"PerceiverEncoder supports spatial modalities only; got "
                    f"{modality} with dims {dims}"
                )
            if height is not None and (h != height or w != width):
                raise ValueError(
                    f"Inconsistent spatial grids across modalities: "
                    f"({height},{width}) vs ({h},{w}) for {modality}"
                )
            batch_size, height, width = b, h, w
            timesteps = max(timesteps, t)
        if batch_size is None or height is None or width is None:
            raise ValueError("No spatial modalities found to define the grid")
        return batch_size, height, width, timesteps

    def _build_query_grid(
        self,
        seed_token: nn.Parameter,
        row_idx: torch.Tensor,
        col_idx: torch.Tensor,
        t_slots: torch.Tensor,
        timestamps: torch.Tensor,
        gsd_ratio: float,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build query/latent content and RoPE coords for a (rows, cols, slots) grid.

        Content = seed token + month embedding of the slot's timestamp, added
        into the third embedding quarter (mirroring CompositeEncodings).
        Coordinates use the same scaling as ``build_rope_positions``:
        row/col in GSD-scaled patch units, t in scaled days (3D modes only).

        Returns:
            content: [B, n_rows * n_cols * n_slots, D], flattened as (h w t).
            coords: [B, N, 3] for 3D RoPE modes, [B, N, 2] for 2D modes.
        """
        n_rows, n_cols, n_slots = len(row_idx), len(col_idx), len(t_slots)

        months = timestamps[:, t_slots, 1].long().to(device)  # [B, n_slots]
        month_embed = self.anchor_month_embed(months)  # [B, n_slots, n_quarter]
        content = repeat(
            seed_token,
            "d -> b h w t d",
            b=batch_size,
            h=n_rows,
            w=n_cols,
            t=n_slots,
        ).clone()
        n = self._month_quarter
        content[..., 2 * n : 3 * n] += repeat(
            month_embed, "b t d -> b h w t d", h=n_rows, w=n_cols
        ).to(content.dtype)
        content = content.flatten(1, 3)  # [B, N, D]

        rows = row_idx.to(device=device, dtype=torch.float32) * gsd_ratio
        cols = col_idx.to(device=device, dtype=torch.float32) * gsd_ratio
        row_g = repeat(rows, "h -> h w t", w=n_cols, t=n_slots)
        col_g = repeat(cols, "w -> h w t", h=n_rows, t=n_slots)

        if PositionEncoding.is_3d_rope(self.position_encoding):
            days = timestamps_to_days(timestamps).to(torch.float32).to(device) * (
                self.rope_temporal_coordinate_scale
            )  # [B, T_max]
            t_vals = days[:, t_slots]  # [B, n_slots]
            coords = torch.zeros(
                (batch_size, n_rows, n_cols, n_slots, 3),
                dtype=torch.float32,
                device=device,
            )
            coords[..., 0] = repeat(t_vals, "b t -> b h w t", h=n_rows, w=n_cols)
            coords[..., 1] = repeat(row_g, "h w t -> b h w t", b=batch_size)
            coords[..., 2] = repeat(col_g, "h w t -> b h w t", b=batch_size)
        else:
            coords = torch.zeros(
                (batch_size, n_rows, n_cols, n_slots, 2),
                dtype=torch.float32,
                device=device,
            )
            coords[..., 0] = repeat(row_g, "h w t -> b h w t", b=batch_size)
            coords[..., 1] = repeat(col_g, "h w t -> b h w t", b=batch_size)
        coords = coords.flatten(1, 3)  # [B, N, coord_dim]
        return content, coords

    def apply_attn(
        self,
        x: dict[str, torch.Tensor],
        timestamps: torch.Tensor,
        patch_size: int,
        input_res: int,
        token_exit_cfg: dict[str, int] | None = None,
        fast_pass: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any] | None]:
        """Cross-attend tokens into latents, process, and read out densely."""
        if token_exit_cfg is not None and any(
            depth > 0 for depth in token_exit_cfg.values()
        ):
            raise NotImplementedError(
                "PerceiverEncoder does not support token_exit_cfg with "
                "nonzero exits; the frozen-target path uses all-zero exits "
                "which never reaches apply_attn."
            )

        tokens_only_dict, original_masks_dict, modalities_to_dims_dict = (
            self.split_tokens_masks_and_dims(x)
        )
        tokens_dict = self.composite_encodings.forward(
            tokens_only_dict, timestamps, patch_size, input_res
        )
        positions = self.build_rope_positions(
            tokens_only_dict,
            original_masks_dict,
            patch_size,
            input_res,
            timestamps=timestamps,
        )
        tokens_dict.update(original_masks_dict)
        tokens, mask = self.collapse_and_combine_hwtc(tokens_dict)

        # K/V = visible (ONLINE_ENCODER) tokens only, same removal as baseline.
        tokens, _indices, new_mask, _seq_lengths, _max_seqlen, bool_mask = (
            self._maybe_remove_masked_tokens(tokens, mask, fast_pass)
        )
        if positions is not None and bool_mask is not None:
            positions, _, _, _, _ = self.remove_masked_tokens(positions, bool_mask)
        attn_mask = self._maybe_get_attn_mask(new_mask, fast_pass)

        batch_size, height, width, timesteps = self._grid_dims(modalities_to_dims_dict)
        device = tokens.device
        gsd_ratio = (
            CompositeEncodings.calculate_gsd_ratio(input_res, patch_size)
            * self.rope_coordinate_scale
        )

        # Anchored latents on a strided grid, centered in their cells.
        s_hw, s_t = self.latent_stride_hw, self.latent_stride_t
        anchor_rows = (
            torch.arange(math.ceil(height / s_hw), dtype=torch.float32) * s_hw
            + (s_hw - 1) / 2.0
        )
        anchor_cols = (
            torch.arange(math.ceil(width / s_hw), dtype=torch.float32) * s_hw
            + (s_hw - 1) / 2.0
        )
        anchor_slots = torch.arange(0, timesteps, s_t, dtype=torch.long)
        latents, latent_pos = self._build_query_grid(
            self.latent_token,
            anchor_rows,
            anchor_cols,
            anchor_slots,
            timestamps,
            gsd_ratio,
            batch_size,
            device,
        )

        # Latent trunk with interleaved reads.
        read_i = 0
        for i_blk, blk in enumerate(self.blocks):
            while read_i < len(self.read_layers) and self.read_layers[read_i] == i_blk:
                latents = self.read_blocks[read_i](
                    x=latents,
                    y=tokens,
                    attn_mask=attn_mask,
                    rope_positions=latent_pos,
                    rope_positions_y=positions,
                )
                read_i += 1
            latents = blk(x=latents, rope_positions=latent_pos)
        latents = self.latent_norm(latents)

        # Dense read-out: one query per (h, w, t); strict bottleneck (K/V =
        # latents only).
        queries, query_pos = self._build_query_grid(
            self.readout_query_token,
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            torch.arange(timesteps, dtype=torch.long),
            timestamps,
            gsd_ratio,
            batch_size,
            device,
        )
        if self.readout_skip_tokens:
            # Skip path: dense queries also attend to visible tokens
            # (legitimate context — masked tokens were removed above), on top
            # of the latents. Relaxes the strict bottleneck for dense tasks.
            kv = torch.cat([latents, tokens.to(latents.dtype)], dim=1)
            kv_pos = torch.cat([latent_pos, positions], dim=1)
            if attn_mask is not None:
                kv_mask = torch.cat(
                    [attn_mask.new_ones(batch_size, latents.shape[1]), attn_mask],
                    dim=1,
                )
            else:
                kv_mask = None
        else:
            kv, kv_pos, kv_mask = latents, latent_pos, None
        for rblk in self.readout_blocks:
            queries = rblk(
                x=queries,
                y=kv,
                attn_mask=kv_mask,
                rope_positions=query_pos,
                rope_positions_y=kv_pos,
            )
        dense = self.norm(queries).view(
            batch_size, height, width, timesteps, self.embedding_size
        )

        # Broadcast the fused dense map to each modality's token layout.
        # Static (T=1) modalities read the first temporal plane, matching the
        # coordinate convention of build_rope_positions.
        output_dict: dict[str, torch.Tensor] = {}
        for modality, dims in modalities_to_dims_dict.items():
            if len(dims) == 6:
                t_m, bandsets = dims[3], dims[4]
                output_dict[modality] = (
                    dense[:, :, :, :t_m]
                    .unsqueeze(4)
                    .expand(-1, -1, -1, -1, bandsets, -1)
                )
            elif len(dims) == 5:
                bandsets = dims[3]
                output_dict[modality] = (
                    dense[:, :, :, 0].unsqueeze(3).expand(-1, -1, -1, bandsets, -1)
                )
            else:  # pragma: no cover - guarded in _grid_dims
                raise NotImplementedError
        output_dict.update(original_masks_dict)
        return output_dict, None

    def apply_fsdp(self, **fsdp_kwargs: Any) -> None:
        """Apply FSDP, sharding read/read-out blocks individually.

        Individual sharding matters beyond memory: FSDP2's
        ``cast_forward_inputs`` casts each block's inputs (tokens, RoPE
        coordinates) to the compute dtype at the block boundary, exactly like
        the baseline's trunk blocks. Under the root shard alone, fp32 tensors
        built inside ``apply_attn`` would hit bf16 weights and crash.
        """
        for block in self.read_blocks:
            block.apply_fsdp(**fsdp_kwargs)
        for block in self.readout_blocks:
            block.apply_fsdp(**fsdp_kwargs)
        super().apply_fsdp(**fsdp_kwargs)


class PerceiverPredictor(PredictorBase):
    """Per-location modality decoder over the fused dense map.

    The encoder hands over per-modality tensors that all view the same fused
    dense map. This predictor conditions each location's dense token on the
    modality/bandset identity (channel embedding) and month (both via the
    inherited ``composite_encodings``), then applies a small per-token MLP
    stack — no attention. Output tokens live in the target (encoder
    embedding) space, matching the frozen-projection targets.
    """

    cross_attn = True  # inherited blocks are unused; kept for base-class init

    def __init__(self, *args: Any, head_depth: int = 2, **kwargs: Any) -> None:
        """Initialize the predictor head.

        Args:
            args: Positional args forwarded to ``PredictorBase``.
            head_depth: Number of pre-norm residual MLP layers applied per
                token before the output projection.
            kwargs: Keyword args forwarded to ``PredictorBase`` (``depth``
                must be 0 — attention blocks are not used).
        """
        super().__init__(*args, **kwargs)
        if len(self.blocks) > 0:
            raise ValueError(
                "PerceiverPredictor is attention-free; configure depth=0 "
                f"(got depth={len(self.blocks)})"
            )
        if head_depth < 1:
            raise ValueError("head_depth must be >= 1")
        self.head_depth = head_depth
        dim = self.embedding_size
        mlp_ratio = 4.0
        self.head_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(head_depth)])
        self.head_mlps = nn.ModuleList(
            [Mlp(dim, int(dim * mlp_ratio)) for _ in range(head_depth)]
        )
        self.head_norms.apply(self._init_weights)
        self.head_mlps.apply(self._init_weights)

    def forward(
        self,
        x: TokensAndMasks,
        timestamps: torch.Tensor,
        patch_size: int,
        input_res: int = BASE_GSD,
    ) -> TokensAndMasks:
        """Decode per-modality predictions from the fused dense map."""
        available_modalities = x.modalities
        modalities_to_process = get_modalities_to_process(
            available_modalities, self.supported_modality_names
        )
        tokens_only_dict: dict[str, torch.Tensor] = {}
        masks_dict: dict[str, torch.Tensor] = {}
        for modality in modalities_to_process:
            x_modality = getattr(x, modality)
            x_modality = self.encoder_to_decoder_embed(self.input_norm(x_modality))
            tokens_only_dict[modality] = x_modality
            masked_name = x.get_masked_modality_name(modality)
            masks_dict[masked_name] = getattr(x, masked_name)

        # Adds the modality/bandset channel embedding and month embedding
        # (spatial/temporal quarters are inactive under RoPE modes).
        tokens_dict = self.composite_encodings.forward(
            tokens_only_dict, timestamps, patch_size, input_res
        )

        output_dict: dict[str, torch.Tensor] = {}
        for modality in modalities_to_process:
            # Composite encodings can promote to fp32 (frozen sinusoid
            # tables); cast back so the head Linears match the FSDP compute
            # dtype. There is no sharded block downstream to do it for us.
            tokens = tokens_dict[modality].to(tokens_only_dict[modality].dtype)
            for norm, mlp in zip(self.head_norms, self.head_mlps):
                tokens = tokens + mlp(norm(tokens))
            num_band_sets = self.tokenization_config.get_num_bandsets(modality)
            per_bandset_outputs = []
            for idx in range(num_band_sets):
                per_bandset = tokens[..., idx, :]
                per_bandset_outputs.append(self.to_output_embed(self.norm(per_bandset)))
            output_dict[modality] = torch.stack(per_bandset_outputs, dim=-2)
            masked_name = x.get_masked_modality_name(modality)
            output_dict[masked_name] = masks_dict[masked_name]
        return TokensAndMasks(**output_dict)


@dataclass
class PerceiverEncoderConfig(EncoderConfig):
    """Configuration for the PerceiverEncoder."""

    latent_stride_hw: int = 2
    latent_stride_t: int = 2
    num_reads: int = 2
    readout_depth: int = 2
    readout_skip_tokens: bool = False

    def validate(self) -> None:
        """Validate the configuration."""
        super().validate()
        if not PositionEncoding.is_rope(self.position_encoding):
            raise ValueError(
                "PerceiverEncoderConfig requires a RoPE position_encoding, "
                f"got {self.position_encoding!r}"
            )

    def build(self) -> "PerceiverEncoder":
        """Build the PerceiverEncoder."""
        self.validate()
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs.pop("supported_modality_names")
        kwargs["supported_modalities"] = self.supported_modalities
        logger.info(f"PerceiverEncoder kwargs: {kwargs}")
        return PerceiverEncoder(**kwargs)


@dataclass
class PerceiverPredictorConfig(PredictorConfig):
    """Configuration for the PerceiverPredictor."""

    depth: int = 0
    head_depth: int = 2

    def validate(self) -> None:
        """Validate the configuration."""
        super().validate()
        if self.depth != 0:
            raise ValueError(
                "PerceiverPredictor is attention-free; depth must be 0, got "
                f"{self.depth}"
            )

    def build(self) -> "PerceiverPredictor":
        """Build the PerceiverPredictor."""
        self.validate()
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs.pop("supported_modality_names")
        kwargs["supported_modalities"] = self.supported_modalities
        logger.info(f"PerceiverPredictor kwargs: {kwargs}")
        return PerceiverPredictor(**kwargs)

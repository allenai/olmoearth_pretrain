"""Multi-encoder: one shallow block stack ("tower") per input-modality group.

The plain :class:`~olmoearth_pretrain.nn.flexi_vit.Encoder` collapses every modality's
tokens into one sequence and runs the shared blocks over all of them (early fusion).
:class:`MultiEncoder` keeps that machinery (patch embeddings, encodings, RoPE
positions, masked-token removal, AttentionPooler, decoder contract) and inserts a
per-group stage before the shared blocks:

    patch embed -> encodings -> [tower_g over group g's tokens]  (per group, no cross-group attention)
                -> (+ fusion embedding of group g)
                -> shared blocks (``depth``; 0 = none)
                -> norm -> AttentionPooler read -> location tokens

With ``depth=0`` the only cross-modal interaction is the AttentionPooler's
cross-attention read over the concatenated tower outputs (Perceiver-IO-style fusion):
adding a modality later means adding a tower, with the location grid and decoder untouched.

A tower is a stack of the same ``Block`` as the trunk, so a group's tower runs the same
RoPE positions and masking rules as the shared blocks. Towers run on the full collapsed
sequence (``Encoder.pre_fusion_tokens``), so everything downstream is unchanged.

Not supported (raises): flash-attn packing inside towers, searchlight, and non-zero
token-exit depths (the target encoder must be the projection-only target or exit at 0).
"""

import logging
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from olmoearth_pretrain.nn.attention import Block
from olmoearth_pretrain.nn.flexi_vit import Encoder, EncoderConfig

logger = logging.getLogger(__name__)


class MultiEncoder(Encoder):
    """Encoder with a per-modality-group tower stage before the shared blocks.

    Args:
        tower_depth: Blocks per tower. 0 makes this exactly the plain Encoder (plus
            the optional fusion embeddings).
        tower_groups: ``{group_name: [modality_name, ...]}``. A modality belongs to
            at most one group; supported modalities in NO group pass through the tower
            stage untouched (no tower, no fusion embedding). Use that for decode-only
            map modalities, which never have visible tokens. None = one group per
            supported modality, named after it.
        fusion_embeddings: Add a learned per-group vector (zero-initialised) to each
            tower's output before fusion, so the shared blocks / pooler read can tell
            sources apart after the towers have mixed the channel embeddings in.
        **encoder_kwargs: Forwarded to :class:`Encoder`. ``depth`` is the number of
            SHARED (fusion) blocks after the towers.
    """

    def __init__(
        self,
        *,
        tower_depth: int = 2,
        tower_groups: dict[str, list[str]] | None = None,
        fusion_embeddings: bool = True,
        **encoder_kwargs: Any,
    ) -> None:
        """Initialize the MultiEncoder."""
        super().__init__(**encoder_kwargs)
        if self.use_flash_attn:
            raise NotImplementedError(
                "MultiEncoder towers do not support use_flash_attn=True yet"
            )
        if tower_depth < 0:
            raise ValueError(f"tower_depth must be >= 0, got {tower_depth}")
        self.tower_depth = tower_depth
        self.tower_groups = self._resolve_tower_groups(
            tower_groups, self.supported_modality_names
        )
        self.modality_to_group = {
            modality: group
            for group, modalities in self.tower_groups.items()
            for modality in modalities
        }
        drop_path = encoder_kwargs.get("drop_path", 0.0)
        self.towers = nn.ModuleDict(
            {
                group: nn.ModuleList(
                    [
                        Block(
                            self.embedding_size,
                            encoder_kwargs["num_heads"],
                            encoder_kwargs["mlp_ratio"],
                            qkv_bias=True,
                            qk_norm=encoder_kwargs.get("qk_norm", False),
                            norm_layer=nn.LayerNorm,
                            cross_attn=False,
                            drop_path=drop_path,
                            use_flash_attn=False,
                            position_encoding=self.position_encoding,
                            rope_base=self.rope_base,
                            rope_mixed_base=self.rope_mixed_base,
                            temporal_rope_dim_frac=self.temporal_rope_dim_frac,
                            rope_temporal_base=self.rope_temporal_base,
                        )
                        for _ in range(tower_depth)
                    ]
                )
                for group in self.tower_groups
            }
        )
        self.use_fusion_embeddings = fusion_embeddings
        if fusion_embeddings:
            self.fusion_embeddings = nn.ParameterDict(
                {
                    group: nn.Parameter(torch.zeros(self.embedding_size))
                    for group in self.tower_groups
                }
            )
        self.towers.apply(self._init_weights)
        logger.info(
            "MultiEncoder towers: %s (tower_depth=%d, fusion depth=%d)",
            self.tower_groups,
            tower_depth,
            len(self.blocks),
        )

    @staticmethod
    def _resolve_tower_groups(
        tower_groups: dict[str, list[str]] | None, supported_modality_names: list[str]
    ) -> dict[str, list[str]]:
        """Default to one group per modality; check groups are disjoint and known."""
        if tower_groups is None:
            return {name: [name] for name in supported_modality_names}
        seen: dict[str, str] = {}
        for group, modalities in tower_groups.items():
            if len(modalities) == 0:
                raise ValueError(f"tower group {group!r} has no modalities")
            for modality in modalities:
                if modality not in supported_modality_names:
                    raise ValueError(
                        f"tower group {group!r} lists {modality!r}, which is not a "
                        f"supported modality {supported_modality_names}"
                    )
                if modality in seen:
                    raise ValueError(
                        f"{modality!r} is in tower groups {seen[modality]!r} and "
                        f"{group!r}; each modality belongs to exactly one group"
                    )
                seen[modality] = group
        return {group: list(modalities) for group, modalities in tower_groups.items()}

    def _group_token_slices(
        self, modalities_to_dims_dict: dict[str, tuple]
    ) -> dict[str, list[tuple[int, int]]]:
        """Token index ranges ``[start, end)`` of each group in the collapsed sequence.

        The collapsed order is the insertion order of ``modalities_to_dims_dict`` (the
        same ``get_modalities_to_process`` order ``collapse_and_combine_hwtc`` uses).
        """
        slices: dict[str, list[tuple[int, int]]] = {g: [] for g in self.tower_groups}
        start = 0
        for modality, dims in modalities_to_dims_dict.items():
            num_tokens = 1
            for d in dims[1:-1]:
                num_tokens *= int(d)
            group = self.modality_to_group.get(modality)
            if group is not None:  # ungrouped modalities pass through
                slices[group].append((start, start + num_tokens))
            start += num_tokens
        return slices

    def _run_tower(
        self,
        blocks: nn.ModuleList,
        tokens: Tensor,
        mask: Tensor,
        positions: Tensor | None,
        fast_pass: bool,
    ) -> Tensor:
        """Run one tower over a group's tokens with the Encoder's masking rules.

        Mirrors ``Encoder.apply_attn``: drop non-visible tokens, attend among the rest
        (padded samples masked out in training), and scatter the results back. Tokens
        that were not visible come back as zeros; they are removed again before the
        shared blocks and are never keys in the pooler read.
        """
        full_tokens = tokens
        tokens, indices, new_mask, _, _, bool_mask = self._maybe_remove_masked_tokens(
            tokens, mask, fast_pass
        )
        if bool_mask is not None and not bool(bool_mask.any()):
            # No visible token in this group anywhere in the batch (e.g. the whole
            # group was dropped). Nothing to attend over; the zeros are masked anyway.
            return torch.zeros_like(full_tokens)
        if positions is not None and bool_mask is not None:
            positions, _, _, _, _ = self.remove_masked_tokens(positions, bool_mask)
        attn_mask = self._maybe_get_attn_mask(new_mask, fast_pass=fast_pass)
        for blk in blocks:
            tokens = blk(x=tokens, attn_mask=attn_mask, rope_positions=positions)
        return self._maybe_add_removed_tokens(tokens, indices, new_mask, fast_pass)

    def pre_fusion_tokens(
        self,
        tokens: Tensor,
        mask: Tensor,
        positions: Tensor | None,
        modalities_to_dims_dict: dict[str, tuple],
        fast_pass: bool,
    ) -> Tensor:
        """Run each group's tower over its tokens; add the group's fusion embedding."""
        if self.tower_depth == 0 and not self.use_fusion_embeddings:
            return tokens
        out = tokens
        for group, ranges in self._group_token_slices(modalities_to_dims_dict).items():
            if not ranges:
                continue  # none of this group's modalities are in the batch
            index = torch.cat(
                [torch.arange(s, e, device=tokens.device) for s, e in ranges]
            )
            group_tokens = tokens.index_select(1, index)
            if self.tower_depth > 0:
                group_tokens = self._run_tower(
                    self.towers[group],
                    group_tokens,
                    mask.index_select(1, index),
                    positions.index_select(1, index) if positions is not None else None,
                    fast_pass,
                )
            if self.use_fusion_embeddings:
                group_tokens = group_tokens + self.fusion_embeddings[group]
            out = out.index_copy(1, index, group_tokens)
        return out

    def apply_attn(
        self,
        x: dict[str, Tensor],
        timestamps: Tensor,
        patch_size: int,
        input_res: int,
        token_exit_cfg: dict[str, int] | None = None,
        fast_pass: bool = False,
        location_query_patch_size: int | None = None,
        searchlight: Any = None,
    ) -> tuple[dict[str, Tensor], dict[str, Any] | None, dict[str, Any] | None]:
        """Encoder.apply_attn with the tower stage; refuses what towers don't support."""
        if searchlight is not None:
            raise NotImplementedError("MultiEncoder does not support searchlight yet")
        if token_exit_cfg is not None and any(v != 0 for v in token_exit_cfg.values()):
            raise NotImplementedError(
                "MultiEncoder only supports token exit depth 0 (projection-only "
                f"target); got {token_exit_cfg}"
            )
        return super().apply_attn(
            x,
            timestamps,
            patch_size,
            input_res,
            token_exit_cfg=token_exit_cfg,
            fast_pass=fast_pass,
            location_query_patch_size=location_query_patch_size,
            searchlight=searchlight,
        )

    def apply_fsdp(self, **fsdp_kwargs: Any) -> None:
        """Shard the tower blocks like the trunk blocks, then the encoder."""
        for tower in self.towers.values():
            for block in tower:
                block.apply_fsdp(**fsdp_kwargs)
        super().apply_fsdp(**fsdp_kwargs)

    def apply_compile(self) -> None:
        """Compile the tower blocks like the trunk blocks."""
        for tower in self.towers.values():
            for block in tower:
                block.apply_compile()
        super().apply_compile()


@dataclass
class MultiEncoderConfig(EncoderConfig):
    """Configuration for :class:`MultiEncoder`.

    ``depth`` (inherited) is the number of SHARED blocks after the towers; 0 means the
    towers fuse only in the AttentionPooler read.
    """

    tower_depth: int = 2
    tower_groups: dict[str, list[str]] | None = None
    fusion_embeddings: bool = True

    def validate(self) -> None:
        """Validate the configuration."""
        super().validate()
        if self.tower_depth < 0:
            raise ValueError(f"tower_depth must be >= 0, got {self.tower_depth}")
        if self.use_flash_attn:
            raise ValueError("MultiEncoder does not support use_flash_attn=True")
        MultiEncoder._resolve_tower_groups(
            self.tower_groups, self.supported_modality_names
        )

    def build(self) -> "MultiEncoder":
        """Build the multi-encoder."""
        self.validate()
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs.pop("supported_modality_names")
        kwargs["supported_modalities"] = self.supported_modalities
        logger.info(f"MultiEncoder kwargs: {kwargs}")
        return MultiEncoder(**kwargs)

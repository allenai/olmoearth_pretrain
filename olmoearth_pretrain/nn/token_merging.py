"""Token Merging (ToMe) for OlmoEarth.

Based on "Token Merging: Your ViT But Faster" (Bolya et al., 2023)
https://arxiv.org/abs/2210.09461

This module provides token merging functionality to speed up transformer
forward passes by progressively merging similar tokens.
"""

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class ToMeConfig:
    """Configuration for Token Merging.

    Args:
        enabled: Whether ToMe is enabled
        merge_layers: List of layer indices after which to perform merging
        r_per_layer: Fixed number of tokens to reduce per merge layer
        r_ratio: Alternative to r_per_layer - ratio of tokens to reduce (e.g., 0.1 = 10%)
    """

    enabled: bool = False
    merge_layers: list[int] = field(default_factory=lambda: [3, 6, 9])
    r_per_layer: int = 16
    r_ratio: float | None = None

    def get_r(self, current_token_count: int, layer_idx: int) -> int:
        """Get number of tokens to reduce at this layer.

        Args:
            current_token_count: Current number of tokens (excluding protected)
            layer_idx: Current layer index

        Returns:
            Number of tokens to reduce
        """
        if self.r_ratio is not None:
            return int(current_token_count * self.r_ratio)
        return self.r_per_layer


@dataclass
class ToMeMergeInfo:
    """Information needed to unmerge tokens.

    Args:
        dst_idx: [B, r] destination indices for merged tokens
        src_idx: [B, r] source indices that were merged
        unm_idx: [B, n_src - r] unmerged source indices
        n_dst: Number of destination tokens
        n_src: Number of source tokens
        r: Number of tokens merged
        original_size: Original token count before merge
    """

    dst_idx: Tensor
    src_idx: Tensor
    unm_idx: Tensor
    n_dst: int
    n_src: int
    r: int
    original_size: int


def bipartite_soft_matching(
    tokens: Tensor,
    r: int,
    protected: int = 0,
) -> tuple[Tensor, ToMeMergeInfo | None]:
    """Bipartite soft matching for token merging.

    Partitions tokens into dst (destinations) and src (sources).
    Merges r most similar src tokens into their best-matching dst.

    Args:
        tokens: [B, N, D] token tensor
        r: Number of tokens to reduce
        protected: Number of tokens at the START to never merge (e.g., register tokens)

    Returns:
        merged_tokens: [B, N - r, D]
        merge_info: Information to unmerge later, or None if no merging done
    """
    B, N, D = tokens.shape

    if r <= 0:
        return tokens, None

    # Protect leading tokens (register tokens)
    if protected > 0:
        protected_tokens = tokens[:, :protected]
        tokens = tokens[:, protected:]
        N = N - protected

    # Can't merge more than we have
    r = min(r, N // 2)
    if r <= 0:
        if protected > 0:
            tokens = torch.cat([protected_tokens, tokens], dim=1)
        return tokens, None

    # Split into dst and src (alternating assignment)
    # dst gets even indices, src gets odd indices
    n_dst = (N + 1) // 2
    n_src = N - n_dst

    dst_tokens = tokens[:, 0::2]  # [B, n_dst, D] - even indices
    src_tokens = tokens[:, 1::2]  # [B, n_src, D] - odd indices

    # Compute cosine similarity between dst and src
    dst_norm = dst_tokens / (dst_tokens.norm(dim=-1, keepdim=True) + 1e-6)
    src_norm = src_tokens / (src_tokens.norm(dim=-1, keepdim=True) + 1e-6)

    # [B, n_dst, n_src] similarity matrix
    scores = torch.bmm(dst_norm, src_norm.transpose(1, 2))

    # For each src token, find best matching dst
    node_max, node_idx = scores.max(dim=1)  # [B, n_src] - best dst for each src

    # Sort src tokens by their best similarity (descending)
    # Top r will be merged, rest will be kept
    edge_idx = node_max.argsort(dim=-1, descending=True)  # [B, n_src]

    # Indices of src tokens to merge (top r most similar)
    src_idx = edge_idx[:, :r]  # [B, r]
    # Indices of src tokens to keep (rest)
    unm_idx = edge_idx[:, r:]  # [B, n_src - r]
    # Corresponding dst indices for merged tokens
    dst_idx = node_idx.gather(dim=-1, index=src_idx)  # [B, r]

    # Merge: for each (src, dst) pair, average them into dst
    merged_dst = dst_tokens.clone()

    # Count how many src tokens merge into each dst (for averaging)
    ones = torch.ones(B, r, 1, device=tokens.device, dtype=tokens.dtype)
    dst_counts = torch.zeros(B, n_dst, 1, device=tokens.device, dtype=tokens.dtype)
    dst_counts.scatter_add_(1, dst_idx.unsqueeze(-1), ones)

    # Sum src tokens into dst positions
    src_to_merge = src_tokens.gather(
        dim=1, index=src_idx.unsqueeze(-1).expand(-1, -1, D)
    )  # [B, r, D]

    merged_dst.scatter_add_(1, dst_idx.unsqueeze(-1).expand(-1, -1, D), src_to_merge)

    # Average: divide by (1 + count) since dst already has 1 token
    merged_dst = merged_dst / (1 + dst_counts)

    # Gather unmerged src tokens
    unmerged_src = src_tokens.gather(
        dim=1, index=unm_idx.unsqueeze(-1).expand(-1, -1, D)
    )  # [B, n_src - r, D]

    # Concatenate: [dst (with merges), unmerged_src]
    merged = torch.cat([merged_dst, unmerged_src], dim=1)  # [B, n_dst + n_src - r, D]

    # Re-add protected tokens at the start
    if protected > 0:
        merged = torch.cat([protected_tokens, merged], dim=1)

    merge_info = ToMeMergeInfo(
        dst_idx=dst_idx,
        src_idx=src_idx,
        unm_idx=unm_idx,
        n_dst=n_dst,
        n_src=n_src,
        r=r,
        original_size=N + protected,
    )

    return merged, merge_info


def unmerge_tokens(
    tokens: Tensor,
    merge_info: ToMeMergeInfo,
    protected: int = 0,
) -> Tensor:
    """Unmerge tokens by duplicating merged tokens back to original positions.

    Args:
        tokens: [B, N_merged, D] merged token tensor
        merge_info: Information from the merge operation
        protected: Number of protected tokens at start

    Returns:
        unmerged_tokens: [B, N_original, D]
    """
    if merge_info is None:
        return tokens

    B, _, D = tokens.shape
    n_dst = merge_info.n_dst
    n_src = merge_info.n_src

    # Separate protected tokens
    if protected > 0:
        protected_tokens = tokens[:, :protected]
        tokens = tokens[:, protected:]

    # Split merged tokens back into dst and unmerged_src
    dst_tokens = tokens[:, :n_dst]  # [B, n_dst, D]
    unmerged_src = tokens[:, n_dst:]  # [B, n_src - r, D]

    # Reconstruct full src by:
    # 1. Create empty src tensor
    # 2. Place unmerged tokens at unm_idx positions
    # 3. Copy merged tokens (from dst) to src_idx positions

    full_src = torch.zeros(B, n_src, D, device=tokens.device, dtype=tokens.dtype)

    # Place unmerged src tokens
    full_src.scatter_(
        1, merge_info.unm_idx.unsqueeze(-1).expand(-1, -1, D), unmerged_src
    )

    # Get merged tokens from their dst positions and place at src positions
    merged_from_dst = dst_tokens.gather(
        dim=1, index=merge_info.dst_idx.unsqueeze(-1).expand(-1, -1, D)
    )  # [B, r, D]

    full_src.scatter_(
        1, merge_info.src_idx.unsqueeze(-1).expand(-1, -1, D), merged_from_dst
    )

    # Interleave dst and src back to original order
    # Original: [dst[0], src[0], dst[1], src[1], ...]
    N = n_dst + n_src
    unmerged = torch.zeros(B, N, D, device=tokens.device, dtype=tokens.dtype)
    unmerged[:, 0::2] = dst_tokens  # even positions
    unmerged[:, 1::2] = full_src  # odd positions

    # Re-add protected tokens
    if protected > 0:
        unmerged = torch.cat([protected_tokens, unmerged], dim=1)

    return unmerged


def apply_tome_unmerge_stack(
    tokens: Tensor,
    merge_info_stack: list[ToMeMergeInfo],
    protected: int = 0,
) -> Tensor:
    """Apply all unmerge operations in reverse order to restore original token count.

    Args:
        tokens: [B, N_merged, D] merged token tensor
        merge_info_stack: List of merge info from each merge operation
        protected: Number of protected tokens at start

    Returns:
        unmerged_tokens: [B, N_original, D]
    """
    for merge_info in reversed(merge_info_stack):
        tokens = unmerge_tokens(tokens, merge_info, protected)
    return tokens

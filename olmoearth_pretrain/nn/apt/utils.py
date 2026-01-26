"""Utilities for APT adaptive patchification.

Includes packing/unpacking for variable-length token sequences
and other helper functions.
"""

import logging

import torch
from torch import Tensor

from olmoearth_pretrain.nn.apt.partitioner import PatchDescriptor

logger = logging.getLogger(__name__)


def pack_adaptive_tokens(
    tokens_list: list[Tensor],
    positions_list: list[Tensor] | None = None,
) -> tuple[Tensor, Tensor, list[int]]:
    """Pack variable-length token sequences for batch processing.

    Concatenates tokens from multiple samples into a single sequence
    for use with flash attention's block-diagonal attention masks.

    Args:
        tokens_list: List of token tensors, each [N_i, D]
        positions_list: Optional list of position tensors, each [N_i, 2]

    Returns:
        Tuple of:
            - packed_tokens: Concatenated tokens [sum(N_i), D]
            - cu_seqlens: Cumulative sequence lengths [B+1]
            - lengths: Original sequence lengths [B]
    """
    lengths = [t.shape[0] for t in tokens_list]
    packed_tokens = torch.cat(tokens_list, dim=0)

    # Compute cumulative sequence lengths for flash attention
    cu_seqlens = torch.zeros(
        len(lengths) + 1, dtype=torch.int32, device=packed_tokens.device
    )
    cu_seqlens[1:] = torch.cumsum(
        torch.tensor(lengths, device=packed_tokens.device), dim=0
    )

    return packed_tokens, cu_seqlens, lengths


def unpack_adaptive_tokens(
    packed_tokens: Tensor,
    lengths: list[int],
) -> list[Tensor]:
    """Unpack concatenated tokens back into individual sequences.

    Args:
        packed_tokens: Concatenated tokens [sum(N_i), D]
        lengths: Original sequence lengths [B]

    Returns:
        List of token tensors, each [N_i, D]
    """
    tokens_list = []
    start = 0
    for length in lengths:
        tokens_list.append(packed_tokens[start : start + length])
        start += length
    return tokens_list


def pack_with_positions(
    tokens_list: list[Tensor],
    positions_list: list[Tensor],
) -> tuple[Tensor, Tensor, Tensor, list[int]]:
    """Pack tokens and positions together.

    Args:
        tokens_list: List of token tensors, each [N_i, D]
        positions_list: List of position tensors, each [N_i, 2]

    Returns:
        Tuple of:
            - packed_tokens: Concatenated tokens [sum(N_i), D]
            - packed_positions: Concatenated positions [sum(N_i), 2]
            - cu_seqlens: Cumulative sequence lengths [B+1]
            - lengths: Original sequence lengths [B]
    """
    packed_tokens, cu_seqlens, lengths = pack_adaptive_tokens(tokens_list)
    packed_positions = torch.cat(positions_list, dim=0)
    return packed_tokens, packed_positions, cu_seqlens, lengths


def descriptors_to_mask_indices(
    patch_descriptors: list[PatchDescriptor],
    base_grid_shape: tuple[int, int],
    base_patch_size: int,
) -> dict[int, list[tuple[int, int]]]:
    """Convert patch descriptors to base-grid cell indices.

    For each patch, returns the list of base-grid cells it covers.
    Useful for mapping between adaptive patches and base-grid masks.

    Args:
        patch_descriptors: List of patch descriptors
        base_grid_shape: (H_grid, W_grid) of the base patch grid
        base_patch_size: Base patch size in pixels

    Returns:
        Dict mapping patch index to list of (y, x) base-grid coordinates
    """
    mapping = {}
    h_grid, w_grid = base_grid_shape

    for idx, desc in enumerate(patch_descriptors):
        size_in_base = desc.size // base_patch_size
        cells = []
        for dy in range(size_in_base):
            for dx in range(size_in_base):
                y = desc.y + dy
                x = desc.x + dx
                if 0 <= y < h_grid and 0 <= x < w_grid:
                    cells.append((y, x))
        mapping[idx] = cells

    return mapping


def expand_tokens_to_base_grid(
    tokens: Tensor,
    patch_descriptors: list[PatchDescriptor],
    base_grid_shape: tuple[int, int],
    base_patch_size: int,
) -> Tensor:
    """Expand adaptive tokens to a dense base-grid feature map.

    For dense prediction tasks that need rectangular feature maps.
    Larger patches are repeated to fill their covered area.

    Args:
        tokens: Token embeddings [N, D]
        patch_descriptors: List of patch descriptors
        base_grid_shape: (H_grid, W_grid) of the base patch grid
        base_patch_size: Base patch size in pixels

    Returns:
        Dense feature map [H_grid, W_grid, D]
    """
    h_grid, w_grid = base_grid_shape
    d = tokens.shape[1]
    device = tokens.device
    dtype = tokens.dtype

    feature_map = torch.zeros((h_grid, w_grid, d), device=device, dtype=dtype)

    for idx, desc in enumerate(patch_descriptors):
        size_in_base = desc.size // base_patch_size
        token = tokens[idx]  # [D]

        for dy in range(size_in_base):
            for dx in range(size_in_base):
                y = desc.y + dy
                x = desc.x + dx
                if 0 <= y < h_grid and 0 <= x < w_grid:
                    feature_map[y, x] = token

    return feature_map


def compute_position_embeddings(
    positions: Tensor,
    max_grid_size: int,
    embedding_dim: int,
    device: torch.device | None = None,
) -> Tensor:
    """Compute 2D sinusoidal position embeddings for adaptive patch positions.

    Uses the center position of each patch to compute embeddings.

    Args:
        positions: Patch center positions [N, 2] in base patch units
        max_grid_size: Maximum grid size for normalization
        embedding_dim: Dimension of position embeddings (should be divisible by 4)
        device: Device to create embeddings on

    Returns:
        Position embeddings [N, embedding_dim]
    """
    if device is None:
        device = positions.device

    dim = embedding_dim // 4  # Split between x and y, sin and cos

    # Normalize positions to [0, 1]
    x_norm = positions[:, 0] / max_grid_size
    y_norm = positions[:, 1] / max_grid_size

    # Create frequency bands
    freq_seq = torch.arange(dim, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (freq_seq / dim))

    # Compute embeddings
    x_emb = x_norm.unsqueeze(1) * inv_freq.unsqueeze(0)  # [N, dim]
    y_emb = y_norm.unsqueeze(1) * inv_freq.unsqueeze(0)  # [N, dim]

    pos_emb = torch.cat(
        [
            torch.sin(x_emb),
            torch.cos(x_emb),
            torch.sin(y_emb),
            torch.cos(y_emb),
        ],
        dim=1,
    )  # [N, embedding_dim]

    return pos_emb


def get_scale_for_patch_size(patch_size: int, base_patch_size: int) -> int:
    """Get scale index for a given patch size.

    Args:
        patch_size: Patch size in pixels
        base_patch_size: Base patch size in pixels

    Returns:
        Scale index (0 = base, 1 = 2x, etc.)
    """
    ratio = patch_size // base_patch_size
    scale = 0
    while ratio > 1:
        ratio //= 2
        scale += 1
    return scale


def validate_patch_descriptors(
    patch_descriptors: list[PatchDescriptor],
    image_shape: tuple[int, int],
    base_patch_size: int,
) -> bool:
    """Validate that patch descriptors form a valid tiling.

    Args:
        patch_descriptors: List of patch descriptors
        image_shape: (H, W) of the image
        base_patch_size: Base patch size in pixels

    Returns:
        True if valid, False otherwise
    """
    h, w = image_shape
    h_grid = h // base_patch_size
    w_grid = w // base_patch_size

    # Create coverage grid
    coverage = set()

    for desc in patch_descriptors:
        size_in_base = desc.size // base_patch_size
        for dy in range(size_in_base):
            for dx in range(size_in_base):
                y = desc.y + dy
                x = desc.x + dx
                cell = (y, x)
                if cell in coverage:
                    logger.warning(f"Overlapping coverage at {cell}")
                    return False
                coverage.add(cell)

    # Check all cells are covered
    expected = {(y, x) for y in range(h_grid) for x in range(w_grid)}
    missing = expected - coverage
    extra = coverage - expected

    if missing:
        logger.warning(f"Missing coverage: {len(missing)} cells")
        return False
    if extra:
        logger.warning(f"Extra coverage: {len(extra)} cells")
        return False

    return True

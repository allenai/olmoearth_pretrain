"""Adaptive Patch Embedding with ZeroMLP.

Handles mixed-size patches by combining resize-based embeddings
with subpatch detail aggregation via zero-initialized MLP.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from olmoearth_pretrain.nn.apt.partitioner import PatchDescriptor
from olmoearth_pretrain.nn.flexi_patch_embed import FlexiPatchEmbed

logger = logging.getLogger(__name__)


class ConvDownsample(nn.Module):
    """Convolutional downsampling for aggregating subpatch embeddings.

    Reduces a 2^scale x 2^scale grid of subpatch embeddings to a single embedding.
    """

    def __init__(self, embedding_size: int, scale: int):
        """Initialize the conv downsample module.

        Args:
            embedding_size: Size of embeddings
            scale: Scale index (1 = 2x2, 2 = 4x4, etc.)
        """
        super().__init__()
        self.embedding_size = embedding_size
        self.scale = scale
        self.grid_size = 2**scale

        # Stack of 2x2 stride-2 convs to reduce grid to 1x1
        layers = []
        for _ in range(scale):
            layers.append(
                nn.Conv2d(
                    embedding_size,
                    embedding_size,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                )
            )
        self.convs = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Subpatch embeddings with shape [B, D, H, W] where H=W=2^scale

        Returns:
            Aggregated embedding with shape [B, D]
        """
        # x: [B, D, 2^scale, 2^scale]
        out = self.convs(x)  # [B, D, 1, 1]
        return out.squeeze(-1).squeeze(-1)  # [B, D]



class AdaptivePatchEmbed(nn.Module):
    """Adaptive patch embedding for mixed-size patches.

    For each patch:
    1. Resize path: Resize large patch to base size, apply base_embed -> e_resize
    2. Detail path: Split into subpatches, embed each, aggregate via conv -> e_sub
    3. Combine: e_final = e_resize + zero_mlp(e_sub)
    """

    def __init__(
        self,
        num_scales: int,
        embedding_size: int,
        base_patch_size: int,
    ):
        """Initialize adaptive patch embedding.

        Args:
            base_patch_embed: Pretrained FlexiPatchEmbed for base patch size
            num_scales: Number of patch size scales (1 = base only)
            embedding_size: Size of embeddings
            base_patch_size: Base (smallest) patch size in pixels
        """
        super().__init__()
        self.num_scales = num_scales
        self.embedding_size = embedding_size
        self.base_patch_size = base_patch_size

        # Compute patch sizes for each scale
        self.patch_sizes = [base_patch_size * (2**i) for i in range(num_scales)]

        # Conv downsampling for each scale > 0
        self.conv_downsample = nn.ModuleList()
        for scale in range(1, num_scales):
            self.conv_downsample.append(ConvDownsample(embedding_size, scale))



    def merge_token_block(
        self,
        block: torch.Tensor,
        scale: int,
    ) -> torch.Tensor:
        """Merge a block of base tokens into a single coarse token.

        Args:
            block: Token block with shape [B, D, 2^scale, 2^scale]
            scale: Scale index (> 0)

        Returns:
            Merged token with shape [B, D]
        """
        conv_idx = scale - 1  # scale 1 -> index 0, scale 2 -> index 1
        return self.conv_downsample[conv_idx](block)  # [B, D]

    def forward(
        self,
        base_patch_embeddings: torch.Tensor,
        patch_descriptors: list[list[list[PatchDescriptor]]],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Merge base-size tokens into adaptive tokens per batch and timestep.

        Args:
            base_patch_embeddings: [B, H, W, T, D] tokens from standard
                patchification + encodings.
            patch_descriptors: Nested list [B][T] of PatchDescriptor lists from the partitioner

        Returns:
            all_tokens: List of [N_i, D] tensors, one per batch element (all timesteps concatenated)
            all_positions: List of [N_i, 4] int tensors (y, x, size_in_base, timestep) per batch element
        """
        b, h, w, t, d = base_patch_embeddings.shape
        all_tokens = []
        all_positions = []

        for bi in range(b):
            sample_tokens = []
            sample_positions = []

            for ti in range(t):
                descs = patch_descriptors[bi][ti]

                for desc in descs:
                    size_in_base = 2 ** desc.scale

                    if desc.scale == 0:
                        # Base scale: just take the token directly
                        token = base_patch_embeddings[bi, desc.y, desc.x, ti, :]  # [D]
                        sample_tokens.append(token.unsqueeze(0))
                    else:
                        # Coarser scale: extract the 2^s x 2^s block and merge
                        block = base_patch_embeddings[
                            bi,
                            desc.y : desc.y + size_in_base,
                            desc.x : desc.x + size_in_base,
                            ti,
                            :,
                        ]  # [size_in_base, size_in_base, D]
                        # ConvDownsample expects [1, D, H, W]
                        block = rearrange(block, "h w d -> 1 d h w")
                        token = self.merge_token_block(block, desc.scale)  # [1, D]
                        sample_tokens.append(token)

                    sample_positions.append([desc.y, desc.x, size_in_base, ti])

            if sample_tokens:
                all_tokens.append(torch.cat(sample_tokens, dim=0))  # [N_i, D]
                all_positions.append(
                    torch.tensor(sample_positions, dtype=torch.long, device=base_patch_embeddings.device)
                )  # [N_i, 4]
            else:
                all_tokens.append(torch.zeros(0, d, device=base_patch_embeddings.device))
                all_positions.append(torch.zeros(0, 4, dtype=torch.long, device=base_patch_embeddings.device))

        return all_tokens, all_positions

class AdaptiveMultiModalPatchEmbed(nn.Module):
    """Adaptive patch embedding for multiple modalities.

    Wraps AdaptivePatchEmbed for use with the helios multi-modal architecture.
    """

    def __init__(
        self,
        modality_patch_embeds: dict[str, AdaptivePatchEmbed],
        apt_modalities: list[str],
    ):
        """Initialize multi-modal adaptive embedding.

        Args:
            modality_patch_embeds: Dict mapping modality name to AdaptivePatchEmbed
            apt_modalities: List of modality names that should use APT
        """
        super().__init__()
        self.modality_patch_embeds = nn.ModuleDict(modality_patch_embeds)
        self.apt_modalities = apt_modalities

    def forward(
        self,
        modality_data: dict[str, torch.Tensor],
        apt_partitions: dict[str, list[PatchDescriptor]],
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass for all modalities.

        Args:
            modality_data: Dict mapping modality name to image tensor
            apt_partitions: Dict mapping modality name to patch descriptors

        Returns:
            Dict mapping modality name to (tokens, positions) tuples
        """
        results = {}

        for modality_name, embed_module in self.modality_patch_embeds.items():
            if modality_name not in modality_data:
                continue

            image = modality_data[modality_name]
            descriptors = apt_partitions.get(modality_name, [])

            if descriptors:
                tokens, positions = embed_module.forward(image, descriptors)
                results[modality_name] = (tokens, positions)

        return results

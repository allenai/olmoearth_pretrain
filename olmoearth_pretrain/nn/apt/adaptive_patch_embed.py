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


class ZeroMLP(nn.Module):
    """Zero-initialized MLP for combining embeddings.

    Initialized with zeros so that initially e_final = e_resize,
    preserving pretrained model behavior.
    """

    def __init__(self, embedding_size: int):
        """Initialize the ZeroMLP.

        Args:
            embedding_size: Size of embeddings
        """
        super().__init__()
        self.linear = nn.Linear(embedding_size, embedding_size)

        # Zero initialization
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.linear(x)


class AdaptivePatchEmbed(nn.Module):
    """Adaptive patch embedding for mixed-size patches.

    For each patch:
    1. Resize path: Resize large patch to base size, apply base_embed -> e_resize
    2. Detail path: Split into subpatches, embed each, aggregate via conv -> e_sub
    3. Combine: e_final = e_resize + zero_mlp(e_sub)
    """

    def __init__(
        self,
        base_patch_embed: FlexiPatchEmbed,
        num_scales: int = 3,
        embedding_size: int = 768,
        base_patch_size: int = 16,
    ):
        """Initialize adaptive patch embedding.

        Args:
            base_patch_embed: Pretrained FlexiPatchEmbed for base patch size
            num_scales: Number of patch size scales (1 = base only)
            embedding_size: Size of embeddings
            base_patch_size: Base (smallest) patch size in pixels
        """
        super().__init__()
        self.base_patch_embed = base_patch_embed
        self.num_scales = num_scales
        self.embedding_size = embedding_size
        self.base_patch_size = base_patch_size

        # Compute patch sizes for each scale
        self.patch_sizes = [base_patch_size * (2**i) for i in range(num_scales)]

        # Conv downsampling for each scale > 0
        self.conv_downsample = nn.ModuleList()
        for scale in range(1, num_scales):
            self.conv_downsample.append(ConvDownsample(embedding_size, scale))

        # ZeroMLP for combining resize and detail paths
        self.zero_mlp = ZeroMLP(embedding_size)

    def embed_base_patch(self, patch: torch.Tensor) -> torch.Tensor:
        """Embed a base-size patch.

        Args:
            patch: Patch with shape [B, H, W, C] where H=W=base_patch_size

        Returns:
            Embedding with shape [B, D]
        """
        # base_patch_embed expects [B, H, W, C] and returns [B, 1, 1, D]
        # for a single-patch input
        # Pass patch_size to tell FlexiPatchEmbed what size our input patches are
        # The input H, W should match base_patch_size
        input_size = patch.shape[1]  # H dimension
        embedded = self.base_patch_embed(patch, patch_size=input_size)

        # Flatten spatial dims
        if embedded.ndim == 4:
            embedded = embedded.view(embedded.shape[0], -1)
        elif embedded.ndim == 3:
            embedded = embedded.squeeze(1)

        return embedded

    def embed_large_patch(
        self,
        patch: torch.Tensor,
        scale: int,
    ) -> torch.Tensor:
        """Embed a large patch using resize + detail aggregation.

        Args:
            patch: Patch with shape [B, H, W, C] where H=W=patch_sizes[scale]
            scale: Scale index (> 0)

        Returns:
            Embedding with shape [B, D]
        """
        b, h, w, c = patch.shape
        patch_size = self.patch_sizes[scale]

        # Path 1: Resize to base size and embed
        patch_resized = rearrange(patch, "b h w c -> b c h w")
        patch_resized = F.interpolate(
            patch_resized,
            size=(self.base_patch_size, self.base_patch_size),
            mode="bicubic",
            antialias=True,
        )
        patch_resized = rearrange(patch_resized, "b c h w -> b h w c")
        e_resize = self.embed_base_patch(patch_resized)

        # Path 2: Split into subpatches, embed each, aggregate
        # Number of subpatches per dimension
        num_sub = patch_size // self.base_patch_size  # = 2^scale

        # Split into subpatches: [B, H, W, C] -> [B, num_sub, num_sub, base_size, base_size, C]
        subpatches = rearrange(
            patch,
            "b (nh ph) (nw pw) c -> b nh nw ph pw c",
            nh=num_sub,
            nw=num_sub,
            ph=self.base_patch_size,
            pw=self.base_patch_size,
        )

        # Embed each subpatch
        subpatches_flat = rearrange(subpatches, "b nh nw ph pw c -> (b nh nw) ph pw c")
        subpatch_embeds = self.embed_base_patch(subpatches_flat)  # [(B*nh*nw), D]

        # Reshape to grid
        subpatch_embeds = rearrange(
            subpatch_embeds,
            "(b nh nw) d -> b d nh nw",
            b=b,
            nh=num_sub,
            nw=num_sub,
        )

        # Aggregate via conv downsample
        conv_idx = scale - 1  # scale 1 -> conv 0, etc.
        e_sub = self.conv_downsample[conv_idx](subpatch_embeds)  # [B, D]

        # Combine: e_final = e_resize + zero_mlp(e_sub)
        e_final = e_resize + self.zero_mlp(e_sub)

        return e_final

    def forward_single_scale(
        self,
        patches: torch.Tensor,
        scale: int,
    ) -> torch.Tensor:
        """Forward pass for patches of a single scale.

        Args:
            patches: Patches with shape [N, H, W, C]
            scale: Scale index

        Returns:
            Embeddings with shape [N, D]
        """
        if scale == 0:
            return self.embed_base_patch(patches)
        else:
            return self.embed_large_patch(patches, scale)

    def forward(
        self,
        image: torch.Tensor,
        patch_descriptors: list[PatchDescriptor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for adaptive patches.

        Args:
            image: Full image with shape [H, W, C] or [B, H, W, C]
            patch_descriptors: List of patch descriptors from partitioner

        Returns:
            Tuple of:
                - tokens: Embeddings with shape [N, D]
                - positions: Patch center positions with shape [N, 2] in base patch units
        """
        if image.ndim == 3:
            image = image.unsqueeze(0)

        device = image.device
        b, h, w, c = image.shape

        if b != 1:
            raise ValueError(
                "AdaptivePatchEmbed.forward expects unbatched or batch_size=1 input. "
                "Use forward_batch for batched inputs."
            )

        tokens = []
        positions = []

        for desc in patch_descriptors:
            # Extract patch from image
            x_px = desc.x * self.base_patch_size
            y_px = desc.y * self.base_patch_size
            size = desc.size

            patch = image[:, y_px : y_px + size, x_px : x_px + size, :]

            # Embed based on scale
            embedding = self.forward_single_scale(patch, desc.scale)
            tokens.append(embedding)

            # Position is center of patch in base patch units
            size_in_base = size // self.base_patch_size
            center_x = desc.x + size_in_base / 2.0
            center_y = desc.y + size_in_base / 2.0
            positions.append([center_x, center_y])

        tokens = torch.cat(tokens, dim=0)  # [N, D]
        positions = torch.tensor(
            positions, device=device, dtype=torch.float32
        )  # [N, 2]

        return tokens, positions

    def forward_batch(
        self,
        images: list[torch.Tensor],
        patch_descriptors_batch: list[list[PatchDescriptor]],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[int]]:
        """Forward pass for a batch of images with their patch descriptors.

        Args:
            images: List of images, each [H, W, C]
            patch_descriptors_batch: List of patch descriptor lists

        Returns:
            Tuple of:
                - tokens_list: List of embeddings, each [N_i, D]
                - positions_list: List of positions, each [N_i, 2]
                - lengths: Token count per image
        """
        tokens_list = []
        positions_list = []
        lengths = []

        for image, descriptors in zip(images, patch_descriptors_batch):
            tokens, positions = self.forward(image, descriptors)
            tokens_list.append(tokens)
            positions_list.append(positions)
            lengths.append(len(descriptors))

        return tokens_list, positions_list, lengths


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

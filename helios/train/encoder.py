"""Mock Encoder code for the Latent MIM loss set up."""

import torch
import torch.nn as nn


class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block implementation for self-attention and MLP processing."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        """Initialize the Transformer Encoder Block components."""
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Transformer Encoder Block."""
        # Self-attention block
        x = x + self.attn(*[self.norm1(x)] * 3)[0]
        # MLP block
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEncoder(nn.Module):
    """Module that encodes input data into patches and processes them through a transformer."""

    def __init__(
        self,
        in_channels: int,
        patch_size: int = 16,
        time_patch_size: int = 4,  # New parameter for temporal patch size
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        mask_ratio: float = 0.75,
    ):
        """Initialize the PatchEncoder with specified parameters.

        Args:
            in_channels: Number of input channels.
            patch_size: Size of spatial patches.
            time_patch_size: Size of temporal patches.
            embed_dim: Embedding dimension.
            depth: Number of transformer blocks.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio for MLP hidden dimension.
            dropout: Dropout rate.
            mask_ratio: Ratio of patches to mask.
        """
        super().__init__()
        self.patch_size = patch_size
        self.time_patch_size = time_patch_size
        self.mask_ratio = mask_ratio
        self.embed_dim = embed_dim

        # 3D Patch embedding layer
        self.patch_embed = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(time_patch_size, patch_size, patch_size),
            stride=(time_patch_size, patch_size, patch_size),
        )

        # Position embedding will now account for time dimension
        # If input is (B, C, 32, 224, 224) with patch sizes (4, 16, 16)
        # Then we get (32/4 * 224/16 * 224/16) = 8 * 14 * 14 = 1568 patches
        self.num_patches = None  # Will be set dynamically in forward pass
        max_num_patches = 2048  # Set this to your maximum expected number of patches
        self.pos_embed = nn.Parameter(torch.zeros(1, max_num_patches, embed_dim))

        # Transformer encoder layers
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        # Initialize position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def random_augment(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random noise augmentation to patches."""
        # Add Gaussian noise with mean 0 and std 0.1
        noise = torch.randn_like(x) * 0.1
        x_aug = x + noise
        return x_aug

    def forward(self, x: torch.Tensor, apply_aug: bool = True) -> torch.Tensor:
        """Process input tensor through patch embedding and transformer blocks.

        Args:
            x: Input tensor of shape (B, C, T, H, W).
            apply_aug: Whether to apply augmentation.

        Returns:
            Encoded tensor after transformer processing.
        """
        # x shape: (B, C, T, H, W)
        # Patch embedding
        patches = self.patch_embed(x)  # (B, embed_dim, T', H', W')
        B, C, T, H, W = patches.shape
        patches = patches.flatten(2).transpose(1, 2)  # B, N, C where N = T*H*W\

        # Set num_patches if not set
        if self.num_patches is None:
            self.num_patches = T * H * W

        # Add position embeddings
        patches = patches + self.pos_embed[:, : patches.size(1), :]

        if apply_aug:
            # Apply random augmentation
            aug_patches = self.random_augment(patches)

            # Pass through transformer blocks
            x = aug_patches
            for block in self.transformer_blocks:
                x = block(x)
            encoded = self.norm(x)

            return encoded
        else:
            # Pass through transformer blocks without augmentation
            x = patches
            for block in self.transformer_blocks:
                x = block(x)
            encoded = self.norm(x)

            return encoded


# Encode all the modalities into a single vector by looping over them

# Mask out random patches and encode only the rest

# Predict the masked out latents and compute the patch disc loss

# Even simpler pass in some simple augmentations and compute the patch disk across this


# The Encoder needs to patch the data into 16 X 16 patches and then encode each patch into a single vector

"""Unit tests for ChannelAttentionPatchEmbed."""

import pytest
import torch

from olmoearth_pretrain.data.constants import Modality, ModalitySpec
from olmoearth_pretrain.nn.flexi_patch_embed import (
    ChannelAttentionPatchEmbed,
    FlexiPatchEmbed,
)
from olmoearth_pretrain.nn.flexi_vit import EncoderConfig, MultiModalPatchEmbeddings
from olmoearth_pretrain.nn.tokenization import ModalityTokenization, TokenizationConfig


@pytest.fixture
def s2_modality() -> ModalitySpec:
    """Sentinel-2 L2A modality spec fixture."""
    return Modality.SENTINEL2_L2A


@pytest.fixture
def s2_single_bandset_config() -> TokenizationConfig:
    """Single bandset tokenization config for S2 and Landsat."""
    return TokenizationConfig(
        overrides={
            "sentinel2_l2a": ModalityTokenization(
                band_groups=[
                    [
                        "B02",
                        "B03",
                        "B04",
                        "B08",
                        "B05",
                        "B06",
                        "B07",
                        "B8A",
                        "B11",
                        "B12",
                        "B01",
                        "B09",
                    ]
                ]
            ),
            "landsat": ModalityTokenization(
                band_groups=[
                    ["B8", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"]
                ],
            ),
        }
    )


class TestChannelAttentionPatchEmbed:
    """Tests for ChannelAttentionPatchEmbed module."""

    def test_output_shape_4d(self, s2_modality: ModalitySpec) -> None:
        """4D input [B, H, W, C] produces correct output shape."""
        embed = ChannelAttentionPatchEmbed(
            modality_spec=s2_modality,
            base_patch_size_at_16=8,
            num_bands=12,
            embedding_size=64,
            attn_dim=128,
            num_heads=4,
        )
        p = s2_modality.image_tile_size_factor * 8
        H = W = p * 2
        x = torch.randn(2, H, W, 12)
        out = embed(x)
        assert out.shape == (2, 2, 2, 64)

    def test_output_shape_5d(self, s2_modality: ModalitySpec) -> None:
        """5D input [B, H, W, T, C] produces correct output shape."""
        embed = ChannelAttentionPatchEmbed(
            modality_spec=s2_modality,
            base_patch_size_at_16=8,
            num_bands=12,
            embedding_size=64,
            attn_dim=128,
            num_heads=4,
        )
        p = s2_modality.image_tile_size_factor * 8
        H = W = p * 2
        x = torch.randn(2, H, W, 3, 12)
        out = embed(x)
        assert out.shape == (2, 2, 2, 3, 64)

    @pytest.mark.parametrize("runtime_patch_size", [1, 2, 4])
    def test_flexi_patch_size(
        self, s2_modality: ModalitySpec, runtime_patch_size: int
    ) -> None:
        """Variable patch sizes produce outputs with different spatial dims."""
        base_ps = 8
        embed = ChannelAttentionPatchEmbed(
            modality_spec=s2_modality,
            base_patch_size_at_16=base_ps,
            num_bands=12,
            embedding_size=64,
            attn_dim=128,
            num_heads=4,
        )
        factor = s2_modality.image_tile_size_factor
        p = base_ps * factor
        H = W = p * 4
        x = torch.randn(2, H, W, 12)
        out = embed(x, patch_size=runtime_patch_size)
        expected_patches = H // (runtime_patch_size * factor)
        assert out.shape == (2, expected_patches, expected_patches, 64)

    def test_band_mask_no_error(self, s2_modality: ModalitySpec) -> None:
        """Forward with band_mask runs without error."""
        embed = ChannelAttentionPatchEmbed(
            modality_spec=s2_modality,
            base_patch_size_at_16=8,
            num_bands=12,
            embedding_size=64,
            attn_dim=128,
            num_heads=4,
        )
        p = s2_modality.image_tile_size_factor * 8
        H = W = p * 2
        x = torch.randn(2, H, W, 12)
        band_mask = torch.zeros(2, 12, dtype=torch.bool)
        band_mask[0, :3] = True
        band_mask[1, 5:8] = True
        out = embed(x, band_mask=band_mask)
        assert out.shape == (2, 2, 2, 64)
        assert torch.isfinite(out).all()

    def test_band_mask_5d(self, s2_modality: ModalitySpec) -> None:
        """Band mask works with 5D (temporal) input."""
        embed = ChannelAttentionPatchEmbed(
            modality_spec=s2_modality,
            base_patch_size_at_16=8,
            num_bands=12,
            embedding_size=64,
            attn_dim=128,
            num_heads=4,
        )
        p = s2_modality.image_tile_size_factor * 8
        H = W = p * 2
        x = torch.randn(2, H, W, 3, 12)
        band_mask = torch.zeros(2, 12, dtype=torch.bool)
        band_mask[0, :4] = True
        out = embed(x, band_mask=band_mask)
        assert out.shape == (2, 2, 2, 3, 64)
        assert torch.isfinite(out).all()

    def test_no_mask_vs_all_false_mask_equivalent(
        self, s2_modality: ModalitySpec
    ) -> None:
        """No mask and all-False mask produce identical outputs."""
        embed = ChannelAttentionPatchEmbed(
            modality_spec=s2_modality,
            base_patch_size_at_16=8,
            num_bands=4,
            embedding_size=32,
            attn_dim=64,
            num_heads=4,
        )
        p = s2_modality.image_tile_size_factor * 8
        H = W = p * 2
        x = torch.randn(1, H, W, 4)
        out_no_mask = embed(x)
        band_mask = torch.zeros(1, 4, dtype=torch.bool)
        out_false_mask = embed(x, band_mask=band_mask)
        assert torch.allclose(out_no_mask, out_false_mask, atol=1e-5)

    def test_gradient_flows(self, s2_modality: ModalitySpec) -> None:
        """Gradients flow through all parameters."""
        embed = ChannelAttentionPatchEmbed(
            modality_spec=s2_modality,
            base_patch_size_at_16=8,
            num_bands=4,
            embedding_size=32,
            attn_dim=64,
            num_heads=4,
        )
        p = s2_modality.image_tile_size_factor * 8
        H = W = p * 2
        x = torch.randn(2, H, W, 4)
        band_mask = torch.zeros(2, 4, dtype=torch.bool)
        band_mask[0, 0] = True
        out = embed(x, band_mask=band_mask)
        out.sum().backward()
        for name, param in embed.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"


class TestMultiModalPatchEmbeddingsWithChannelAttn:
    """Tests for MultiModalPatchEmbeddings with channel attention dispatch."""

    def test_dispatch_creates_channel_attn_for_multi_band(
        self, s2_single_bandset_config: TokenizationConfig
    ) -> None:
        """ChannelAttentionPatchEmbed is used for bandsets with >1 band."""
        patch_emb = MultiModalPatchEmbeddings(
            supported_modality_names=["sentinel2_l2a", "sentinel1", "worldcover"],
            max_patch_size=8,
            embedding_size=64,
            tokenization_config=s2_single_bandset_config,
            channel_attn_dim=128,
            channel_attn_num_heads=4,
        )
        s2_module = patch_emb.per_modality_embeddings["sentinel2_l2a"][
            "sentinel2_l2a__0"
        ]
        assert isinstance(s2_module, ChannelAttentionPatchEmbed)

        s1_module = patch_emb.per_modality_embeddings["sentinel1"]["sentinel1__0"]
        assert isinstance(s1_module, ChannelAttentionPatchEmbed)

        wc_module = patch_emb.per_modality_embeddings["worldcover"]["worldcover__0"]
        assert isinstance(wc_module, FlexiPatchEmbed)

    def test_dispatch_without_channel_attn(
        self, s2_single_bandset_config: TokenizationConfig
    ) -> None:
        """Without channel_attn_dim, all spatial modalities use FlexiPatchEmbed."""
        patch_emb = MultiModalPatchEmbeddings(
            supported_modality_names=["sentinel2_l2a"],
            max_patch_size=8,
            embedding_size=64,
            tokenization_config=s2_single_bandset_config,
        )
        s2_module = patch_emb.per_modality_embeddings["sentinel2_l2a"][
            "sentinel2_l2a__0"
        ]
        assert isinstance(s2_module, FlexiPatchEmbed)

    def test_encoder_config_builds_with_channel_attn(
        self, s2_single_bandset_config: TokenizationConfig
    ) -> None:
        """EncoderConfig with channel_attn_dim builds successfully."""
        config = EncoderConfig(
            supported_modality_names=["sentinel2_l2a", "worldcover"],
            embedding_size=64,
            num_heads=4,
            depth=2,
            mlp_ratio=2.0,
            max_patch_size=8,
            max_sequence_length=12,
            tokenization_config=s2_single_bandset_config,
            channel_attn_dim=128,
            channel_attn_num_heads=4,
        )
        encoder = config.build()
        s2_module = encoder.patch_embeddings.per_modality_embeddings["sentinel2_l2a"][
            "sentinel2_l2a__0"
        ]
        assert isinstance(s2_module, ChannelAttentionPatchEmbed)
        assert s2_module.attn_dim == 128

    def test_encoder_config_validates_attn_dim_heads(self) -> None:
        """Validation catches channel_attn_dim not divisible by num_heads."""
        config = EncoderConfig(
            supported_modality_names=["sentinel2_l2a"],
            embedding_size=64,
            num_heads=4,
            depth=2,
            mlp_ratio=2.0,
            max_patch_size=8,
            max_sequence_length=12,
            channel_attn_dim=100,
            channel_attn_num_heads=8,
        )
        with pytest.raises(ValueError, match="divisible"):
            config.build()

"""Unit tests for the flexi_patch_embed module."""

import pytest
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.nn.flexi_patch_embed import FlexiPatchEmbed


class TestFlexiPatchEmbedPseudoinverse:
    """Tests for FlexiPatchEmbed with pseudoinverse resizing."""

    @pytest.fixture
    def patch_embed_input_resize(self) -> FlexiPatchEmbed:
        """Create FlexiPatchEmbed with input resizing (default)."""
        return FlexiPatchEmbed(
            modality_spec=Modality.SENTINEL2_L2A,
            patch_size_at_16=8,
            in_chans=12,
            embedding_size=128,
            use_pseudoinverse=False,
        )

    @pytest.fixture
    def patch_embed_pseudoinverse(self) -> FlexiPatchEmbed:
        """Create FlexiPatchEmbed with pseudoinverse resizing."""
        return FlexiPatchEmbed(
            modality_spec=Modality.SENTINEL2_L2A,
            patch_size_at_16=8,
            in_chans=12,
            embedding_size=128,
            use_pseudoinverse=True,
        )

    def test_pinvs_cached_at_init(
        self, patch_embed_pseudoinverse: FlexiPatchEmbed
    ) -> None:
        """Pinv matrices should be precomputed at init time."""
        # Should have pinvs for patch sizes 1-7 (not 8, since that's the base)
        assert len(patch_embed_pseudoinverse.pinvs) > 0
        # Check that base patch size is not in pinvs (no resize needed)
        assert (
            patch_embed_pseudoinverse.patch_size not in patch_embed_pseudoinverse.pinvs
        )

    def test_no_pinvs_when_disabled(
        self, patch_embed_input_resize: FlexiPatchEmbed
    ) -> None:
        """No pinv matrices should be computed when pseudoinverse is disabled."""
        assert len(patch_embed_input_resize.pinvs) == 0

    def test_base_patch_size_identical(
        self,
        patch_embed_input_resize: FlexiPatchEmbed,
        patch_embed_pseudoinverse: FlexiPatchEmbed,
    ) -> None:
        """Both methods should give identical results at base patch size."""
        # Copy weights so they're identical
        patch_embed_pseudoinverse.proj.weight.data = (
            patch_embed_input_resize.proj.weight.data.clone()
        )
        patch_embed_pseudoinverse.proj.bias.data = (
            patch_embed_input_resize.proj.bias.data.clone()
        )

        x = torch.randn(2, 32, 32, 12)  # [B, H, W, C]

        out_input = patch_embed_input_resize(x, patch_size=8)
        out_pinv = patch_embed_pseudoinverse(x, patch_size=8)

        assert torch.allclose(out_input, out_pinv, atol=1e-5)

    def test_pseudoinverse_output_shape(
        self, patch_embed_pseudoinverse: FlexiPatchEmbed
    ) -> None:
        """Pseudoinverse should produce correct output shape for different patch sizes."""
        x = torch.randn(2, 32, 32, 12)  # [B, H, W, C]

        # Test with different patch sizes
        for patch_size in [4, 8]:
            out = patch_embed_pseudoinverse(x, patch_size=patch_size)
            expected_h_w = 32 // patch_size
            assert out.shape == (2, expected_h_w, expected_h_w, 128)

    def test_pseudoinverse_gradient_flow(
        self, patch_embed_pseudoinverse: FlexiPatchEmbed
    ) -> None:
        """Gradients should flow through pseudoinverse forward pass."""
        x = torch.randn(2, 32, 32, 12, requires_grad=True)

        out = patch_embed_pseudoinverse(x, patch_size=4)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert patch_embed_pseudoinverse.proj.weight.grad is not None

    def test_pseudoinverse_with_time_dimension(
        self, patch_embed_pseudoinverse: FlexiPatchEmbed
    ) -> None:
        """Pseudoinverse should work with time dimension."""
        x = torch.randn(2, 32, 32, 3, 12)  # [B, H, W, T, C]

        out = patch_embed_pseudoinverse(x, patch_size=4)
        expected_h_w = 32 // 4
        assert out.shape == (2, expected_h_w, expected_h_w, 3, 128)

    def test_different_patch_sizes_different_outputs(
        self, patch_embed_pseudoinverse: FlexiPatchEmbed
    ) -> None:
        """Different patch sizes should produce different output shapes."""
        x = torch.randn(2, 32, 32, 12)

        out_4 = patch_embed_pseudoinverse(x, patch_size=4)
        out_8 = patch_embed_pseudoinverse(x, patch_size=8)

        # Shapes should differ
        assert out_4.shape != out_8.shape
        assert out_4.shape == (2, 8, 8, 128)
        assert out_8.shape == (2, 4, 4, 128)

    def test_pinv_computed_on_demand_for_new_size(
        self, patch_embed_pseudoinverse: FlexiPatchEmbed
    ) -> None:
        """Pinv should be computed on demand if patch size not in sequence."""
        # Create with limited patch_size_seq
        embed = FlexiPatchEmbed(
            modality_spec=Modality.SENTINEL2_L2A,
            patch_size_at_16=8,
            in_chans=12,
            embedding_size=128,
            use_pseudoinverse=True,
            patch_size_seq=(2, 4),  # Only precompute for 2 and 4
        )

        x = torch.randn(2, 48, 48, 12)

        # patch_size=6 not in sequence, should be computed on demand
        actual_ps_6 = 6 * Modality.SENTINEL2_L2A.image_tile_size_factor
        assert (actual_ps_6, actual_ps_6) not in embed.pinvs

        out = embed(x, patch_size=6)
        assert out.shape[1] == 48 // 6  # 8

        # Now it should be cached
        assert (actual_ps_6, actual_ps_6) in embed.pinvs

    def test_resize_patch_embed_identity_for_same_size(
        self, patch_embed_pseudoinverse: FlexiPatchEmbed
    ) -> None:
        """resize_patch_embed should return original weights for same size."""
        weight = patch_embed_pseudoinverse.proj.weight
        resized = patch_embed_pseudoinverse.resize_patch_embed(
            weight, patch_embed_pseudoinverse.patch_size
        )
        assert torch.equal(weight, resized)

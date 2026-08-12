"""Unit tests for the register unit-sphere output and the uniformity loss."""

import math

import torch

from olmoearth_pretrain.nn.flexi_vit import SpatialRegisterBottleneck
from olmoearth_pretrain.train.train_module.latent_mim import (
    compute_register_uniformity_loss,
)


class TestRegisterUniformityLoss:
    """Tests for the cross-scene uniformity term."""

    def test_collapsed_scenes_score_worse_than_spread(self) -> None:
        """Every scene sharing one direction is the case the term exists to punish."""
        collapsed = torch.randn(1, 1, 32).expand(8, 4, 32).contiguous()
        spread = torch.randn(8, 4, 32)
        collapsed_loss, _ = compute_register_uniformity_loss(collapsed, weight=1.0)
        spread_loss, _ = compute_register_uniformity_loss(spread, weight=1.0)
        assert collapsed_loss > 0.9
        assert spread_loss < collapsed_loss

    def test_orthogonal_scenes_are_optimal(self) -> None:
        """Mutually orthogonal scenes drive the term to ~0."""
        d = 16
        scenes = torch.eye(d)[:8].reshape(8, 1, d).expand(8, 4, d).contiguous()
        loss, metrics = compute_register_uniformity_loss(scenes, weight=1.0)
        assert loss.abs() < 1e-5
        assert metrics["register/uniformity"].abs() < 1e-5

    def test_ignores_within_scene_structure(self) -> None:
        """Cells within a scene may be identical without being penalized.

        Neighbouring cells of a homogeneous field genuinely are alike, and the
        dense probes read that smoothness -- only cross-scene pairs are scored.
        """
        d = 16
        # Each scene is internally constant; scenes are mutually orthogonal.
        scenes = torch.eye(d)[:8].reshape(8, 1, d).expand(8, 6, d).contiguous()
        loss, _ = compute_register_uniformity_loss(scenes, weight=1.0)
        assert loss.abs() < 1e-5

    def test_sign_blind(self) -> None:
        """Antipodal scenes are as bad as identical ones (the |.| in AEF's term)."""
        base = torch.randn(1, 3, 24)
        antipodal = torch.cat([base, -base], dim=0)
        identical = torch.cat([base, base], dim=0)
        anti_loss, _ = compute_register_uniformity_loss(antipodal, weight=1.0)
        same_loss, _ = compute_register_uniformity_loss(identical, weight=1.0)
        assert torch.allclose(anti_loss, same_loss, atol=1e-5)

    def test_weight_scales_the_term(self) -> None:
        """The returned loss is the weighted metric."""
        x = torch.randn(6, 3, 16)
        loss, metrics = compute_register_uniformity_loss(x, weight=0.25)
        assert torch.allclose(loss, 0.25 * metrics["register/uniformity"])

    def test_single_scene_is_skipped(self) -> None:
        """A batch of one has no cross-scene pair, so the term contributes nothing."""
        loss, metrics = compute_register_uniformity_loss(torch.randn(1, 8, 16), 1.0)
        assert loss.item() == 0.0
        assert metrics == {}

    def test_gradient_reaches_the_input(self) -> None:
        """Unlike the distillation terms this one is not detached."""
        x = torch.randn(4, 2, 16, requires_grad=True)
        loss, _ = compute_register_uniformity_loss(x, weight=1.0)
        loss.backward()
        assert x.grad is not None and x.grad.abs().sum() > 0


class TestRegisterUnitNorm:
    """Tests for the bottleneck's unit-sphere output."""

    def _bottleneck(
        self,
        register_dim: int,
        unit_norm: bool = False,
        unit_norm_scale: float | None = None,
    ) -> "SpatialRegisterBottleneck":
        return SpatialRegisterBottleneck(
            encoder_embedding_size=32,
            register_dim=register_dim,
            register_grid=(2, 2),
            num_heads=2,
            mlp_ratio=2.0,
            read_depth=1,
            latent_transformer_depth=1,
            use_2d_rope=False,
            unit_norm=unit_norm,
            unit_norm_scale=unit_norm_scale,
        )

    def test_off_by_default(self) -> None:
        """Existing configs are untouched."""
        assert self._bottleneck(16).unit_norm is False

    def test_default_scale_is_layernorm_norm(self) -> None:
        """sqrt(register_dim) is LayerNorm's own output norm, so init is unchanged."""
        bottleneck = self._bottleneck(64, unit_norm=True)
        assert math.isclose(bottleneck.unit_norm_scale, math.sqrt(64))

    def test_explicit_scale_is_respected(self) -> None:
        """The radius can be pinned to hold a measured operating point."""
        bottleneck = self._bottleneck(64, unit_norm=True, unit_norm_scale=5.7)
        assert math.isclose(bottleneck.unit_norm_scale, 5.7)

    def test_forward_output_lies_on_the_sphere(self) -> None:
        """Every cell comes out at the configured radius, for every consumer."""
        torch.manual_seed(0)
        bottleneck = self._bottleneck(16, unit_norm=True)
        patch_tokens = torch.randn(3, 5, 32)
        out, _ = bottleneck(
            patch_tokens=patch_tokens, patch_positions=None, visible_mask=None
        )
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.full_like(norms, math.sqrt(16)), atol=1e-4)

    def test_forward_unnormalized_by_default(self) -> None:
        """Without the flag the norms vary, as they do today."""
        torch.manual_seed(0)
        bottleneck = self._bottleneck(16)
        out, _ = bottleneck(
            patch_tokens=torch.randn(3, 5, 32),
            patch_positions=None,
            visible_mask=None,
        )
        assert out.norm(dim=-1).std() > 1e-6

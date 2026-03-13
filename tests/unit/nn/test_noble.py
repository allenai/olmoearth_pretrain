"""Unit tests for the NOBLE module."""

import pytest
import torch
import torch.nn as nn

from olmoearth_pretrain.nn.noble import (
    CosNet,
    GELUActivation,
    NobleBranch,
    NobleConfig,
    NobleLinear,
    get_noble_config,
)


class TestCosNet:
    """Unit tests for the CosNet activation."""

    def test_forward_shape(self) -> None:
        """Test that CosNet preserves input shape."""
        hidden_dim = 32
        cosnet = CosNet(hidden_dim)
        x = torch.randn(4, 16, hidden_dim)
        y = cosnet(x)
        assert y.shape == x.shape

    def test_forward_bounded(self) -> None:
        """Test that CosNet output is bounded by [-1, 1] (due to cos)."""
        hidden_dim = 32
        cosnet = CosNet(hidden_dim)
        x = torch.randn(4, 16, hidden_dim)
        y = cosnet(x)
        assert y.min() >= -1.0
        assert y.max() <= 1.0

    def test_learnable_params(self) -> None:
        """Test that CosNet has learnable parameters."""
        hidden_dim = 32
        cosnet = CosNet(hidden_dim)
        params = list(cosnet.parameters())
        assert len(params) == 5  # w1, b1, w_mid.weight, w2, b2

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through CosNet."""
        hidden_dim = 32
        cosnet = CosNet(hidden_dim)
        x = torch.randn(4, 16, hidden_dim, requires_grad=True)
        y = cosnet(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)


class TestNobleBranch:
    """Unit tests for the NobleBranch."""

    def test_forward_shape(self) -> None:
        """Test NobleBranch output shape."""
        in_features, out_features, rank = 64, 128, 16
        branch = NobleBranch(in_features, out_features, rank)
        x = torch.randn(4, 16, in_features)
        y = branch(x)
        assert y.shape == (4, 16, out_features)

    def test_cosnet_activation(self) -> None:
        """Test NobleBranch with CosNet activation."""
        in_features, out_features, rank = 64, 128, 16
        branch = NobleBranch(in_features, out_features, rank, activation="cosnet")
        assert isinstance(branch.activation, CosNet)

    def test_gelu_activation(self) -> None:
        """Test NobleBranch with GELU activation."""
        in_features, out_features, rank = 64, 128, 16
        branch = NobleBranch(in_features, out_features, rank, activation="gelu")
        assert isinstance(branch.activation, GELUActivation)

    def test_invalid_activation(self) -> None:
        """Test that invalid activation raises error."""
        with pytest.raises(ValueError, match="Unknown activation"):
            NobleBranch(64, 128, 16, activation="invalid")  # type: ignore[arg-type]

    def test_init_scale(self) -> None:
        """Test that W_up is initialized with small values."""
        in_features, out_features, rank = 64, 128, 16
        init_scale = 0.01
        branch = NobleBranch(in_features, out_features, rank, init_scale=init_scale)
        # W_up should have small values around init_scale
        w_up_std = branch.w_up.weight.std().item()
        assert w_up_std < init_scale * 5  # Allow some variance

    def test_gradient_flow(self) -> None:
        """Test gradient flow through NobleBranch."""
        in_features, out_features, rank = 64, 128, 16
        branch = NobleBranch(in_features, out_features, rank)
        x = torch.randn(4, 16, in_features, requires_grad=True)
        y = branch(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None


class TestNobleLinear:
    """Unit tests for NobleLinear."""

    def test_forward_shape(self) -> None:
        """Test NobleLinear output shape."""
        in_features, out_features = 64, 128
        noble_linear = NobleLinear(in_features, out_features)
        x = torch.randn(4, 16, in_features)
        y = noble_linear(x)
        assert y.shape == (4, 16, out_features)

    def test_adds_branch_output(self) -> None:
        """Test that NobleLinear adds branch output to linear output."""
        in_features, out_features = 64, 64
        noble_linear = NobleLinear(in_features, out_features, bias=False)
        x = torch.randn(4, 16, in_features)

        # Get individual outputs
        linear_out = noble_linear.linear(x)
        branch_out = noble_linear.branch(x)
        combined_out = noble_linear(x)

        # Combined should be sum of linear and branch
        assert torch.allclose(combined_out, linear_out + branch_out, atol=1e-6)

    def test_rank_ratio(self) -> None:
        """Test rank is computed from rank_ratio."""
        in_features, out_features = 64, 128
        rank_ratio = 0.25
        noble_linear = NobleLinear(in_features, out_features, rank_ratio=rank_ratio)
        expected_rank = int(min(in_features, out_features) * rank_ratio)
        assert noble_linear.branch.w_down.out_features == expected_rank

    def test_explicit_rank(self) -> None:
        """Test explicit rank overrides rank_ratio."""
        in_features, out_features = 64, 128
        rank = 8
        noble_linear = NobleLinear(in_features, out_features, rank=rank)
        assert noble_linear.branch.w_down.out_features == rank

    def test_with_bias(self) -> None:
        """Test NobleLinear with bias."""
        noble_linear = NobleLinear(64, 128, bias=True)
        assert noble_linear.linear.bias is not None

    def test_without_bias(self) -> None:
        """Test NobleLinear without bias."""
        noble_linear = NobleLinear(64, 128, bias=False)
        assert noble_linear.linear.bias is None


class TestNobleConfig:
    """Unit tests for NobleConfig."""

    def test_default_values(self) -> None:
        """Test default config values."""
        config = NobleConfig()
        assert config.enabled is True
        assert config.rank_ratio == 0.25
        assert config.activation == "cosnet"
        assert config.init_scale == 0.01
        assert config.apply_to_qkv is True
        assert config.apply_to_proj is True
        assert config.apply_to_mlp is True

    def test_validate_rank_ratio(self) -> None:
        """Test validation of rank_ratio."""
        with pytest.raises(ValueError, match="rank_ratio"):
            NobleConfig(rank_ratio=0).validate()
        with pytest.raises(ValueError, match="rank_ratio"):
            NobleConfig(rank_ratio=1.5).validate()

    def test_validate_activation(self) -> None:
        """Test validation of activation."""
        with pytest.raises(ValueError, match="activation"):
            NobleConfig(activation="invalid").validate()  # type: ignore[arg-type]

    def test_validate_init_scale(self) -> None:
        """Test validation of init_scale."""
        with pytest.raises(ValueError, match="init_scale"):
            NobleConfig(init_scale=-0.1).validate()

    def test_make_linear_enabled_qkv(self) -> None:
        """Test make_linear creates NobleLinear for qkv when enabled."""
        config = NobleConfig(enabled=True, apply_to_qkv=True)
        linear = config.make_linear(64, 128, layer_type="qkv")
        assert isinstance(linear, NobleLinear)

    def test_make_linear_disabled_qkv(self) -> None:
        """Test make_linear creates nn.Linear for qkv when disabled."""
        config = NobleConfig(enabled=True, apply_to_qkv=False)
        linear = config.make_linear(64, 128, layer_type="qkv")
        assert isinstance(linear, nn.Linear)
        assert not isinstance(linear, NobleLinear)

    def test_make_linear_disabled_entirely(self) -> None:
        """Test make_linear creates nn.Linear when NOBLE is disabled."""
        config = NobleConfig(enabled=False)
        linear = config.make_linear(64, 128, layer_type="qkv")
        assert isinstance(linear, nn.Linear)
        assert not isinstance(linear, NobleLinear)

    def test_make_linear_proj(self) -> None:
        """Test make_linear for proj layer type."""
        config = NobleConfig(enabled=True, apply_to_proj=True)
        linear = config.make_linear(64, 64, layer_type="proj")
        assert isinstance(linear, NobleLinear)

    def test_make_linear_mlp(self) -> None:
        """Test make_linear for mlp layer type."""
        config = NobleConfig(enabled=True, apply_to_mlp=True)
        linear = config.make_linear(64, 256, layer_type="mlp")
        assert isinstance(linear, NobleLinear)


class TestGetNobleConfig:
    """Unit tests for get_noble_config helper."""

    def test_returns_disabled_when_none(self) -> None:
        """Test that None input returns disabled config."""
        config = get_noble_config(None)
        assert isinstance(config, NobleConfig)
        assert config.enabled is False

    def test_returns_config_when_provided(self) -> None:
        """Test that provided config is returned as-is."""
        original = NobleConfig(enabled=True, rank_ratio=0.5)
        result = get_noble_config(original)
        assert result is original


class TestNobleIntegration:
    """Integration tests for NOBLE with attention components."""

    def test_attention_with_noble(self) -> None:
        """Test Attention module works with NOBLE config."""
        from olmoearth_pretrain.nn.attention import Attention

        noble_config = NobleConfig(enabled=True)
        attn = Attention(
            dim=64,
            num_heads=4,
            noble_config=noble_config,
        )

        x = torch.randn(2, 16, 64)
        y = attn(x)
        assert y.shape == x.shape

    def test_mlp_with_noble(self) -> None:
        """Test Mlp module works with NOBLE config."""
        from olmoearth_pretrain.nn.attention import Mlp

        noble_config = NobleConfig(enabled=True)
        mlp = Mlp(
            in_features=64,
            hidden_features=256,
            noble_config=noble_config,
        )

        x = torch.randn(2, 16, 64)
        y = mlp(x)
        assert y.shape == x.shape

    def test_block_with_noble(self) -> None:
        """Test Block works with NOBLE config."""
        from olmoearth_pretrain.nn.attention import Block

        noble_config = NobleConfig(enabled=True)
        block = Block(
            dim=64,
            num_heads=4,
            noble_config=noble_config,
        )

        x = torch.randn(2, 16, 64)
        y = block(x)
        assert y.shape == x.shape

    def test_noble_adds_parameters(self) -> None:
        """Test that NOBLE adds extra parameters to attention."""
        from olmoearth_pretrain.nn.attention import Attention

        attn_base = Attention(dim=64, num_heads=4, noble_config=None)
        attn_noble = Attention(
            dim=64, num_heads=4, noble_config=NobleConfig(enabled=True)
        )

        base_params = sum(p.numel() for p in attn_base.parameters())
        noble_params = sum(p.numel() for p in attn_noble.parameters())

        # NOBLE should add parameters
        assert noble_params > base_params

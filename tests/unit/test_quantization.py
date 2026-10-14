"""Unit tests for olmoearth_pretrain.quantization module."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from olmoearth_pretrain.quantization import (
    check_modelopt_available,
    count_quantizable_layers,
    count_quantizer_nodes,
    get_model_memory_mb,
    get_modelopt_install_instructions,
)

MODELOPT_AVAILABLE = check_modelopt_available()


class TestCheckModeloptAvailable:
    """Tests for modelopt availability detection."""

    def test_returns_bool(self) -> None:
        """check_modelopt_available returns a boolean."""
        result = check_modelopt_available()
        assert isinstance(result, bool)


class TestGetModeloptInstallInstructions:
    """Tests for install instructions string."""

    def test_returns_nonempty_string(self) -> None:
        """Instructions string is non-empty and mentions nvidia-modelopt."""
        instructions = get_modelopt_install_instructions()
        assert isinstance(instructions, str)
        assert len(instructions) > 0
        assert "nvidia-modelopt" in instructions


class TestCountQuantizableLayers:
    """Tests for counting quantizable layers in a model."""

    def test_empty_model(self) -> None:
        """Model with no quantizable layers returns empty dict."""
        model = nn.Module()
        counts = count_quantizable_layers(model)
        assert counts == {}

    def test_linear_layers(self) -> None:
        """Counts nn.Linear layers correctly."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
        counts = count_quantizable_layers(model)
        assert counts["nn.Linear"] == 2
        assert "nn.Conv2d" not in counts

    def test_conv2d_layers(self) -> None:
        """Counts nn.Conv2d layers correctly."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.Conv2d(16, 32, 3),
        )
        counts = count_quantizable_layers(model)
        assert counts["nn.Conv2d"] == 2

    def test_layernorm_marked_as_skipped(self) -> None:
        """LayerNorm layers are counted but marked as skipped."""
        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.LayerNorm(10),
        )
        counts = count_quantizable_layers(model)
        assert counts["nn.Linear"] == 1
        assert counts["nn.LayerNorm (skipped)"] == 1

    def test_mixed_model(self) -> None:
        """Model with Linear, Conv2d, and LayerNorm."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.Linear(10, 20),
            nn.LayerNorm(20),
            nn.Linear(20, 5),
        )
        counts = count_quantizable_layers(model)
        assert counts["nn.Linear"] == 2
        assert counts["nn.Conv2d"] == 1
        assert counts["nn.LayerNorm (skipped)"] == 1


class TestGetModelMemoryMb:
    """Tests for model memory footprint estimation."""

    def test_known_size(self) -> None:
        """Linear(10, 10) has 10*10 + 10 = 110 float32 params = 440 bytes."""
        model = nn.Linear(10, 10)
        memory = get_model_memory_mb(model)
        expected = (10 * 10 + 10) * 4 / (1024 * 1024)  # 110 float32 params
        assert abs(memory - expected) < 1e-6

    def test_returns_positive_float(self) -> None:
        """Memory is a positive float for any non-empty model."""
        model = nn.Linear(100, 100)
        memory = get_model_memory_mb(model)
        assert isinstance(memory, float)
        assert memory > 0

    def test_empty_model_returns_zero(self) -> None:
        """Model with no parameters returns 0."""
        model = nn.Module()
        memory = get_model_memory_mb(model)
        assert memory == 0.0


class TestCountQuantizerNodes:
    """Tests for counting quantizer nodes in a model."""

    def test_unquantized_model_has_zero(self) -> None:
        """Regular model has no quantizer nodes."""
        model = nn.Sequential(nn.Linear(10, 10), nn.ReLU())
        count = count_quantizer_nodes(model)
        assert count == 0

    def test_detects_quantizer_by_classname(self) -> None:
        """Modules with 'Quantizer' in class name are counted."""

        class FakeTensorQuantizer(nn.Module):
            pass

        model = nn.Sequential(
            nn.Linear(10, 10),
            FakeTensorQuantizer(),
            FakeTensorQuantizer(),
        )
        count = count_quantizer_nodes(model)
        assert count == 2


@pytest.mark.skipif(not MODELOPT_AVAILABLE, reason="nvidia-modelopt not installed")
class TestGetQuantConfig:
    """Tests for _get_quant_config (requires modelopt)."""

    def test_fp4_default(self) -> None:
        """FP4 default config returns a dict."""
        from olmoearth_pretrain.quantization import _get_quant_config

        cfg = _get_quant_config("fp4", "default")
        assert isinstance(cfg, dict)

    def test_fp8_default(self) -> None:
        """FP8 default config returns a dict."""
        from olmoearth_pretrain.quantization import _get_quant_config

        cfg = _get_quant_config("fp8", "default")
        assert isinstance(cfg, dict)

    def test_invalid_precision_raises(self) -> None:
        """Unknown precision raises ValueError."""
        from olmoearth_pretrain.quantization import _get_quant_config

        with pytest.raises(ValueError, match="Unknown precision"):
            _get_quant_config("int4", "default")

    def test_invalid_config_raises(self) -> None:
        """Unknown config raises ValueError."""
        from olmoearth_pretrain.quantization import _get_quant_config

        with pytest.raises(ValueError, match="Unknown config"):
            _get_quant_config("fp8", "nonexistent")

    def test_mlp_only_config(self) -> None:
        """MLP-only config disables attention quantizers."""
        from olmoearth_pretrain.quantization import _get_quant_config

        cfg = _get_quant_config("fp8", "mlp_only")
        assert isinstance(cfg, dict)

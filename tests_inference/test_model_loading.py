"""Tests for model loading with inference-only dependencies (no olmo-core).

These tests verify that models can be loaded from config.json files
without requiring olmo-core as a dependency.

This test directory is separate from tests/ because the main conftest.py
imports modules that depend on olmo-core. These tests run in CI with
only base dependencies installed.

To run locally without olmo-core:
    uv run --group dev pytest -v tests_inference/
"""

import pytest
import torch

from olmoearth_pretrain.config import (
    OLMO_CORE_AVAILABLE,
    Config,
    _StandaloneConfig,
)
from olmoearth_pretrain.nn.flexi_vit import EncoderConfig


class TestConfigExport:
    """Tests for the exported Config class behavior."""

    def test_config_is_standalone_when_olmo_core_unavailable(self) -> None:
        """Test that Config is _StandaloneConfig when olmo-core is not installed."""
        if not OLMO_CORE_AVAILABLE:
            assert Config is _StandaloneConfig
        else:
            # When olmo-core is available, Config should be olmo-core's Config
            from olmo_core.config import Config as OlmoCoreConfig

            assert Config is OlmoCoreConfig

    def test_config_has_from_dict_method(self) -> None:
        """Test that the exported Config has from_dict for model loading."""
        assert hasattr(Config, "from_dict")


class TestEncoderConfigLoading:
    """Tests for loading EncoderConfig using the exported Config."""

    @pytest.fixture
    def encoder_config_dict(self) -> dict:
        """Create a minimal EncoderConfig as a dict."""
        return {
            "_CLASS_": "olmoearth_pretrain.nn.flexi_vit.EncoderConfig",
            "supported_modality_names": ["sentinel2_l2a"],
            "embedding_size": 64,
            "num_heads": 2,
            "depth": 2,
            "mlp_ratio": 4.0,
            "max_sequence_length": 64,
        }

    def test_load_encoder_config_from_dict(self, encoder_config_dict: dict) -> None:
        """Test loading EncoderConfig from a dict."""
        config = EncoderConfig.from_dict(encoder_config_dict)

        assert isinstance(config, EncoderConfig)
        assert config.embedding_size == 64
        assert config.num_heads == 2
        assert config.depth == 2
        assert config.supported_modality_names == ["sentinel2_l2a"]

    def test_build_encoder_from_config(self, encoder_config_dict: dict) -> None:
        """Test building an Encoder from the loaded config."""
        config = EncoderConfig.from_dict(encoder_config_dict)
        encoder = config.build()

        assert encoder is not None
        assert isinstance(encoder, torch.nn.Module)

        # Verify parameters exist
        params = list(encoder.parameters())
        assert len(params) > 0


class TestOlmoCoreAvailabilityFlag:
    """Tests for the OLMO_CORE_AVAILABLE flag."""

    def test_flag_reflects_installation(self) -> None:
        """Test that OLMO_CORE_AVAILABLE correctly reflects olmo-core installation."""
        try:
            import olmo_core  # noqa: F401

            expected = True
        except ImportError:
            expected = False

        assert OLMO_CORE_AVAILABLE == expected

    @pytest.mark.skipif(
        OLMO_CORE_AVAILABLE, reason="Only run when olmo-core is NOT installed"
    )
    def test_standalone_mode_uses_standalone_config(self) -> None:
        """Test that Config is _StandaloneConfig when running in standalone mode."""
        assert Config is _StandaloneConfig

    @pytest.mark.skipif(
        not OLMO_CORE_AVAILABLE, reason="Only run when olmo-core IS installed"
    )
    def test_full_mode_uses_olmo_core(self) -> None:
        """Test that Config is olmo-core's Config when available."""
        from olmo_core.config import Config as OlmoCoreConfig

        assert Config is OlmoCoreConfig

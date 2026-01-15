"""Tests for model loading that run with both minimal and full dependencies.

This directory (tests_minimal_deps/) contains tests that are run twice in CI:
1. With minimal deps only (no olmo-core) → tests _StandaloneConfig path
2. With full deps (with olmo-core) → tests olmo-core Config path

This verifies model loading works regardless of whether olmo-core is installed.

To run locally:
    # Minimal deps (no olmo-core)
    uv run --group dev pytest -v tests_minimal_deps/

    # Full deps (with olmo-core)
    uv run pytest -v tests_minimal_deps/
"""

import json
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from olmoearth_pretrain.config import (
    OLMO_CORE_AVAILABLE,
    Config,
    _StandaloneConfig,
)
from olmoearth_pretrain.model_loader import (
    CONFIG_FILENAME,
    WEIGHTS_FILENAME,
    ModelID,
    load_model_from_id,
    load_model_from_path,
)
from olmoearth_pretrain.nn.flexi_vit import EncoderConfig, PredictorConfig
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

# =============================================================================
# Test Helpers
# =============================================================================


def _create_minimal_model_config() -> dict:
    """Create a minimal model config that can be built."""
    encoder_config = EncoderConfig(
        supported_modality_names=["sentinel2_l2a", "sentinel1"],
        embedding_size=16,
        max_patch_size=8,
        num_heads=2,
        depth=2,
        mlp_ratio=4.0,
        drop_path=0.1,
        max_sequence_length=12,
    )
    decoder_config = PredictorConfig(
        encoder_embedding_size=16,
        decoder_embedding_size=16,
        depth=2,
        mlp_ratio=4.0,
        num_heads=8,
        max_sequence_length=12,
        supported_modality_names=["sentinel2_l2a", "sentinel1"],
    )
    model_config = LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
    )
    # Return the structure expected by model_loader: {"model": <config_dict>}
    return {"model": model_config.as_config_dict()}


def _create_minimal_state_dict() -> dict[str, torch.Tensor]:
    """Create a minimal state dict for testing."""
    return {"dummy_weight": torch.randn(2, 2)}


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def encoder_config_dict() -> dict:
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


@pytest.fixture
def temp_model_dir() -> Generator[Path, None, None]:
    """Create a temporary directory with model artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Write config.json
        config_path = tmpdir_path / CONFIG_FILENAME
        with open(config_path, "w") as f:
            json.dump(_create_minimal_model_config(), f)

        # Write weights.pth
        weights_path = tmpdir_path / WEIGHTS_FILENAME
        torch.save(_create_minimal_state_dict(), weights_path)

        yield tmpdir_path


# =============================================================================
# Config Export Tests
# =============================================================================


class TestConfigExport:
    """Tests for the exported Config class behavior."""

    def test_config_is_correct_type(self) -> None:
        """Test that Config is the correct type based on olmo-core availability."""
        if not OLMO_CORE_AVAILABLE:
            assert Config is _StandaloneConfig
        else:
            from olmo_core.config import Config as OlmoCoreConfig

            assert Config is OlmoCoreConfig

    def test_config_has_from_dict_method(self) -> None:
        """Test that the exported Config has from_dict for model loading."""
        assert hasattr(Config, "from_dict")


# =============================================================================
# EncoderConfig Loading Tests
# =============================================================================


class TestEncoderConfigLoading:
    """Tests for loading EncoderConfig using the exported Config."""

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
        assert len(list(encoder.parameters())) > 0


# =============================================================================
# Model Loading Tests
# =============================================================================


class TestLoadModelFromPath:
    """Tests for load_model_from_path."""

    def test_load_with_pathlib_path(self, temp_model_dir: Path) -> None:
        """Test loading model using pathlib.Path."""
        model = load_model_from_path(temp_model_dir, load_weights=False)
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_load_with_string(self, temp_model_dir: Path) -> None:
        """Test loading model using string path."""
        model = load_model_from_path(str(temp_model_dir), load_weights=False)
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_load_without_weights(self, temp_model_dir: Path) -> None:
        """Test loading model without weights (random init)."""
        model = load_model_from_path(temp_model_dir, load_weights=False)
        assert model is not None
        assert isinstance(model, torch.nn.Module)


class TestLoadModelFromId:
    """Tests for load_model_from_id with mocked HuggingFace downloads."""

    def test_load_from_model_id_without_weights(self, temp_model_dir: Path) -> None:
        """Test loading model from ModelID without weights."""

        def mock_hf_hub_download(repo_id: str, filename: str) -> str:
            return str(temp_model_dir / filename)

        with patch(
            "olmoearth_pretrain.model_loader.hf_hub_download",
            side_effect=mock_hf_hub_download,
        ):
            model = load_model_from_id(ModelID.OLMOEARTH_V1_NANO, load_weights=False)
            assert model is not None
            assert isinstance(model, torch.nn.Module)

    def test_model_id_repo_id(self) -> None:
        """Test that ModelID.repo_id() returns correct format."""
        assert ModelID.OLMOEARTH_V1_NANO.repo_id() == "allenai/OlmoEarth-v1-Nano"
        assert ModelID.OLMOEARTH_V1_TINY.repo_id() == "allenai/OlmoEarth-v1-Tiny"
        assert ModelID.OLMOEARTH_V1_BASE.repo_id() == "allenai/OlmoEarth-v1-Base"
        assert ModelID.OLMOEARTH_V1_LARGE.repo_id() == "allenai/OlmoEarth-v1-Large"

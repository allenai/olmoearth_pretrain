"""Unit tests for model_loader module."""

import json
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from upath import UPath

from olmoearth_pretrain.model_loader import (
    CONFIG_FILENAME,
    WEIGHTS_FILENAME,
    ModelID,
    load_model_from_id,
    load_model_from_path,
)


def _create_minimal_model_config() -> dict:
    """Create a minimal model config that can be built."""
    from olmoearth_pretrain.nn.flexi_vit import EncoderConfig, PredictorConfig
    from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

    encoder_config = EncoderConfig(
        supported_modality_names=["sentinel2", "sentinel1"],
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
        supported_modality_names=["sentinel2", "sentinel1"],
    )
    model_config = LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
    )
    # Return the structure expected by model_loader: {"model": <config_dict>}
    # Use as_config_dict() to include __class__ fields needed for deserialization
    return {"model": model_config.as_config_dict()}


def _create_minimal_state_dict() -> dict[str, torch.Tensor]:
    """Create a minimal state dict for testing."""
    # Just a small tensor to represent model weights
    return {"dummy_weight": torch.randn(2, 2)}


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


class TestLoadModelFromPath:
    """Tests for load_model_from_path with different PathLike types."""

    def test_load_with_pathlib_path(self, temp_model_dir: Path) -> None:
        """Test loading model using pathlib.Path."""
        # Test without weights to focus on PathLike handling
        model = load_model_from_path(temp_model_dir, load_weights=False)
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_load_with_upath(self, temp_model_dir: Path) -> None:
        """Test loading model using UPath."""
        upath = UPath(temp_model_dir)
        model = load_model_from_path(upath, load_weights=False)
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
            """Mock HF hub download to return local temp files."""
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

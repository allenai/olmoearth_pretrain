"""Unit tests for the st_model module."""

import logging

import pytest
import torch

from helios.data.constants import Modality
from helios.nn.flexihelios import get_modalities_to_process, return_modalities_from_dict
from helios.nn.st_model import STBase

logger = logging.getLogger(__name__)


class TestSTBase:
    """Unit tests for the STBase class."""

    @pytest.fixture
    def st_base(self) -> STBase:
        """Create encoder fixture for testing."""
        return STBase(
            embedding_size=8,
            num_heads=2,
            mlp_ratio=4.0,
            depth=2,
            drop_path=0.1,
            supported_modalities=[
                Modality.SENTINEL2_L2A,
                Modality.WORLDCOVER,
                Modality.LATLON,
            ],
            max_sequence_length=12,
        )

    def test_collapse_and_combine_full(
        self,
        st_base: STBase,
    ) -> None:
        """Test collapsing tokens from different modalities into single tensor."""
        B, D = 2, 4
        # (b, h, w, t, b_s, d)
        sentinel2_l2a_tokens = torch.randn(B, 2, 1, 1, 2, D)
        sentinel2_l2a_mask = torch.randint(0, 2, (B, 2, 1, 1, 2)).float()
        latlon = torch.randn(B, 1, D)
        latlon_mask = torch.randint(0, 2, (B, 1)).float()
        x = {
            "sentinel2_l2a": sentinel2_l2a_tokens,
            "sentinel2_l2a_mask": sentinel2_l2a_mask,
            "latlon": latlon,
            "latlon_mask": latlon_mask,
        }
        tokens, masks = st_base.collapse_and_combine_full(x)
        assert tokens.shape == (B, 5, D)
        assert masks.shape == (B, 5)

    def test_collapse_and_split_spatial(
        self,
        st_base: STBase,
    ) -> None:
        """Test collapsing and re-splitting tokens for spatial attention."""
        B, H, W, T, B_S, D = 2, 3, 3, 4, 2, 4
        sentinel2_l2a_tokens = torch.randn(B, H, W, T, B_S, D)
        sentinel2_l2a_mask = torch.randint(0, 2, (B, H, W, T, B_S)).float()
        worldcover_tokens = torch.randn(B, H, W, B_S, D)
        worldcover_mask = torch.randint(0, 2, (B, H, W, B_S)).float()
        latlon = torch.randn(B, 1, D)
        latlon_mask = torch.randint(0, 2, (B, 1)).float()
        x = {
            "sentinel2_l2a": sentinel2_l2a_tokens,
            "sentinel2_l2a_mask": sentinel2_l2a_mask,
            "worldcover": worldcover_tokens,
            "worldcover_mask": worldcover_mask,
            "latlon": latlon,
            "latlon_mask": latlon_mask,
        }
        modalities_to_process = get_modalities_to_process(
            return_modalities_from_dict(x), st_base.supported_modality_names
        )
        modalities_to_dims_dict = {
            modality: x[modality].shape for modality in modalities_to_process
        }
        tokens, masks = st_base.collapse_and_combine_spatial(x)
        assert tokens.shape == (B * (T * B_S + B_S + 1), H * W, D)
        assert masks.shape == (B * (T * B_S + B_S + 1), H * W)

        modality_tokens_dict = st_base.split_and_expand_per_modality_spatial(
            tokens, modalities_to_dims_dict
        )
        assert (modality_tokens_dict["sentinel2_l2a"] == x["sentinel2_l2a"]).all()
        assert (modality_tokens_dict["worldcover"] == x["worldcover"]).all()
        assert (modality_tokens_dict["latlon"] == x["latlon"]).all()

    def test_collapse_and_split_temporal(
        self,
        st_base: STBase,
    ) -> None:
        """Test collapsing and re-splitting tokens for temporal attention."""
        B, H, W, T, B_S, D = 2, 3, 3, 4, 2, 4
        sentinel2_l2a_tokens = torch.randn(B, H, W, T, B_S, D)
        sentinel2_l2a_mask = torch.randint(0, 2, (B, H, W, T, B_S)).float()
        worldcover_tokens = torch.randn(B, H, W, B_S, D)
        worldcover_mask = torch.randint(0, 2, (B, H, W, B_S)).float()
        latlon = torch.randn(B, 1, D)
        latlon_mask = torch.randint(0, 2, (B, 1)).float()
        x = {
            "sentinel2_l2a": sentinel2_l2a_tokens,
            "sentinel2_l2a_mask": sentinel2_l2a_mask,
            "worldcover": worldcover_tokens,
            "worldcover_mask": worldcover_mask,
            "latlon": latlon,
            "latlon_mask": latlon_mask,
        }
        modalities_to_process = get_modalities_to_process(
            return_modalities_from_dict(x), st_base.supported_modality_names
        )
        modalities_to_dims_dict = {
            modality: x[modality].shape for modality in modalities_to_process
        }
        tokens, masks = st_base.collapse_and_combine_temporal(x)
        assert tokens.shape == (B * H * W, T * B_S + B_S + 1, D)
        assert masks.shape == (B * H * W, T * B_S + B_S + 1)

        modality_tokens_dict = st_base.split_and_expand_per_modality_temporal(
            tokens, modalities_to_dims_dict
        )
        assert (modality_tokens_dict["sentinel2_l2a"] == x["sentinel2_l2a"]).all()
        assert (modality_tokens_dict["worldcover"] == x["worldcover"]).all()
        # Use approximate here since we perform mean pooling which can change the value
        # slightly.
        assert ((modality_tokens_dict["latlon"] - x["latlon"]).abs() < 1e-6).all()

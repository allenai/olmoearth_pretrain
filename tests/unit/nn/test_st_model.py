"""Unit tests for the st_model module."""

import logging

import pytest
import torch

from olmoearth_pretrain.datatypes import MaskValue
from olmoearth_pretrain.modalities import Modality
from olmoearth_pretrain.nn.flexi_vit import (
    get_modalities_to_process,
    return_modalities_from_dict,
)
from olmoearth_pretrain.nn.st_model import STBase, STPredictor

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
        worldcover_tokens = torch.randn(B, H, W, 1, B_S, D)
        worldcover_mask = torch.randint(0, 2, (B, H, W, 1, B_S)).float()
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

    def test_unsupported_attention_mode_errors(self, st_base: STBase) -> None:
        """Unsupported attention modes should fail explicitly."""
        with pytest.raises(ValueError, match="Unsupported attention mode"):
            st_base.collapse_and_combine({}, mode="bad", block_idx=0)  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="Unsupported attention mode"):
            st_base.split_and_expand_per_modality(
                torch.empty(1, 0, 8),
                {},
                mode="bad",  # type: ignore[arg-type]
                block_idx=0,
            )

    def test_split_x_y_does_not_mutate_mask(self) -> None:
        """Test ST token splitting without mutating the caller's mask."""
        tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).unsqueeze(-1)
        mask = torch.tensor(
            [
                [
                    MaskValue.MISSING.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.DECODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                ],
                [
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.MISSING.value,
                    MaskValue.DECODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                ],
            ]
        )
        original_mask = mask.clone()

        (
            tokens_to_decode,
            unmasked_tokens,
            tokens_to_decode_mask,
            unmasked_tokens_mask,
            indices,
        ) = STPredictor.split_x_y(tokens, mask)

        assert torch.equal(mask, original_mask)
        assert tokens_to_decode.shape == (2, 1, 1)
        assert unmasked_tokens.shape == (2, 2, 1)
        assert tokens_to_decode_mask.shape == (2, 1)
        assert unmasked_tokens_mask.shape == (2, 2)
        assert indices.shape == (2, 4)

    def test_collapse_and_split_temporal(
        self,
        st_base: STBase,
    ) -> None:
        """Test collapsing and re-splitting tokens for temporal attention."""
        B, H, W, T, B_S, D = 2, 3, 3, 4, 2, 4
        sentinel2_l2a_tokens = torch.randn(B, H, W, T, B_S, D)
        sentinel2_l2a_mask = torch.randint(0, 2, (B, H, W, T, B_S)).float()
        worldcover_tokens = torch.randn(B, H, W, 1, B_S, D)
        worldcover_mask = torch.randint(0, 2, (B, H, W, 1, B_S)).float()
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

    @pytest.mark.parametrize("block_idx", [0, 1])
    @pytest.mark.parametrize("window_size", [2, 3])
    def test_collapse_and_split_windowed(
        self,
        st_base: STBase,
        block_idx: int,
        window_size: int,
    ) -> None:
        """Test collapsing and re-splitting tokens for temporal attention."""
        st_base.windowed_attention_size = window_size
        B, H, W, T, B_S, D = 2, 5, 5, 4, 2, 4
        sentinel2_l2a_tokens = torch.randn(B, H, W, T, B_S, D)
        sentinel2_l2a_mask = torch.randint(0, 2, (B, H, W, T, B_S)).float()
        worldcover_tokens = torch.randn(B, H, W, 1, B_S, D)
        worldcover_mask = torch.randint(0, 2, (B, H, W, 1, B_S)).float()
        x = {
            "sentinel2_l2a": sentinel2_l2a_tokens,
            "sentinel2_l2a_mask": sentinel2_l2a_mask,
            "worldcover": worldcover_tokens,
            "worldcover_mask": worldcover_mask,
        }
        modalities_to_process = get_modalities_to_process(
            return_modalities_from_dict(x), st_base.supported_modality_names
        )
        modalities_to_dims_dict = {
            modality: x[modality].shape for modality in modalities_to_process
        }
        tokens, _ = st_base.collapse_and_combine_windowed(x, block_idx)

        modality_tokens_dict = st_base.split_and_expand_per_modality_windowed(
            tokens, modalities_to_dims_dict, window_size, block_idx
        )
        assert (modality_tokens_dict["sentinel2_l2a"] == x["sentinel2_l2a"]).all()
        assert (modality_tokens_dict["worldcover"] == x["worldcover"]).all()

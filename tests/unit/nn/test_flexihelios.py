"""Unit tests for the flexihelios module."""

import logging

import pytest
import torch
from einops import repeat

from helios.data.constants import Modality, ModalitySpec
from helios.nn.flexihelios import (
    Encoder,
    EncoderConfig,
    FlexiHeliosBase,
    FlexiHeliosCompositeEncodings,
    PoolingType,
    Predictor,
    PredictorConfig,
    ProjectAndAggregate,
    TokensAndMasks,
)
from helios.train.masking import MaskValue

logger = logging.getLogger(__name__)


class TestFlexiHeliosCompositeEncodings:
    """Unit tests for the FlexiHeliosCompositeEncodings class."""

    @pytest.fixture
    def flexi_helios_composite_encodings(
        self,
    ) -> FlexiHeliosCompositeEncodings:
        """Create composite encoder fixture for testing."""
        flexi_helios_composite_encodings = FlexiHeliosCompositeEncodings(
            embedding_size=16,
            supported_modalities=[
                Modality.SENTINEL2_L2A,
                Modality.LATLON,
                Modality.WORLDCOVER,
            ],
            max_sequence_length=12,
            random_channel_embeddings=True,
        )
        return flexi_helios_composite_encodings

    def test_apply_encodings_per_modality_latlon(
        self,
        flexi_helios_composite_encodings: FlexiHeliosCompositeEncodings,
    ) -> None:
        """Test applying encodings to different modalities."""
        B, D = 4, 16
        patch_size = 4
        input_res = 10
        latlon_tokens = torch.randn(B, 1, D)
        ll_enc = flexi_helios_composite_encodings._apply_encodings_per_modality(
            "latlon", latlon_tokens, None, patch_size, input_res
        )
        assert not (ll_enc == 0).all()
        assert not (ll_enc == latlon_tokens).all()
        assert latlon_tokens.shape == ll_enc.shape

    def test_apply_encodings_per_modality_sentinel2_l2a(
        self, flexi_helios_composite_encodings: FlexiHeliosCompositeEncodings
    ) -> None:
        """Test applying encodings to different modalities."""
        B, H, W, T, C, D = 4, 4, 4, 3, 3, 16
        patch_size = 4
        input_res = 10
        timestamps = torch.tensor(
            [[15, 7, 2023], [15, 8, 2023], [15, 9, 2023]], dtype=torch.long
        )
        timestamps = repeat(timestamps, "... -> b ...", b=B)
        sentinel2_l2a_tokens = torch.zeros(B, H, W, T, C, D)
        enc = flexi_helios_composite_encodings._apply_encodings_per_modality(
            "sentinel2_l2a", sentinel2_l2a_tokens, timestamps, patch_size, input_res
        )
        assert not (enc == 0).all()

    def test_apply_encodings_per_modality_worldcover(
        self,
        flexi_helios_composite_encodings: FlexiHeliosCompositeEncodings,
    ) -> None:
        """Test applying encodings to different modalities."""
        B, H, W, C, D = 4, 4, 4, 1, 16
        patch_size = 4
        input_res = 10
        worldcover_tokens = torch.randn(B, H, W, C, D)
        wc_enc = flexi_helios_composite_encodings._apply_encodings_per_modality(
            "worldcover", worldcover_tokens, None, patch_size, input_res
        )
        assert not (wc_enc == 0).all()
        assert not (wc_enc == worldcover_tokens).all()
        assert worldcover_tokens.shape == wc_enc.shape

    def test_apply_encodings_per_modality_grad(
        self, flexi_helios_composite_encodings: FlexiHeliosCompositeEncodings
    ) -> None:
        """Test applying encodings to different modalities."""
        B, H, W, T, C, D = 4, 4, 4, 3, 3, 16
        patch_size = 4
        input_res = 10
        timestamps = torch.tensor(
            [[15, 7, 2023], [15, 8, 2023], [15, 9, 2023]], dtype=torch.long
        )
        timestamps = repeat(timestamps, "... -> b ...", b=B)
        sentinel2_l2a_tokens = torch.zeros(B, H, W, T, C, D)
        assert (
            flexi_helios_composite_encodings.per_modality_channel_embeddings[
                "sentinel2_l2a"
            ].grad
            is None
        )
        enc = flexi_helios_composite_encodings._apply_encodings_per_modality(
            "sentinel2_l2a", sentinel2_l2a_tokens, timestamps, patch_size, input_res
        )
        loss = enc.sum()
        loss.backward()
        assert (
            flexi_helios_composite_encodings.per_modality_channel_embeddings[
                "sentinel2_l2a"
            ].grad
            is not None
        )


# TODO: Add tests for when the inputs are completely masked or different dims or something
class TestFlexiHeliosBase:
    """Unit tests for the FlexiHeliosBase class."""

    @pytest.fixture
    def flexi_helios_base(
        self, supported_modalities: list[ModalitySpec]
    ) -> FlexiHeliosBase:
        """Create encoder fixture for testing."""
        flexi_helios_base = FlexiHeliosBase(
            embedding_size=8,
            num_heads=2,
            mlp_ratio=4.0,
            depth=2,
            drop_path=0.1,
            supported_modalities=supported_modalities,
            max_sequence_length=12,
        )
        return flexi_helios_base

    def test_collapse_and_combine_hwtc(
        self, flexi_helios_base: FlexiHeliosBase
    ) -> None:
        """Test collapsing tokens from different modalities into single tensor."""
        B, D = 2, 4
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
        tokens, masks = flexi_helios_base.collapse_and_combine_hwtc(x)
        assert tokens.shape == (B, 5, D)
        assert masks.shape == (B, 5)

    def test_split_and_expand_per_modality(self) -> None:
        """Test splitting combined tensor back into per-modality tensors."""
        B, D = 2, 4  # Batch size and embedding dimension
        modality_1_channel_groups = 3
        modality_2_channel_groups = 5
        modalities_to_dims_dict: dict[str, tuple[int, ...]] = {
            "modality1": (B, 2, 2, 1, modality_1_channel_groups, D),
            "modality2": (B, 1, 1, 2, modality_2_channel_groups, D),
        }

        modality1_data = torch.randn(B, 4 * modality_1_channel_groups, D)
        modality2_data = torch.randn(B, 4 * modality_2_channel_groups, D)

        x = torch.cat([modality1_data, modality2_data], dim=1)

        # Now call the function
        modality_tokens_dict = FlexiHeliosBase.split_and_expand_per_modality(
            x, modalities_to_dims_dict
        )

        modality1_tokens = modality_tokens_dict["modality1"]
        modality2_tokens = modality_tokens_dict["modality2"]
        assert list(modality1_tokens.shape) == [
            2,
            2,
            2,
            1,
            3,
            4,
        ], f"Incorrect shape for modality1 tokens: {modality1_tokens.shape}"
        assert list(modality2_tokens.shape) == [
            2,
            1,
            1,
            2,
            5,
            4,
        ], f"Incorrect shape for modality2 tokens: {modality2_tokens.shape}"


class TestEncoder:
    """Unit tests for the Encoder class."""

    @pytest.fixture
    def encoder(self, supported_modalities: list[ModalitySpec]) -> Encoder:
        """Create encoder fixture for testing.

        Returns:
            Encoder: Test encoder instance with small test config
        """
        return Encoder(
            embedding_size=8,
            max_patch_size=8,
            min_patch_size=1,
            num_heads=2,
            mlp_ratio=4.0,
            depth=2,
            drop_path=0.1,
            supported_modalities=supported_modalities,
            max_sequence_length=12,
        )

    def test_create_token_exit_ids_normal_usage(self, encoder: Encoder) -> None:
        """Test creating exit IDs for early token exiting - normal usage.

        Tests normal usage with full token_exit_cfg.
        """
        B, H, W, T, D = 1, 2, 2, 2, 4
        sentinel2_l2a_tokens = torch.zeros(B, H, W, T, D)
        latlon_tokens = torch.randn(B, 1, D)
        x = {"sentinel2_l2a": sentinel2_l2a_tokens, "latlon": latlon_tokens}

        token_exit_cfg = {"sentinel2_l2a": 1, "latlon": 2}
        exit_ids_dict = encoder.create_token_exit_ids(x, token_exit_cfg)
        assert (
            "sentinel2_l2a" in exit_ids_dict
        ), "Expected 'sentinel2_l2a' key in the result dict"
        sentinel2_l2a_exit_ids = exit_ids_dict["sentinel2_l2a"]
        assert (
            sentinel2_l2a_exit_ids.shape == sentinel2_l2a_tokens.shape
        ), "Shape of exit IDs should match the shape of the modality tokens."

        assert (exit_ids_dict["sentinel2_l2a"] == 1).all()
        assert (exit_ids_dict["latlon"] == 2).all()

    def test_create_token_exit_ids_missing_exit_cfg_band_group(
        self, encoder: Encoder
    ) -> None:
        """Test creating exit IDs for early token exiting - error cases.

        Tests error handling for:
        - Missing band group in token_exit_cfg (KeyError)
        """
        B, H, W, T, D = 1, 2, 2, 2, 4
        sentinel2_l2a_tokens = torch.zeros(B, H, W, T, D)
        x = {"sentinel2_l2a": sentinel2_l2a_tokens}

        with pytest.raises(KeyError):
            incomplete_exit_cfg = {"rgb": 1}  # Missing the "nir" key
            encoder.create_token_exit_ids(x, incomplete_exit_cfg)

    def test_remove_masked_tokens(self) -> None:
        """Test removing masked tokens and tracking indices."""
        d = 2
        x = torch.tensor([[0, 1, 0], [1, 0, 1]]).float()
        x = repeat(x, "b n -> b n d", d=d)
        print(f"x shape: {x.shape}")
        mask = torch.tensor([[0, 1, 0], [1, 0, 1]]).bool()

        expected_tokens = torch.tensor(
            [
                [[1.0, 1.0], [0.0, 0.0]],
                [[1.0, 1.0], [1.0, 1.0]],
            ]
        )
        num_tokens_to_keep = torch.sum(mask)
        expected_indices = torch.tensor([[1, 0, 2], [0, 2, 1]])
        expected_updated_mask = torch.tensor([[1, 0], [1, 1]]).bool()
        tokens, indices, updated_mask, seqlens, max_length = (
            Encoder.remove_masked_tokens(x, mask)
        )
        kept_unmasked_tokens = torch.sum(updated_mask)
        assert torch.equal(tokens, expected_tokens)
        assert torch.equal(indices, expected_indices)
        assert torch.equal(updated_mask, expected_updated_mask)
        assert kept_unmasked_tokens == num_tokens_to_keep
        assert seqlens.shape == (2,)
        assert max_length == 2

    def test_add_removed_tokens(self) -> None:
        """Test adding removed tokens back into tensor."""
        partial_tokens = torch.tensor(
            [
                [[1.0, 11.0], [2.0, 22.0]],
                [[5.0, 55.0], [6.0, 66.0]],
            ]
        )
        indices = torch.tensor(
            [
                [0, 1, 2],
                [1, 0, 2],
            ]
        )
        partial_mask = torch.tensor(
            [
                [1, 1],
                [1, 0],
            ]
        ).bool()

        expected_out = torch.tensor(
            [
                [[1.0, 11.0], [2.0, 22.0], [0.0, 0.0]],
                [[0.0, 0.0], [5.0, 55.0], [0.0, 0.0]],
            ]
        )
        expected_mask = torch.tensor(
            [
                [1, 1, 0],
                [0, 1, 0],
            ]
        ).bool()

        out, full_mask = Encoder.add_removed_tokens(
            partial_tokens, indices, partial_mask
        )
        assert torch.equal(out, expected_out)
        assert torch.equal(full_mask, expected_mask)

    def test_encoder_config(self, supported_modalities: list[ModalitySpec]) -> None:
        """Tests we can build with default args."""
        supported_modality_names = [m.name for m in supported_modalities]
        config = EncoderConfig(supported_modality_names)
        _ = config.build()


class TestPredictor:
    """Unit tests for the Predictor class."""

    @pytest.fixture
    def predictor(self, supported_modalities: list[ModalitySpec]) -> Predictor:
        """Create predictor fixture for testing."""
        return Predictor(
            supported_modalities=supported_modalities,
            encoder_embedding_size=8,
            decoder_embedding_size=8,
            depth=2,
            mlp_ratio=4.0,
            num_heads=2,
            max_sequence_length=12,
            drop_path=0.1,
            output_embedding_size=8,
        )

    def test_add_masks(self, predictor: Predictor) -> None:
        """Test adding masks to tokens."""
        B, H, W, T, C, D = (
            1,
            2,
            2,
            1,
            2,
            8,
        )  # Changed D from 16 to 8 to match predictor's embedding size
        sentinel2_l2a_tokens = torch.randn(B, H, W, T, C, D)
        sentinel2_l2a_mask = torch.zeros(B, H, W, T, C, dtype=torch.float32)
        # Set one pixel to be decoded (mask value 2)
        sentinel2_l2a_mask[0, 0, 0, 0, 0] = MaskValue.DECODER.value

        latlon = torch.randn(B, 2, D)
        latlon_mask = torch.zeros(B, 2, dtype=torch.float32)

        tokens_and_masks = {
            "sentinel2_l2a": sentinel2_l2a_tokens,
            "sentinel2_l2a_mask": sentinel2_l2a_mask,
            "latlon": latlon,
            "latlon_mask": latlon_mask,
        }
        replaced_dict = predictor.add_masks(tokens_and_masks)

        # We expect replaced_dict to have the key "sentinel2_l2a"
        assert (
            "sentinel2_l2a" in replaced_dict
        ), "Expected replaced_dict to have key 'sentinel2_l2a'"

        replaced_sentinel2_l2a = replaced_dict["sentinel2_l2a"]
        assert replaced_sentinel2_l2a.shape == sentinel2_l2a_tokens.shape, (
            f"Expected shape {sentinel2_l2a_tokens.shape}, "
            f"got {replaced_sentinel2_l2a.shape}"
        )

        # Check the single pixel we set to be decoded
        replaced_location = replaced_sentinel2_l2a[0, 0, 0, 0, 0, :]

        # Check an unchanged location
        unchanged_location = replaced_sentinel2_l2a[0, 0, 0, 0, 1, :]

        assert torch.allclose(
            replaced_location, predictor.mask_token, atol=1e-6
        ), "Tokens at masked location should be replaced with mask token."
        assert torch.allclose(
            unchanged_location, sentinel2_l2a_tokens[0, 0, 0, 0, 1, :], atol=1e-6
        ), "Tokens at non-masked location should remain the same."

    def test_split_x_y(self) -> None:
        """Test splitting the tokens into decoded, unmasked, and missing groups."""
        tokens = torch.tensor(
            [[1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 16, 17, 18]]
        ).unsqueeze(-1)
        # should we handle target encoder values here?
        mask = torch.tensor(
            [
                [
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.DECODER.value,
                    MaskValue.DECODER.value,
                    MaskValue.DECODER.value,
                ],
                [
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.DECODER.value,
                    MaskValue.DECODER.value,
                ],
            ]
        )

        # Add some missing tokens (value MISSING)
        mask[0, 0] = MaskValue.MISSING.value  # First token in first batch is missing
        mask[1, 1] = MaskValue.MISSING.value  # Second token in second batch is missing

        (
            tokens_to_decode,
            unmasked_tokens,
            tokens_to_decode_mask,
            unmasked_tokens_mask,
            indices,
            seqlens_tokens_to_decode,
            seqlens_unmasked_tokens,
            max_length_of_decoded_tokens,
            max_length_of_unmasked_tokens,
        ) = Predictor.split_x_y(tokens, mask)
        # Check shapes
        assert unmasked_tokens.shape == (2, 6, 1)
        assert tokens_to_decode.shape == (2, 3, 1)

        expected_unmasked_tokens = torch.tensor(
            [[1, 2, 3, 4, 5, 6], [10, 12, 13, 14, 15, 16]]
        )
        assert torch.equal(unmasked_tokens.squeeze(-1), expected_unmasked_tokens)
        assert torch.equal(
            unmasked_tokens_mask, torch.tensor([[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
        )

        expected_tokens_to_decode = torch.tensor([[7, 8, 9], [17, 18, 11]])
        assert torch.equal(tokens_to_decode.squeeze(-1), expected_tokens_to_decode)
        assert torch.equal(tokens_to_decode_mask, torch.tensor([[1, 1, 1], [1, 1, 0]]))
        assert torch.equal(seqlens_tokens_to_decode, torch.tensor([3, 2]))
        assert torch.equal(seqlens_unmasked_tokens, torch.tensor([5, 6]))
        assert max_length_of_decoded_tokens == 3
        assert max_length_of_unmasked_tokens == 6

    def test_split_and_recombine_with_missing_tokens(self) -> None:
        """Test splitting the tokens into decoded, unmasked, and missing groups with missing tokens."""
        # Create a batch with two samples, with tokens in a non-sorted order
        # and different numbers of missing tokens per batch
        tokens = torch.tensor(
            [
                [1, 2, 3, 4, 5, 6, 7, 8, 9],  # First batch
                [10, 11, 12, 13, 14, 15, 16, 17, 18],  # Second batch
            ]
        ).unsqueeze(-1)

        # Create masks with different patterns of missing tokens
        # First batch: 1 missing token, 3 decoder tokens, 5 encoder tokens
        # Second batch: 3 missing tokens, 2 decoder tokens, 4 encoder tokens
        # The tokens are intentionally not sorted by mask value
        mask = torch.tensor(
            [
                [
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.DECODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.MISSING.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.DECODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.DECODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                ],
                [
                    MaskValue.MISSING.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.DECODER.value,
                    MaskValue.MISSING.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.DECODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.MISSING.value,
                    MaskValue.ONLINE_ENCODER.value,
                ],
            ]
        )

        (
            tokens_to_decode,
            unmasked_tokens,
            tokens_to_decode_mask,
            unmasked_tokens_mask,
            indices,
            _,
            _,
            _,
            _,
        ) = Predictor.split_x_y(tokens, mask)

        # Check shapes
        assert tokens_to_decode.shape == (2, 3, 1)
        assert unmasked_tokens.shape == (2, 5, 1)

        expected_unmasked_tokens = torch.tensor([[1, 3, 5, 7, 9], [17, 11, 14, 16, 18]])
        assert torch.equal(unmasked_tokens.squeeze(-1), expected_unmasked_tokens)

        expected_unmasked_tokens_mask = torch.tensor([[1, 1, 1, 1, 1], [0, 1, 1, 1, 1]])
        assert torch.equal(unmasked_tokens_mask, expected_unmasked_tokens_mask)

        expected_tokens_to_decode = torch.tensor([[2, 6, 8], [12, 15, 10]])
        assert torch.equal(tokens_to_decode.squeeze(-1), expected_tokens_to_decode)

        expected_tokens_to_decode_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
        assert torch.equal(tokens_to_decode_mask, expected_tokens_to_decode_mask)

        # Test that we can combine the tokens back correctly
        combined_tokens = Predictor.combine_x_y(
            tokens_to_decode=tokens_to_decode,
            unmasked_tokens=unmasked_tokens,
            tokens_to_decode_mask=tokens_to_decode_mask,
            unmasked_tokens_mask=unmasked_tokens_mask,
            indices=indices,
        )
        # Check the shape of the combined tokens
        assert combined_tokens.shape == tokens.shape
        missing_mask = mask == MaskValue.MISSING.value
        # Check that all values are the same but missing values in mask are set to 0
        assert (combined_tokens[missing_mask] == 0).all()
        target_encoder_only_mask = mask == MaskValue.TARGET_ENCODER_ONLY.value
        missing_or_target_encoder_only_mask = missing_mask | target_encoder_only_mask
        # Ensuring Encode decode tokens are put back together correctly
        assert torch.equal(
            combined_tokens[~missing_or_target_encoder_only_mask],
            tokens[~missing_or_target_encoder_only_mask],
        )

    def test_split_and_recombine_with_missing_tokens_and_target_encoder_only_tokens(
        self,
    ) -> None:
        """Test splitting the tokens into decoded, unmasked, and missing groups with missing tokens and target encoder only tokens."""
        tokens = torch.tensor(
            [
                [1, 2, 3, 4, 5, 6, 7, 8, 9],  # First batch
                [10, 11, 12, 13, 14, 15, 16, 17, 18],  # Second batch
            ]
        ).unsqueeze(-1)

        mask = torch.tensor(
            [
                [
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.TARGET_ENCODER_ONLY.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.MISSING.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.TARGET_ENCODER_ONLY.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.DECODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                ],
                [
                    MaskValue.MISSING.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.TARGET_ENCODER_ONLY.value,
                    MaskValue.MISSING.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.TARGET_ENCODER_ONLY.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.DECODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                ],
            ]
        )

        (
            tokens_to_decode,
            unmasked_tokens,
            tokens_to_decode_mask,
            unmasked_tokens_mask,
            indices,
            _,
            _,
            _,
            _,
        ) = Predictor.split_x_y(tokens, mask)

        combined_tokens = Predictor.combine_x_y(
            tokens_to_decode=tokens_to_decode,
            unmasked_tokens=unmasked_tokens,
            tokens_to_decode_mask=tokens_to_decode_mask,
            unmasked_tokens_mask=unmasked_tokens_mask,
            indices=indices,
        )
        # check that it is zero wherever there is a target encoder only token
        target_encoder_only_mask = mask == MaskValue.TARGET_ENCODER_ONLY.value
        missing_mask = mask == MaskValue.MISSING.value
        missing_or_target_encoder_only_mask = missing_mask | target_encoder_only_mask
        assert (combined_tokens[missing_or_target_encoder_only_mask] == 0).all()
        assert torch.equal(
            combined_tokens[~missing_or_target_encoder_only_mask],
            tokens[~missing_or_target_encoder_only_mask],
        )

    def test_combine_x_y(self) -> None:
        """Test combining the decoded, unmasked, and missing groups back into the original tokens."""
        # x is the query (i.e. the masked tokens)
        tokens_to_decode = torch.tensor([[14, 15, 16], [15, 16, 1]]).unsqueeze(-1)
        # y is the keys and values (i.e. the unmasked tokens)
        unmasked_tokens = torch.tensor([[5, 6, 7, 8], [4, 5, 6, 7]]).unsqueeze(-1)
        tokens_to_decode_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
        unmasked_tokens_mask = torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1]])
        indices = torch.tensor(
            [[6, 7, 8, 4, 5, 0, 1, 2, 3], [7, 8, 3, 4, 5, 6, 0, 1, 2]]
        )

        tokens = Predictor.combine_x_y(
            tokens_to_decode=tokens_to_decode,
            unmasked_tokens=unmasked_tokens,
            tokens_to_decode_mask=tokens_to_decode_mask,
            unmasked_tokens_mask=unmasked_tokens_mask,
            indices=indices,
        )
        assert torch.equal(
            tokens,
            torch.tensor(
                [[5, 6, 7, 8, 0, 0, 14, 15, 16], [5, 6, 7, 0, 0, 0, 0, 15, 16]]
            ).unsqueeze(-1),
        )

    def test_predictor_config(self, supported_modalities: list[ModalitySpec]) -> None:
        """Tests we can build with default args."""
        supported_modality_names = [m.name for m in supported_modalities]
        config = PredictorConfig(supported_modality_names)
        _ = config.build()


class TestTokensAndMasks:
    """Test TestTokensAndMasks."""

    def test_flatten_tokens_and_masks(self) -> None:
        """Test TokensAndMasks.flatten_tokens_and_masks."""
        b, h, w, t, d = 2, 4, 4, 3, 128
        sentinel_2 = torch.ones((b, h, w, t, d))
        sentinel_2[0, 0, 0, 0, :] = 0  # set one "token" to 0s
        sentinel_2_mask = torch.zeros((b, h, w, t)).long()
        sentinel_2_mask[0, 0, 0, 0] = 1  # set the same token's mask to 1
        t_and_m = TokensAndMasks(
            sentinel2_l2a=sentinel_2, sentinel2_l2a_mask=sentinel_2_mask
        )
        x, mask = t_and_m.flatten_tokens_and_masks()

        assert x.shape == (b, h * w * t, d)
        assert mask.shape == (b, h * w * t)
        assert (x[mask.bool()] == 0).all()
        assert (x[(1 - mask).bool()] == 1).all()

    def test_pool_unmasked_tokens(self) -> None:
        """Test TokensAndMasks.pool_unmasked_tokens."""
        b, h, w, t, b_s, d = 2, 4, 4, 3, 1, 128
        # Setup for mean pooling
        sentinel_2_mean = torch.ones((b, h, w, t, b_s, d))
        sentinel_2_mean[0, 0, 0, 0, :] = 0  # set one "token" to 0s
        sentinel_2_mask_mean = torch.zeros((b, h, w, t, b_s)).long()
        sentinel_2_mask_mean[0, 0, 0, 0] = 1  # set the same token's mask to 1
        t_and_m_mean = TokensAndMasks(
            sentinel2_l2a=sentinel_2_mean, sentinel2_l2a_mask=sentinel_2_mask_mean
        )
        # Setup for max pooling
        sentinel_2_max = torch.ones((b, h, w, t, b_s, d)) * 2  # set all tokens to 2
        sentinel_2_max[0, 0, 0, 0, :] = 3  # set one "token" to 3s for max pooling
        sentinel_2_mask_max = torch.zeros((b, h, w, t, b_s)).long()
        sentinel_2_mask_max[0, 0, 0, 0] = 1  # set the same token's mask to 1
        t_and_m_max = TokensAndMasks(
            sentinel2_l2a=sentinel_2_max, sentinel2_l2a_mask=sentinel_2_mask_max
        )

        # Test max pooling
        pooled_max = t_and_m_max.pool_unmasked_tokens(PoolingType.MAX)
        assert pooled_max.shape == (b, d)
        assert (pooled_max == 2).all()  # check the 3 tokens have been ignored

        # Test mean pooling
        pooled_mean = t_and_m_mean.pool_unmasked_tokens(PoolingType.MEAN)
        assert pooled_mean.shape == (b, d)
        assert (pooled_mean == 1).all()  # check the 0 tokens have been ignored

    def test_spatial_pool_unmasked_tokens(self) -> None:
        """Test TokensAndMasks.pool_unmasked_tokens."""
        b, h, w, t, b_s, d = 2, 4, 4, 3, 3, 128
        # Setup for mean pooling
        sentinel_2_mean = torch.ones((b, h, w, t, b_s, d))
        sentinel_2_mask_mean = torch.zeros((b, h, w, t, b_s)).long()
        # s1 should be ignored since its masked
        sentinel_1_mean = torch.ones((b, h, w, t, b_s, d)) * 2
        sentinel_1_mask_mean = torch.ones((b, h, w, t, b_s)).long()
        t_and_m_mean = TokensAndMasks(
            sentinel2_l2a=sentinel_2_mean,
            sentinel2_l2a_mask=sentinel_2_mask_mean,
            sentinel1=sentinel_1_mean,
            sentinel1_mask=sentinel_1_mask_mean,
        )

        # Test mean pooling
        pooled_mean = t_and_m_mean.pool_unmasked_tokens(
            PoolingType.MEAN, spatial_pooling=True
        )
        assert pooled_mean.shape == (b, h, w, d)
        assert (pooled_mean == 1).all()  # check the sen1 tokens have been ignored

        # Setup for max pooling
        sentinel_2_max = torch.ones((b, h, w, t, b_s, d))
        sentinel_2_max[:, :, :, 0] = 2  # set one timestep to 2s
        sentinel_2_mask_max = torch.zeros((b, h, w, t, b_s)).long()
        # s1 should be ignored since its masked
        sentinel_1_mean = torch.ones((b, h, w, t, b_s, d)) * 2
        sentinel_1_mask_mean = torch.ones((b, h, w, t, b_s)).long()
        t_and_m_max = TokensAndMasks(
            sentinel2_l2a=sentinel_2_max,
            sentinel2_l2a_mask=sentinel_2_mask_max,
            sentinel1=sentinel_1_mean,
            sentinel1_mask=sentinel_1_mask_mean,
        )

        # Test max pooling
        pooled_max = t_and_m_max.pool_unmasked_tokens(
            PoolingType.MAX, spatial_pooling=True
        )
        assert pooled_max.shape == (b, h, w, d)
        assert (pooled_max == 2).all()  # check the 3 tokens have been ignored

    def test_missing_modalities_ignored(self) -> None:
        """Test TokensAndMasks.modalities does not return missing modalities."""
        b, h, w, t, b_s, d = 2, 4, 4, 3, 3, 128
        # Setup for mean pooling
        sentinel_2_mean = torch.ones((b, h, w, t, b_s, d))
        sentinel_2_mask_mean = torch.zeros((b, h, w, t, b_s)).long()
        # s1 should be ignored since its masked
        sentinel_1_mean = torch.ones((b, h, w, t, b_s, d)) * 2
        sentinel_1_mask_mean = torch.ones((b, h, w, t, b_s)).long()
        t_and_m_mean = TokensAndMasks(
            sentinel2_l2a=sentinel_2_mean,
            sentinel2_l2a_mask=sentinel_2_mask_mean,
            sentinel1=sentinel_1_mean,
            sentinel1_mask=sentinel_1_mask_mean,
        )

        modalities = t_and_m_mean.modalities
        assert len(modalities) == 2  # s2, s1
        assert set(modalities) == set(["sentinel2_l2a", "sentinel1"])


class TestProjectionAndAggregation:
    """Test ProjectAndAggregate."""

    def test_layer_in_all_configs(self) -> None:
        """Test ProjectAndAggregate."""
        b, h, w, t, d = 2, 4, 4, 3, 128
        sentinel_2 = torch.ones((b, h, w, t, d))
        sentinel_2[0, 0, 0, 0, :] = 0  # set one "token" to 0s
        sentinel_2_mask = torch.zeros((b, h, w, t)).long()
        sentinel_2_mask[0, 0, 0, 0] = 1  # set the same token's mask to 1
        t_and_m = TokensAndMasks(
            sentinel2_l2a=sentinel_2, sentinel2_l2a_mask=sentinel_2_mask
        )

        for i in [1, 2, 3]:
            for pre_aggregate in [True, False]:
                layer = ProjectAndAggregate(
                    embedding_size=d, num_layers=i, aggregate_then_project=pre_aggregate
                )
                # for now, lets just check it all runs properly
                _ = layer(t_and_m)


# TODO: write a unit test for the FlexiPatchEmbeddings

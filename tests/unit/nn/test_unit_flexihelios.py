"""Unit tests for the core model code."""

import pytest
import torch
from einops import repeat

from helios.nn.flexihelios import Encoder, FlexiHeliosBase, Predictor, TokensAndMasks
from helios.train.masking import MaskValue


class TestFlexiHeliosBase:
    """Unit tests for the FlexiHeliosBase class."""

    @pytest.fixture
    def flexi_helios_base(self) -> FlexiHeliosBase:
        """Create encoder fixture for testing."""
        modalities_dict = dict({"s2": dict({"rgb": [0, 1, 2], "nir": [3]})})
        flexi_helios_base = FlexiHeliosBase(
            embedding_size=8,
            num_heads=2,
            mlp_ratio=4.0,
            depth=2,
            drop_path=0.1,
            modalities_to_channel_groups_dict=modalities_dict,
            max_sequence_length=12,
            base_patch_size=4,
            use_channel_embs=True,
        )
        return flexi_helios_base

    def test_collapse_and_combine_hwtc(
        self, flexi_helios_base: FlexiHeliosBase
    ) -> None:
        """Test collapsing tokens from different modalities into single tensor."""
        B, D = 2, 4
        s2_tokens = torch.randn(B, 2, 1, 1, 2, D)
        s2_mask = torch.randint(0, 2, (B, 2, 1, 1, 2)).float()
        latlon = torch.randn(B, 2, 1, 1, 2, 2)
        latlon_mask = torch.randint(0, 2, (B, 2, 1, 1, 2)).float()
        x = TokensAndMasks(
            s2=s2_tokens, s2_mask=s2_mask, latlon=latlon, latlon_mask=latlon_mask
        )
        tokens, masks = flexi_helios_base.collapse_and_combine_hwtc(x)
        assert tokens.shape == (B, 4, D)
        assert masks.shape == (B, 4)

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
    def encoder(self) -> Encoder:
        """Create encoder fixture for testing.

        Returns:
            Encoder: Test encoder instance with small test config
        """
        modalities_dict = dict({"s2": dict({"rgb": [0, 1, 2], "nir": [3]})})
        return Encoder(
            embedding_size=8,
            max_patch_size=8,
            num_heads=2,
            mlp_ratio=4.0,
            depth=2,
            drop_path=0.1,
            modalities_to_channel_groups_dict=modalities_dict,
            max_sequence_length=12,
            base_patch_size=4,
            use_channel_embs=True,
        )

    def test_create_token_exit_ids_normal_usage(self, encoder: Encoder) -> None:
        """Test creating exit IDs for early token exiting - normal usage.

        Tests normal usage with full token_exit_cfg.
        """
        B, H, W, T, D = 1, 2, 2, 2, 4
        s2_tokens = torch.zeros(B, H, W, T, D)
        x = {"s2": s2_tokens}

        token_exit_cfg = {"rgb": 1, "nir": 2}
        exit_ids_dict = encoder.create_token_exit_ids(x, token_exit_cfg)
        assert "s2" in exit_ids_dict, "Expected 's2' key in the result dict"
        s2_exit_ids = exit_ids_dict["s2"]
        assert (
            s2_exit_ids.shape == s2_tokens.shape
        ), "Shape of exit IDs should match the shape of the modality tokens."

        assert (
            s2_exit_ids[:, :, :, 0, :] == 1
        ).all(), "Expected the first band group ('rgb') tokens to be set to 1"
        assert (
            s2_exit_ids[:, :, :, 1, :] == 2
        ).all(), "Expected the second band group ('nir') tokens to be set to 2"

    def test_create_token_exit_ids_missing_exit_cfg_band_group(
        self, encoder: Encoder
    ) -> None:
        """Test creating exit IDs for early token exiting - error cases.

        Tests error handling for:
        - Missing band group in token_exit_cfg (KeyError)
        """
        B, H, W, T, D = 1, 2, 2, 2, 4
        s2_tokens = torch.zeros(B, H, W, T, D)
        x = {"s2": s2_tokens}

        with pytest.raises(KeyError):
            incomplete_exit_cfg = {"rgb": 1}  # Missing the "nir" key
            encoder.create_token_exit_ids(x, incomplete_exit_cfg)

    def test_remove_masked_tokens(self) -> None:
        """Test removing masked tokens and tracking indices."""
        d = 2
        x = torch.tensor([[0, 1, 0], [1, 0, 1]]).float()
        x = repeat(x, "b n -> b n d", d=d)
        print(f"x shape: {x.shape}")
        mask = torch.tensor([[1, 0, 1], [0, 1, 0]]).float()

        expected_tokens = torch.tensor(
            [
                [[1.0, 1.0], [0.0, 0.0]],
                [[1.0, 1.0], [1.0, 1.0]],
            ]
        )
        num_tokens_to_keep = torch.sum(~mask.bool())
        expected_indices = torch.tensor([[1, 0, 2], [0, 2, 1]])
        expected_updated_mask = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
        tokens, indices, updated_mask = Encoder.remove_masked_tokens(x, mask)
        kept_unmasked_tokens = torch.sum(~updated_mask.bool())
        assert torch.equal(tokens, expected_tokens)
        assert torch.equal(indices, expected_indices)
        assert torch.equal(updated_mask, expected_updated_mask)
        assert kept_unmasked_tokens == num_tokens_to_keep

    @pytest.mark.parametrize(
        "block_idx,exit_after,expected",
        [
            (0, None, False),
            (0, 1, False),
            (1, 1, True),
            (1, 2, False),
            (2, 1, True),
        ],
    )
    def test_should_exit(
        self, block_idx: int, exit_after: int | None, expected: bool
    ) -> None:
        """Test exit condition logic.

        Args:
            block_idx: Current block index
            exit_after: Number of layers after which to exit, or None
            expected: Expected output
        """
        assert Encoder.should_exit(block_idx, exit_after) is expected

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
                [0.0, 0.0],
                [0.0, 1.0],
            ]
        )

        expected_out = torch.tensor(
            [
                [[1.0, 11.0], [2.0, 22.0], [0.0, 0.0]],
                [[0.0, 0.0], [5.0, 55.0], [0.0, 0.0]],
            ]
        )
        expected_mask = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
            ]
        )

        out, full_mask = Encoder.add_removed_tokens(
            partial_tokens, indices, partial_mask
        )
        assert torch.equal(out, expected_out)
        assert torch.equal(full_mask, expected_mask)


class TestPredictor:
    """Unit tests for the Predictor class."""

    @pytest.fixture
    def predictor(self) -> Predictor:
        """Create predictor fixture for testing."""
        modalities_to_channel_groups_dict = dict(
            {"s2": dict({"rgb": [0, 1, 2], "nir": [3]})}
        )
        return Predictor(
            modalities_to_channel_groups_dict=modalities_to_channel_groups_dict,
            encoder_embedding_size=8,
            decoder_embedding_size=8,
            depth=2,
            mlp_ratio=4.0,
            num_heads=2,
            max_sequence_length=12,
            max_patch_size=8,
            drop_path=0.1,
            learnable_channel_embeddings=True,
            output_embedding_size=8,
        )

    def test_add_masks(self, predictor: Predictor) -> None:
        """Test that marked tokens are replaced with the learnable mask token."""
        B, H, W, T, C_G, D = 1, 2, 2, 1, 5, 8
        s2_tokens = torch.randn(B, H, W, T, C_G, D)

        s2_mask = torch.zeros(B, H, W, T, C_G).float()
        s2_mask[0, 0, 0, 0, 0] = MaskValue.DECODER_ONLY.value
        latlon = torch.randn(B, 2, 1, 1, 2, 2)
        latlon_mask = torch.randint(0, 2, (B, 2, 1, 1, 2)).float()

        tokens_and_masks = TokensAndMasks(
            s2=s2_tokens, s2_mask=s2_mask, latlon=latlon, latlon_mask=latlon_mask
        )

        replaced_dict = predictor.add_masks(tokens_and_masks)

        # We expect replaced_dict to have the key "s2", shaped like s2_tokens
        assert "s2" in replaced_dict, "Expected an output key for modality s2"
        replaced_s2 = replaced_dict["s2"]
        assert replaced_s2.shape == s2_tokens.shape, (
            f"Expected shape {s2_tokens.shape} for replaced tokens, "
            f"got {replaced_s2.shape}"
        )
        replaced_location = replaced_s2[
            0, 0, 0, 0, 0, :
        ]  # the single pixel & channel we set to 2

        unchanged_location = replaced_s2[0, 0, 0, 0, 1, :]

        assert torch.allclose(
            replaced_location, predictor.mask_token, atol=1e-6
        ), "Tokens at masked=2 location were not replaced by the learnable mask token."

        assert torch.allclose(
            unchanged_location, s2_tokens[0, 0, 0, 0, 1, :], atol=1e-6
        ), "Tokens at non-masked location should remain the same."

    def test_split_x_y(self) -> None:
        """Test splitting the tokens into x and y."""
        tokens = torch.tensor(
            [[5, 6, 7, 8, 2, 13, 14, 15, 16], [5, 6, 7, 1, 2, 3, 4, 15, 16]]
        ).unsqueeze(-1)
        mask = torch.tensor([[0, 0, 0, 0, 1, 1, 2, 2, 2], [0, 0, 0, 1, 1, 1, 1, 2, 2]])

        x, y, x_mask, y_mask, _ = Predictor.split_x_y(tokens, mask)
        assert torch.equal(x, torch.tensor([[14, 15, 16], [15, 16, 1]]).unsqueeze(-1))
        assert torch.equal(y, torch.tensor([[5, 6, 7, 8], [4, 5, 6, 7]]).unsqueeze(-1))
        assert torch.equal(x_mask, torch.tensor([[1, 1, 1], [1, 1, 0]]))
        assert torch.equal(y_mask, torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1]]))

    def test_combine_x_y(self) -> None:
        """Test combining the x and y tokens."""
        # x is the query (i.e. the masked tokens)
        x = torch.tensor([[14, 15, 16], [15, 16, 1]]).unsqueeze(-1)
        # y is the keys and values (i.e. the unmasked tokens)
        y = torch.tensor([[5, 6, 7, 8], [4, 5, 6, 7]]).unsqueeze(-1)
        x_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
        y_mask = torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1]])
        indices = torch.tensor(
            [[6, 7, 8, 4, 5, 0, 1, 2, 3], [7, 8, 3, 4, 5, 6, 0, 1, 2]]
        )

        tokens = Predictor.combine_x_y(x, y, x_mask, y_mask, indices)
        assert torch.equal(
            tokens,
            torch.tensor(
                [[5, 6, 7, 8, 0, 0, 14, 15, 16], [5, 6, 7, 0, 0, 0, 0, 15, 16]]
            ).unsqueeze(-1),
        )


# TODO: write a unit test for the FlexiPatchEmbeddings

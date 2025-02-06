"""Unit tests for the core model code"""

import pytest
import torch
from einops import repeat

from helios.nn.model import Encoder, TokensAndMasks, TokensOnly


class TestEncoder:
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

    def test_collapse_and_combine_hwtc(self, encoder: Encoder) -> None:
        """Test collapsing tokens from different modalities into single tensor.

        Args:
            encoder: Test encoder instance
        """
        B, D = 2, 4
        s2_tokens = torch.randn(B, 2, 1, 1, 2, D)
        s2_mask = torch.randint(0, 2, (B, 2, 1, 1, 2)).float()
        x = TokensAndMasks(s2=s2_tokens, s2_mask=s2_mask)
        tokens, masks = encoder.collapse_and_combine_hwtc(x)
        assert tokens.shape == (B, 4, D)
        assert masks.shape == (B, 4)

    def test_create_token_exit_ids_normal_usage(self, encoder: Encoder) -> None:
        """Test creating exit IDs for early token exiting - normal usage.

        Tests normal usage with full token_exit_cfg.
        """
        B, H, W, T, D = 1, 2, 2, 2, 4
        s2_tokens = torch.zeros(B, H, W, T, D)
        x = TokensOnly(s2_tokens)

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
        x = TokensOnly(s2_tokens)

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

    def test_split_and_expand_per_modality(self) -> None:
        """Test splitting combined tensor back into per-modality tensors."""
        B, D = 2, 4  # Batch size and embedding dimension
        modality_1_channel_groups = 3
        modality_2_channel_groups = 5
        modalities_to_dims_dict = OrderedDict(
            {
                "modality1": (B, 2, 2, 1, modality_1_channel_groups, D),
                "modality2": (B, 1, 1, 2, modality_2_channel_groups, D),
            }
        )

        modality1_data = torch.randn(B, 4 * modality_1_channel_groups, D)
        modality2_data = torch.randn(B, 4 * modality_2_channel_groups, D)

        x = torch.cat([modality1_data, modality2_data], dim=1)

        # Now call the function
        modality_tokens_dict = Encoder.split_and_expand_per_modality(
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


# TODO: write unit tests for applying the Composite encodings

# TODO: write a unit test for the FlexiPatchEmbeddings

"""Integration tests for the model.

Any methods that piece together multiple steps or are the entire forward pass for a module should be here
"""

import pytest
import torch

from helios.nn.model import Encoder, Predictor, TokensAndMasks
from helios.train.masking import MaskedHeliosSample, MaskValue


class TestEncoder:
    """Integration tests for the Encoder class."""

    @pytest.fixture
    def encoder(self) -> Encoder:
        """Create encoder fixture for testing.

        Returns:
            Encoder: Test encoder instance with small test config
        """
        modalities_dict = {"s2": {"rgb": [0, 1, 2], "nir": [3]}}
        return Encoder(
            embedding_size=16,
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

    def test_apply_attn(self, encoder: Encoder) -> None:
        """Test applying attention layers with masking via the apply_attn method.

        Args:
            encoder: Test encoder instance
        """
        s2_tokens = torch.randn(1, 2, 2, 3, 2, 16)
        s2_mask = torch.zeros(1, 2, 2, 3, 2, dtype=torch.long)

        # Mask the first and second "positions" in this 2x2 grid.
        s2_mask[0, 0, 0, 0] = 1  # mask first token
        s2_mask[0, 0, 1, 0] = 1  # mask second token

        # Construct the TokensAndMasks namedtuple with mock modality data + mask.
        x = TokensAndMasks(s2=s2_tokens, s2_mask=s2_mask)

        timestamps = (
            torch.tensor(
                [[15, 7, 2023], [15, 8, 2023], [15, 9, 2023]], dtype=torch.long
            )
            .unsqueeze(0)
            .permute(0, 2, 1)
        )  # [B, 3, T]
        patch_size = 4
        input_res = 10

        output = encoder.apply_attn(
            x=x, timestamps=timestamps, patch_size=patch_size, input_res=input_res
        )

        assert isinstance(
            output, TokensAndMasks
        ), "apply_attn should return a TokensAndMasks object."

        # Ensure shape is preserved in the output tokens.
        assert (
            output.s2.shape == s2_tokens.shape
        ), f"Expected output 's2' shape {s2_tokens.shape}, got {output.s2.shape}."

        # Confirm the mask was preserved and that masked tokens are zeroed out in the output.
        assert (output.s2_mask == s2_mask).all(), "Mask should be preserved in output"
        assert (
            output.s2[s2_mask >= MaskValue.TARGET_ENCODER_ONLY.value] == 0
        ).all(), "Masked tokens should be 0 in output"

    def test_forward_exit_config_none(self, encoder: Encoder) -> None:
        """Test full forward pass without exit configuration.

        In this scenario we do not provide a token exit configuration so that all transformer
        layers are executed normally.

        Args:
            encoder: Test encoder instance
        """
        B, H, W, T, C = 1, 8, 8, 4, 4  # 4 channels: first 3 for 'rgb', last for 'nir'
        num_channel_groups = 2  # "rgb" and "nir"
        s2 = torch.randn(B, H, W, T, C)
        s2_mask = torch.zeros(B, H, W, T, num_channel_groups, dtype=torch.long)
        latlon = torch.randn(B, 2)
        latlon_mask = torch.ones(B, 2, dtype=torch.float32)
        days = torch.randint(0, 25, (B, 1, T), dtype=torch.long)
        months = torch.randint(0, 12, (B, 1, T), dtype=torch.long)
        years = torch.randint(2018, 2020, (B, 1, T), dtype=torch.long)
        timestamps = torch.cat([days, months, years], dim=1)  # Shape: (B, 3, T)

        x = MaskedHeliosSample(s2, s2_mask, latlon, latlon_mask, timestamps)

        patch_size = 4
        input_res = 1

        # No early exit configuration is provided.
        output = encoder.forward(
            x, patch_size, input_res, exit_after_n_layers=None, token_exit_cfg=None
        )

        # After patchification the spatial dimensions reduce.
        expected_H = H // patch_size
        expected_W = W // patch_size
        expected_embedding_size = encoder.embedding_size
        # Expected output shape [B, new_H, new_W, T, num_channel_groups, embedding_size]
        expected_shape = (
            B,
            expected_H,
            expected_W,
            T,
            num_channel_groups,
            expected_embedding_size,
        )
        assert (
            output.s2.shape == expected_shape
        ), f"Expected output s2 shape {expected_shape}, got {output.s2.shape}"

        expected_mask_shape = (B, expected_H, expected_W, T, num_channel_groups)
        assert (
            output.s2_mask.shape == expected_mask_shape
        ), f"Expected output s2_mask shape {expected_mask_shape}, got {output.s2_mask.shape}"

    def test_forward_exit_config_exists(self, encoder: Encoder) -> None:
        """Test full forward pass with a token exit configuration.

        In this scenario (with an exit configuration) we set tokens in each band group to exit early,
        here we set both "rgb" and "nir" to exit at a given transformer layer.

        Args:
            encoder: Test encoder instance
        """
        B, H, W, T, C = 1, 8, 8, 4, 4  # 4 channels: first 3 for 'rgb', last for 'nir'
        num_channel_groups = 2  # two channel groups
        s2 = torch.randn(B, H, W, T, C)
        s2_mask = torch.zeros(B, H, W, T, num_channel_groups, dtype=torch.long)
        latlon = torch.randn(B, 2)
        latlon_mask = torch.ones(B, 2, dtype=torch.float32)
        # Generate valid timestamps with month in [1, 12]
        days = torch.randint(0, 25, (B, 1, T), dtype=torch.long)
        months = torch.randint(0, 12, (B, 1, T), dtype=torch.long)
        years = torch.randint(2018, 2020, (B, 1, T), dtype=torch.long)
        timestamps = torch.cat([days, months, years], dim=1)

        x = MaskedHeliosSample(s2, s2_mask, latlon, latlon_mask, timestamps)

        patch_size = 4
        input_res = 1

        # Token exit configuration: for our modality "s2" with two channel groups "rgb" and "nir",
        # we set "rgb" tokens to exit after block 1 and "nir" tokens to exit at block 0.
        token_exit_cfg = {"rgb": 1, "nir": 0}
        exit_after_n_layers = 1

        output = encoder.forward(
            x,
            patch_size,
            input_res,
            exit_after_n_layers=exit_after_n_layers,
            token_exit_cfg=token_exit_cfg,
        )

        expected_H = H // patch_size
        expected_W = W // patch_size
        expected_embedding_size = encoder.embedding_size
        expected_shape = (
            B,
            expected_H,
            expected_W,
            T,
            num_channel_groups,
            expected_embedding_size,
        )
        assert (
            output.s2.shape == expected_shape
        ), f"Expected output s2 shape {expected_shape}, got {output.s2.shape}"

        expected_mask_shape = (B, expected_H, expected_W, T, num_channel_groups)
        assert (
            output.s2_mask.shape == expected_mask_shape
        ), f"Expected output s2_mask shape {expected_mask_shape}, got {output.s2_mask.shape}"


class TestPredictor:
    """Integration tests for the Predictor class."""

    @pytest.fixture(scope="class")
    def predictor(self) -> Predictor:
        """Create predictor fixture for testing.

        Returns:
            Predictor: Test predictor instance with small test config
        """
        modalities_to_channel_groups_dict = {"s2": {"rgb": [0, 1, 2], "nir": [3]}}
        return Predictor(
            modalities_to_channel_groups_dict=modalities_to_channel_groups_dict,
            encoder_embedding_size=8,
            decoder_embedding_size=16,
            depth=2,
            mlp_ratio=4.0,
            num_heads=2,
            max_sequence_length=12,
            max_patch_size=8,
            drop_path=0.1,
            learnable_channel_embeddings=True,
            output_embedding_size=8,
        )

    def test_predictor_forward(self, predictor: Predictor) -> None:
        """Test the full forward pass of the Predictor."""
        B = 1  # Batch size
        H = 2  # Spatial height
        W = 2  # Spatial width
        T = 3  # Number of timesteps
        num_groups = 2  # Number of channel groups (as defined in modalities_to_channel_groups_dict)

        embedding_dim = predictor.encoder_to_decoder_embed.in_features

        s2_tokens = torch.randn(B, H, W, T, num_groups, embedding_dim)

        s2_mask = torch.full(
            (B, H, W, T, num_groups),
            fill_value=MaskValue.DECODER_ONLY.value,
            dtype=torch.float32,
        )

        # Create dummy latitude and longitude data (and its mask)
        latlon = torch.randn(B, 2)
        latlon_mask = torch.ones(B, 2, dtype=torch.float32)

        encoded_tokens = TokensAndMasks(
            s2=s2_tokens, s2_mask=s2_mask, latlon=latlon, latlon_mask=latlon_mask
        )
        timestamps = torch.tensor(
            [[[1, 15, 30], [6, 7, 8], [2018, 2018, 2018]]],
            dtype=torch.long,
        )

        patch_size = 4
        input_res = 1

        output = predictor.forward(encoded_tokens, timestamps, patch_size, input_res)

        expected_token_shape = (B, H, W, T, num_groups, predictor.output_embedding_size)
        assert (
            output.s2.shape == expected_token_shape
        ), f"Expected tokens shape {expected_token_shape}, got {output.s2.shape}"

        expected_mask_shape = (B, H, W, T, num_groups)
        assert (
            output.s2_mask.shape == expected_mask_shape
        ), f"Expected mask shape {expected_mask_shape}, got {output.s2_mask.shape}"


def test_end_to_end_with_exit_config() -> None:
    """Test the full end to end forward pass of the model with an exit configuration."""
    B, H, W, T, C = 1, 8, 8, 4, 4  # 4 channels: first 3 for 'rgb', last for 'nir'
    num_channel_groups = 2  # "rgb" and "nir"
    # Create dummy s2 data: shape (B, H, W, T, C)
    s2 = torch.randn(B, H, W, T, C)
    # Create a dummy mask for s2 with shape (B, H, W, T, num_channel_groups)
    # Here we assume 0 (ONLINE_ENCODER) means the token is visible.
    s2_mask = torch.zeros(B, H, W, T, num_channel_groups, dtype=torch.long)
    # Dummy latitude-longitude data.
    latlon = torch.randn(B, 2)
    latlon_mask = torch.ones(B, 2, dtype=torch.float32)
    # Generate valid timestamps:
    # - days: range 1..31,
    # - months: range 1..13,
    # - years: e.g. 2018-2019.
    days = torch.randint(1, 31, (B, 1, T), dtype=torch.long)
    months = torch.randint(1, 13, (B, 1, T), dtype=torch.long)
    years = torch.randint(2018, 2020, (B, 1, T), dtype=torch.long)
    timestamps = torch.cat([days, months, years], dim=1)  # Shape: (B, 3, T)

    x = MaskedHeliosSample(s2, s2_mask, latlon, latlon_mask, timestamps)

    patch_size = 4
    input_res = 1

    modalities_to_channel_groups_dict = {"s2": {"rgb": [0, 1, 2], "nir": [3]}}
    # Shared constants for encoder and predictor
    MAX_PATCH_SIZE = 8
    NUM_HEADS = 2
    MLP_RATIO = 4.0
    MAX_SEQ_LENGTH = 12
    DEPTH = 2
    DROP_PATH = 0.1
    ENCODER_EMBEDDING_SIZE = 16
    DECODER_EMBEDDING_SIZE = 16
    encoder = Encoder(
        modalities_to_channel_groups_dict=modalities_to_channel_groups_dict,
        embedding_size=ENCODER_EMBEDDING_SIZE,
        max_patch_size=MAX_PATCH_SIZE,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        max_sequence_length=MAX_SEQ_LENGTH,
        base_patch_size=4,
        use_channel_embs=True,
        depth=DEPTH,
        drop_path=DROP_PATH,
    )
    predictor = Predictor(
        modalities_to_channel_groups_dict=modalities_to_channel_groups_dict,
        encoder_embedding_size=ENCODER_EMBEDDING_SIZE,
        decoder_embedding_size=DECODER_EMBEDDING_SIZE,
        depth=DEPTH,
        mlp_ratio=MLP_RATIO,
        num_heads=NUM_HEADS,
        max_sequence_length=MAX_SEQ_LENGTH,
        max_patch_size=MAX_PATCH_SIZE,
        drop_path=DROP_PATH,
    )
    output = encoder.forward(
        x,
        patch_size,
        input_res,
        exit_after_n_layers=1,
        token_exit_cfg={"rgb": 1, "nir": 0},
    )
    output = predictor.forward(output, timestamps, patch_size, input_res)
    patched_H = H // patch_size
    patched_W = W // patch_size
    assert output.s2.shape == (
        B,
        patched_H,
        patched_W,
        T,
        num_channel_groups,
        predictor.output_embedding_size,
    )
    assert output.s2_mask.shape == (B, patched_H, patched_W, T, num_channel_groups)

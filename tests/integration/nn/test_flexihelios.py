"""Integration tests for the model.

Any methods that piece together multiple steps or are the entire forward pass for a module should be here
"""

import logging

import pytest
import torch
from einops import rearrange

from helios.data.constants import Modality, ModalitySpec
from helios.nn.flexihelios import (
    Encoder,
    FlexiHeliosPatchEmbeddings,
    Predictor,
    TokensAndMasks,
)
from helios.train.masking import MaskedHeliosSample, MaskValue

logger = logging.getLogger(__name__)


@pytest.fixture
def modality_band_set_len_and_total_bands(
    supported_modalities: list[ModalitySpec],
) -> dict[str, tuple[int, int]]:
    """Get the number of band sets and total bands for each modality.

    Returns:
        Dictionary mapping modality name to tuple of (num_band_sets, total_bands)
    """
    return {
        modality.name: (
            len(modality.band_sets),
            modality.num_bands,
        )
        for modality in supported_modalities
    }


@pytest.fixture
def supported_modality_names(supported_modalities: list[ModalitySpec]) -> list[str]:
    """Get the names of the supported modalities."""
    return [modality.name for modality in supported_modalities]


class TestFlexiHeliosPatchEmbeddings:
    """Integration tests for the FlexiHeliosPatchEmbeddings class."""

    @pytest.fixture
    def patch_embeddings(
        self,
    ) -> FlexiHeliosPatchEmbeddings:
        """Create patch embeddings fixture for testing.

        Returns:
            FlexiHeliosPatchEmbeddings: Test patch embeddings instance with small test config
        """
        supported_modality_names = ["sentinel2_l2a", "latlon"]
        return FlexiHeliosPatchEmbeddings(
            supported_modality_names=supported_modality_names,
            embedding_size=16,
            max_patch_size=8,
        )

    def test_forward(
        self,
        patch_embeddings: FlexiHeliosPatchEmbeddings,
        modality_band_set_len_and_total_bands: dict[str, tuple[int, int]],
    ) -> None:
        """Test the forward pass of the patch embeddings."""
        sentinel_2_num_band_sets, sentinel_2_num_bands = (
            modality_band_set_len_and_total_bands["sentinel2_l2a"]
        )
        latlon_num_band_sets, latlon_num_bands = modality_band_set_len_and_total_bands[
            "latlon"
        ]
        B, H, W, T, num_bands = 1, 16, 16, 3, sentinel_2_num_bands
        sentinel2_l2a = torch.randn((B, H, W, T, num_bands))
        sentinel2_l2a_mask = torch.zeros((B, H, W, T, num_bands), dtype=torch.long)
        patch_size = 4

        latlon = torch.randn(B, latlon_num_bands)
        latlon_mask = torch.randint(0, 2, (B, latlon_num_bands), dtype=torch.float32)
        days = torch.randint(0, 25, (B, T, 1), dtype=torch.long)
        months = torch.randint(0, 12, (B, T, 1), dtype=torch.long)
        years = torch.randint(2018, 2020, (B, T, 1), dtype=torch.long)
        timestamps = torch.cat([days, months, years], dim=-1)  # Shape: (B, T, 3)

        masked_sample_dict = {
            "sentinel2_l2a": sentinel2_l2a,
            "sentinel2_l2a_mask": sentinel2_l2a_mask,
            "latlon": latlon,
            "latlon_mask": latlon_mask,
            "timestamps": timestamps,
        }
        sample = MaskedHeliosSample(**masked_sample_dict)
        output = patch_embeddings.forward(sample, patch_size)
        embedding_size = patch_embeddings.embedding_size
        assert output["sentinel2_l2a"].shape == (
            B,
            H // patch_size,
            W // patch_size,
            T,
            sentinel_2_num_band_sets,  # of band sets
            embedding_size,
        )
        assert output["sentinel2_l2a_mask"].shape == (
            B,
            H // patch_size,
            W // patch_size,
            T,
            sentinel_2_num_band_sets,  # of band sets
        )
        assert output["latlon"].shape == (
            B,
            latlon_num_band_sets,
            embedding_size,
        )  # B, C_G , D
        assert output["latlon_mask"].shape == (B, latlon_num_band_sets)  # B, C_G


class TestEncoder:
    """Integration tests for the Encoder class."""

    @pytest.fixture
    def encoder(self, supported_modalities: list[ModalitySpec]) -> Encoder:
        """Create encoder fixture for testing.

        Returns:
            Encoder: Test encoder instance with small test config
        """
        return Encoder(
            embedding_size=16,
            max_patch_size=8,
            min_patch_size=1,
            num_heads=2,
            mlp_ratio=4.0,
            depth=2,
            drop_path=0.1,
            supported_modalities=supported_modalities,
            max_sequence_length=12,
            use_channel_embs=True,
        )

    def test_apply_attn(
        self,
        encoder: Encoder,
        modality_band_set_len_and_total_bands: dict[str, tuple[int, int]],
    ) -> None:
        """Test applying attention layers with masking via the apply_attn method."""
        sentinel2_l2a_num_band_sets, _ = modality_band_set_len_and_total_bands[
            "sentinel2_l2a"
        ]
        latlon_num_band_sets, _ = modality_band_set_len_and_total_bands["latlon"]
        B, H, W, T, C, D = 1, 2, 2, 3, sentinel2_l2a_num_band_sets, 16
        sentinel2_l2a_tokens = torch.randn(B, H, W, T, C, D)
        sentinel2_l2a_mask = torch.zeros(B, H, W, T, C, dtype=torch.long)

        # Mask the first and second "positions" in this 2x2 grid.
        sentinel2_l2a_mask[0, 0, 0, 0] = 1  # mask first token
        sentinel2_l2a_mask[0, 0, 1, 0] = 1  # mask second token
        latlon = torch.randn(B, latlon_num_band_sets, D)
        latlon_mask = torch.randint(
            0, 2, (B, latlon_num_band_sets), dtype=torch.float32
        )

        # Construct the TokensAndMasks namedtuple with mock modality data + mask.
        x = {
            "sentinel2_l2a": sentinel2_l2a_tokens,
            "sentinel2_l2a_mask": sentinel2_l2a_mask,
            "latlon": latlon,
            "latlon_mask": latlon_mask,
        }

        timestamps = torch.tensor(
            [[15, 7, 2023], [15, 8, 2023], [15, 9, 2023]], dtype=torch.long
        ).unsqueeze(0)
        patch_size = 4
        input_res = 10

        output = encoder.apply_attn(
            x=x, timestamps=timestamps, patch_size=patch_size, input_res=input_res
        )

        # Ensure shape is preserved in the output tokens.
        assert (
            output["sentinel2_l2a"].shape == sentinel2_l2a_tokens.shape
        ), f"Expected output 'sentinel2_l2a' shape {sentinel2_l2a_tokens.shape}, got {output['sentinel2_l2a'].shape}."

        # Confirm the mask was preserved and that masked tokens are zeroed out in the output.
        assert (
            output["sentinel2_l2a_mask"] == sentinel2_l2a_mask
        ).all(), "Mask should be preserved in output"
        assert (
            output["sentinel2_l2a"][
                sentinel2_l2a_mask >= MaskValue.TARGET_ENCODER_ONLY.value
            ]
            == 0
        ).all(), "Masked tokens should be 0 in output"

    def test_forward_exit_config_none(
        self,
        encoder: Encoder,
        modality_band_set_len_and_total_bands: dict[str, tuple[int, int]],
    ) -> None:
        """Test full forward pass without exit configuration.

        In this scenario we do not provide a token exit configuration so that all transformer
        layers are executed normally.
        """
        sentinel2_l2a_num_band_sets, sentinel2_l2a_num_bands = (
            modality_band_set_len_and_total_bands["sentinel2_l2a"]
        )
        latlon_num_band_sets, latlon_num_bands = modality_band_set_len_and_total_bands[
            "latlon"
        ]
        B, H, W, T, C = 1, 8, 8, 4, sentinel2_l2a_num_bands
        sentinel2_l2a = torch.randn(B, H, W, T, C)
        sentinel2_l2a_mask = torch.zeros(B, H, W, T, C, dtype=torch.long)
        latlon = torch.randn(B, latlon_num_bands)
        latlon_mask = torch.zeros(B, latlon_num_bands, dtype=torch.float32)
        days = torch.randint(0, 25, (B, T, 1), dtype=torch.long)
        months = torch.randint(0, 12, (B, T, 1), dtype=torch.long)
        years = torch.randint(2018, 2020, (B, T, 1), dtype=torch.long)
        timestamps = torch.cat([days, months, years], dim=-1)  # Shape: (B, T, 3)

        masked_sample_dict = {
            "sentinel2_l2a": sentinel2_l2a,
            "sentinel2_l2a_mask": sentinel2_l2a_mask,
            "latlon": latlon,
            "latlon_mask": latlon_mask,
            "timestamps": timestamps,
        }
        x = MaskedHeliosSample(**masked_sample_dict)

        patch_size = 4
        input_res = 1

        # No early exit configuration is provided.
        output = encoder.forward(x, patch_size, input_res, token_exit_cfg=None)

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
            sentinel2_l2a_num_band_sets,
            expected_embedding_size,
        )
        assert output.sentinel2_l2a is not None
        assert output.sentinel2_l2a_mask is not None
        assert output.latlon is not None
        assert output.latlon_mask is not None
        assert (
            output.sentinel2_l2a.shape == expected_shape
        ), f"Expected output sentinel2_l2a shape {expected_shape}, got {output.sentinel2_l2a.shape}"

        expected_mask_shape = (
            B,
            expected_H,
            expected_W,
            T,
            sentinel2_l2a_num_band_sets,
        )
        assert (
            output.sentinel2_l2a_mask.shape == expected_mask_shape
        ), f"Expected output sentinel2_l2a_mask shape {expected_mask_shape}, got {output.sentinel2_l2a_mask.shape}"
        assert output.latlon.shape == (
            B,
            latlon_num_band_sets,
            expected_embedding_size,
        ), f"Expected output latlon shape {latlon.shape}, got {output.latlon.shape}"
        assert (
            output.latlon_mask.shape
            == (
                B,
                latlon_num_band_sets,
            )
        ), f"Expected output latlon_mask shape {latlon_mask.shape}, got {output.latlon_mask.shape}"

        # test the gradients are correct too
        output.sentinel2_l2a.sum().backward()

        for name, param in encoder.named_parameters():
            # the composite_encodings is a bug which will be fixed now
            if not any(
                ignore_param in name
                for ignore_param in [
                    "pos_embed",
                    "month_embed",
                    "composite_encodings.per_modality_channel_embeddings.latlon",
                ]
            ):
                assert param.grad is not None, name

    def test_forward_exit_config_exists(
        self,
        encoder: Encoder,
        modality_band_set_len_and_total_bands: dict[str, tuple[int, int]],
    ) -> None:
        """Test full forward pass with a token exit configuration.

        In this scenario (with an exit configuration) we set tokens in each modality for early exit.
        """
        sentinel2_l2a_num_band_sets, sentinel2_l2a_num_bands = (
            modality_band_set_len_and_total_bands["sentinel2_l2a"]
        )
        latlon_num_band_sets, latlon_num_bands = modality_band_set_len_and_total_bands[
            "latlon"
        ]
        B, H, W, T, C = 1, 2, 2, 1, sentinel2_l2a_num_bands
        sentinel2_l2a = torch.randn(B, H, W, T, C)
        sentinel2_l2a_mask = torch.zeros(B, H, W, T, C, dtype=torch.long)
        latlon = torch.randn(B, latlon_num_bands)
        latlon_mask = torch.zeros((B, latlon_num_bands), dtype=torch.float32)
        # Generate valid timestamps with month in [1, 12]
        days = torch.randint(0, 25, (B, T, 1), dtype=torch.long)
        months = torch.randint(0, 12, (B, T, 1), dtype=torch.long)
        years = torch.randint(2018, 2020, (B, T, 1), dtype=torch.long)
        timestamps = torch.cat([days, months, years], dim=-1)

        masked_sample_dict = {
            "sentinel2_l2a": sentinel2_l2a,
            "sentinel2_l2a_mask": sentinel2_l2a_mask,
            "latlon": latlon,
            "latlon_mask": latlon_mask,
            "timestamps": timestamps,
        }
        x = MaskedHeliosSample(**masked_sample_dict)

        patch_size = 2
        input_res = 1

        token_exit_cfg = {"sentinel2_l2a": 2, "latlon": 0}

        output = encoder.forward(
            x,
            patch_size,
            input_res,
            token_exit_cfg=token_exit_cfg,
        )

        expected_H = H // patch_size
        expected_W = W // patch_size
        expected_embedding_size = encoder.embedding_size
        expected_shape_sentinel2_l2a = (
            B,
            expected_H,
            expected_W,
            T,
            sentinel2_l2a_num_band_sets,
            expected_embedding_size,
        )
        assert output.sentinel2_l2a is not None
        assert output.sentinel2_l2a_mask is not None
        assert output.latlon is not None
        assert output.latlon_mask is not None
        assert (
            output.sentinel2_l2a.shape == expected_shape_sentinel2_l2a
        ), f"Expected output sentinel2_l2a shape {expected_shape_sentinel2_l2a}, got {output.sentinel2_l2a.shape}"

        expected_mask_shape = (
            B,
            expected_H,
            expected_W,
            T,
            sentinel2_l2a_num_band_sets,
        )
        assert (
            output.sentinel2_l2a_mask.shape == expected_mask_shape
        ), f"Expected output sentinel2_l2a_mask shape {expected_mask_shape}, got {output.sentinel2_l2a_mask.shape}"
        expected_shape_latlon = (
            B,
            latlon_num_band_sets,
            expected_embedding_size,
        )
        assert (
            output.latlon.shape == expected_shape_latlon
        ), f"Expected output latlon shape {expected_shape_latlon}, got {output.latlon.shape}"

        output.sentinel2_l2a.sum().backward()
        for name, param in encoder.named_parameters():
            # the composite_encodings is a bug which will be fixed now
            if not (
                any(
                    ignore_param in name
                    for ignore_param in [
                        "pos_embed",
                        "month_embed",
                        "composite_encodings.per_modality_channel_embeddings.latlon",
                    ]
                )
                or ("block" in name)
            ):
                assert param.grad is not None, name

    def test_entire_modality_masked(
        self,
        encoder: Encoder,
        modality_band_set_len_and_total_bands: dict[str, tuple[int, int]],
    ) -> None:
        """Test that when an entire modality is masked."""
        sentinel2_l2a_num_band_sets, sentinel2_l2a_num_bands = (
            modality_band_set_len_and_total_bands["sentinel2_l2a"]
        )
        latlon_num_band_sets, latlon_num_bands = modality_band_set_len_and_total_bands[
            "latlon"
        ]
        B, H, W, T, C = 1, 8, 8, 4, sentinel2_l2a_num_bands
        sentinel2_l2a = torch.randn(B, H, W, T, C)
        latlon = torch.randn(B, latlon_num_bands)
        # Mask the entirety of each modality
        sentinel2_l2a_mask = torch.ones(B, H, W, T, C, dtype=torch.long)
        # Make 1 token in all S2 channel groups
        sentinel2_l2a_mask[0, 0, 0, 0, :] = 0
        latlon_mask = torch.ones(B, 2, dtype=torch.float32)
        days = torch.randint(0, 25, (B, T, 1), dtype=torch.long)
        months = torch.randint(0, 12, (B, T, 1), dtype=torch.long)
        years = torch.randint(2018, 2020, (B, T, 1), dtype=torch.long)
        timestamps = torch.cat([days, months, years], dim=-1)  # Shape: (B, T, 3)

        masked_sample_dict = {
            "sentinel2_l2a": sentinel2_l2a,
            "sentinel2_l2a_mask": sentinel2_l2a_mask,
            "latlon": latlon,
            "latlon_mask": latlon_mask,
            "timestamps": timestamps,
        }
        x = MaskedHeliosSample(**masked_sample_dict)

        patch_size = 4
        input_res = 1

        output = encoder.forward(x, patch_size, input_res, token_exit_cfg=None)

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
            sentinel2_l2a_num_band_sets,
            expected_embedding_size,
        )
        assert output.sentinel2_l2a is not None
        assert output.sentinel2_l2a_mask is not None
        assert output.latlon is not None
        assert output.latlon_mask is not None
        assert (
            output.sentinel2_l2a.shape == expected_shape
        ), f"Expected output sentinel2_l2a shape {expected_shape}, got {output.sentinel2_l2a.shape}"

        expected_mask_shape = (
            B,
            expected_H,
            expected_W,
            T,
            sentinel2_l2a_num_band_sets,
        )
        assert (
            output.sentinel2_l2a_mask.shape == expected_mask_shape
        ), f"Expected output sentinel2_l2a_mask shape {expected_mask_shape}, got {output.sentinel2_l2a_mask.shape}"
        assert output.latlon.shape == (
            B,
            1,
            expected_embedding_size,
        ), f"Expected output latlon shape {latlon.shape}, got {output.latlon.shape}"
        assert (
            output.latlon_mask.shape
            == (
                B,
                1,
            )
        ), f"Expected output latlon_mask shape {latlon_mask.shape}, got {output.latlon_mask.shape}"

        output.sentinel2_l2a.sum().backward()
        for name, param in encoder.named_parameters():
            # the composite_encodings is a bug which will be fixed now
            if not (
                any(
                    ignore_param in name
                    for ignore_param in [
                        "pos_embed",
                        "month_embed",
                        "composite_encodings.per_modality_channel_embeddings.latlon",
                        "patch_embeddings.per_modality_embeddings.latlon",
                    ]
                )
                or ("block" in name)
            ):
                assert param.grad is not None, name


class TestPredictor:
    """Integration tests for the Predictor class."""

    @pytest.fixture
    def predictor(self, supported_modalities: list[ModalitySpec]) -> Predictor:
        """Create predictor fixture for testing.

        Returns:
            Predictor: Test predictor instance with small test config
        """
        return Predictor(
            supported_modalities=supported_modalities,
            encoder_embedding_size=8,
            decoder_embedding_size=16,
            depth=2,
            mlp_ratio=4.0,
            num_heads=2,
            max_sequence_length=12,
            drop_path=0.1,
            learnable_channel_embeddings=True,
            output_embedding_size=8,
        )

    def test_predictor_forward_masked_out_channels(
        self,
        predictor: Predictor,
        modality_band_set_len_and_total_bands: dict[str, tuple[int, int]],
    ) -> None:
        """Test the full forward pass of the Predictor."""
        B = 1  # Batch size
        H = 2  # Spatial height
        W = 2  # Spatial width
        T = 3  # Number of timesteps
        sentinel2_l2a_num_band_sets, _ = modality_band_set_len_and_total_bands[
            "sentinel2_l2a"
        ]
        latlon_num_band_sets, _ = modality_band_set_len_and_total_bands["latlon"]
        embedding_dim = predictor.encoder_to_decoder_embed.in_features

        sentinel2_l2a_tokens = torch.randn(
            B, H, W, T, sentinel2_l2a_num_band_sets, embedding_dim, requires_grad=False
        )

        sentinel2_l2a_mask = torch.full(
            (B, H, W, T, sentinel2_l2a_num_band_sets),
            fill_value=MaskValue.DECODER.value,
            dtype=torch.float32,
        )
        sentinel2_l2a_mask[:, :, :, :, 0] = MaskValue.ONLINE_ENCODER.value
        # Create dummy latitude and longitude data (and its mask)
        latlon = torch.randn(
            B, latlon_num_band_sets, embedding_dim, requires_grad=False
        )
        latlon_mask = torch.zeros(B, latlon_num_band_sets, dtype=torch.float32)

        encoded_tokens = TokensAndMasks(
            sentinel2_l2a=sentinel2_l2a_tokens,
            sentinel2_l2a_mask=sentinel2_l2a_mask,
            latlon=latlon,
            latlon_mask=latlon_mask,
        )
        timestamps = rearrange(
            torch.tensor(
                [[[1, 15, 30], [6, 7, 8], [2018, 2018, 2018]]],
                dtype=torch.long,
            ),
            "b d t -> b t d",
        )

        patch_size = 4
        input_res = 1

        output = predictor.forward(encoded_tokens, timestamps, patch_size, input_res)

        expected_token_shape = (
            B,
            H,
            W,
            T,
            sentinel2_l2a_num_band_sets,
            predictor.output_embedding_size,
        )
        assert output.sentinel2_l2a is not None
        assert output.sentinel2_l2a_mask is not None
        assert output.latlon is not None
        assert output.latlon_mask is not None
        assert (
            output.sentinel2_l2a.shape == expected_token_shape
        ), f"Expected tokens shape {expected_token_shape}, got {output.sentinel2_l2a.shape}"

        expected_mask_shape = (B, H, W, T, sentinel2_l2a_num_band_sets)
        assert (
            output.sentinel2_l2a_mask.shape == expected_mask_shape
        ), f"Expected mask shape {expected_mask_shape}, got {output.sentinel2_l2a_mask.shape}"
        assert output.latlon.shape == (
            B,
            latlon_num_band_sets,
            predictor.output_embedding_size,
        )
        assert output.latlon_mask.shape == (B, latlon_num_band_sets)
        output.sentinel2_l2a.sum().backward()
        for name, param in predictor.named_parameters():
            if not any(
                x in name
                for x in [
                    "pos_embed",
                    "month_embed",
                    "composite_encodings.per_modality_channel_embeddings.latlon",
                ]
            ):
                assert param.grad is not None, name

    def test_predictor_forward(
        self,
        predictor: Predictor,
        modality_band_set_len_and_total_bands: dict[str, tuple[int, int]],
    ) -> None:
        """Test the full forward pass of the Predictor."""
        B = 1  # Batch size
        H = 2  # Spatial height
        W = 2  # Spatial width
        T = 3  # Number of timesteps
        sentinel2_l2a_num_band_sets, _ = modality_band_set_len_and_total_bands[
            "sentinel2_l2a"
        ]
        latlon_num_band_sets, _ = modality_band_set_len_and_total_bands["latlon"]
        embedding_dim = predictor.encoder_to_decoder_embed.in_features

        sentinel2_l2a_tokens = torch.randn(
            B, H, W, T, sentinel2_l2a_num_band_sets, embedding_dim, requires_grad=True
        )

        sentinel2_l2a_mask = torch.full(
            (B, H, W, T, sentinel2_l2a_num_band_sets),
            fill_value=MaskValue.DECODER.value,
            dtype=torch.float32,
        )
        # Create dummy latitude and longitude data (and its mask)
        latlon = torch.randn(B, latlon_num_band_sets, embedding_dim, requires_grad=True)
        latlon_mask = torch.full(
            (B, latlon_num_band_sets),
            fill_value=MaskValue.DECODER.value,
            dtype=torch.float32,
        )

        encoded_tokens = TokensAndMasks(
            sentinel2_l2a=sentinel2_l2a_tokens,
            sentinel2_l2a_mask=sentinel2_l2a_mask,
            latlon=latlon,
            latlon_mask=latlon_mask,
        )
        timestamps = rearrange(
            torch.tensor(
                [[[1, 15, 30], [6, 7, 8], [2018, 2018, 2018]]],
                dtype=torch.long,
            ),
            "b d t -> b t d",
        )

        patch_size = 4
        input_res = 1

        output = predictor.forward(encoded_tokens, timestamps, patch_size, input_res)

        expected_token_shape = (
            B,
            H,
            W,
            T,
            sentinel2_l2a_num_band_sets,
            predictor.output_embedding_size,
        )
        assert output.sentinel2_l2a is not None
        assert output.sentinel2_l2a_mask is not None
        assert output.latlon is not None
        assert output.latlon_mask is not None
        assert (
            output.sentinel2_l2a.shape == expected_token_shape
        ), f"Expected tokens shape {expected_token_shape}, got {output.sentinel2_l2a.shape}"

        expected_mask_shape = (B, H, W, T, sentinel2_l2a_num_band_sets)
        assert (
            output.sentinel2_l2a_mask.shape == expected_mask_shape
        ), f"Expected mask shape {expected_mask_shape}, got {output.sentinel2_l2a_mask.shape}"
        assert output.latlon.shape == (
            B,
            latlon_num_band_sets,
            predictor.output_embedding_size,
        )
        assert output.latlon_mask.shape == (B, latlon_num_band_sets)
        output.sentinel2_l2a.sum().backward()
        for name, param in predictor.named_parameters():
            if not any(
                x in name
                for x in [
                    "pos_embed",
                    "month_embed",
                    "composite_encodings.per_modality_channel_embeddings.latlon",
                ]
            ):
                assert param.grad is not None, name


def test_end_to_end_with_exit_config(
    modality_band_set_len_and_total_bands: dict[str, tuple[int, int]],
    masked_sample_dict: dict[str, torch.Tensor],
) -> None:
    """Test the full end to end forward pass of the model with an exit configuration."""
    supported_modalities = [
        Modality.SENTINEL2_L2A,
        Modality.LATLON,
        Modality.WORLDCOVER,
    ]
    token_exit_cfg = {"sentinel2_l2a": 3, "latlon": 0, "worldcover": 0}
    sentinel2_l2a_num_band_sets = modality_band_set_len_and_total_bands[
        "sentinel2_l2a"
    ][0]
    latlon_num_band_sets = modality_band_set_len_and_total_bands["latlon"][0]
    B, H, W, T, _ = masked_sample_dict["sentinel2_l2a"].shape
    x = MaskedHeliosSample(**masked_sample_dict)

    patch_size = 4
    input_res = 1
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
        supported_modalities=supported_modalities,
        embedding_size=ENCODER_EMBEDDING_SIZE,
        max_patch_size=MAX_PATCH_SIZE,
        min_patch_size=1,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        max_sequence_length=MAX_SEQ_LENGTH,
        use_channel_embs=True,
        depth=DEPTH,
        drop_path=DROP_PATH,
    )
    predictor = Predictor(
        supported_modalities=supported_modalities,
        encoder_embedding_size=ENCODER_EMBEDDING_SIZE,
        decoder_embedding_size=DECODER_EMBEDDING_SIZE,
        depth=DEPTH,
        mlp_ratio=MLP_RATIO,
        num_heads=NUM_HEADS,
        max_sequence_length=MAX_SEQ_LENGTH,
        drop_path=DROP_PATH,
        learnable_channel_embeddings=True,
    )
    output = encoder.forward(
        x,
        patch_size,
        input_res,
        token_exit_cfg=token_exit_cfg,
    )
    output = predictor.forward(output, x.timestamps, patch_size, input_res)
    patched_H = H // patch_size
    patched_W = W // patch_size
    assert output.sentinel2_l2a is not None
    assert output.sentinel2_l2a_mask is not None
    assert output.latlon is not None
    assert output.latlon_mask is not None
    assert output.sentinel2_l2a.shape == (
        B,
        patched_H,
        patched_W,
        T,
        sentinel2_l2a_num_band_sets,
        predictor.output_embedding_size,
    )
    assert output.sentinel2_l2a_mask.shape == (
        B,
        patched_H,
        patched_W,
        T,
        sentinel2_l2a_num_band_sets,
    )
    assert output.latlon.shape == (
        B,
        latlon_num_band_sets,
        predictor.output_embedding_size,
    )
    assert output.latlon_mask.shape == (
        B,
        latlon_num_band_sets,
    )
    assert output.worldcover is not None
    assert output.worldcover_mask is not None
    assert output.worldcover.shape == (
        B,
        patched_H,
        patched_W,
        1,
        1,
        predictor.output_embedding_size,
    )
    assert output.worldcover_mask.shape == (
        B,
        patched_H,
        patched_W,
        1,
        1,
    )
    output.worldcover.sum().backward()
    for name, param in predictor.named_parameters():
        if not any(
            x in name
            for x in [
                "pos_embed",
                "month_embed",
                "composite_encodings.per_modality_channel_embeddings.latlon",
            ]
        ):
            assert param.grad is not None, name

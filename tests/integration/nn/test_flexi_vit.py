"""Integration tests for the model.

Any methods that piece together multiple steps or are the entire forward pass for a module should be here
"""

import logging

import pytest
import torch
from einops import rearrange

from olmoearth_pretrain.data.constants import Modality, ModalitySpec
from olmoearth_pretrain.nn.flexi_vit import (
    Encoder,
    MultiModalPatchEmbeddings,
    Predictor,
    TokensAndMasks,
)
from olmoearth_pretrain.nn.utils import unpack_encoder_output
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample, MaskValue

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


class TestMultiModalPatchEmbeddings:
    """Integration tests for the MultiModalPatchEmbeddings class."""

    @pytest.fixture
    def patch_embeddings(
        self,
    ) -> MultiModalPatchEmbeddings:
        """Create patch embeddings fixture for testing.

        Returns:
            MultiModalPatchEmbeddings: Test patch embeddings instance with small test config
        """
        supported_modality_names = ["sentinel2_l2a", "latlon"]
        return MultiModalPatchEmbeddings(
            supported_modality_names=supported_modality_names,
            embedding_size=16,
            max_patch_size=8,
        )

    def test_forward(
        self,
        patch_embeddings: MultiModalPatchEmbeddings,
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
        sample = MaskedOlmoEarthSample(**masked_sample_dict)
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

    def test_forward_with_missing_modalities(
        self,
        patch_embeddings: MultiModalPatchEmbeddings,
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
        # ones means it should all be masked out from the perspective of the patch embeddings
        latlon_mask = torch.ones((B, latlon_num_bands), dtype=torch.float32)
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
        sample = MaskedOlmoEarthSample(**masked_sample_dict)
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
        assert (output["latlon_mask"] == 1).all()
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

        for fast_pass in [True, False]:
            output, _, _ = encoder.apply_attn(
                x=x,
                timestamps=timestamps,
                patch_size=patch_size,
                input_res=input_res,
                fast_pass=fast_pass,
            )

            # Ensure shape is preserved in the output tokens.
            assert output["sentinel2_l2a"].shape == sentinel2_l2a_tokens.shape, (
                f"Expected output 'sentinel2_l2a' shape {sentinel2_l2a_tokens.shape}, got {output['sentinel2_l2a'].shape}."
            )

            # Confirm the mask was preserved and that masked tokens are zeroed out in the output.
            assert (output["sentinel2_l2a_mask"] == sentinel2_l2a_mask).all(), (
                "Mask should be preserved in output"
            )
            if fast_pass:
                assert (
                    output["sentinel2_l2a"][
                        sentinel2_l2a_mask >= MaskValue.TARGET_ENCODER_ONLY.value
                    ]
                    != 0
                ).all(), (
                    "Masked tokens should not be 0 in output because mask is overridden"
                )
            else:
                assert (
                    output["sentinel2_l2a"][
                        sentinel2_l2a_mask >= MaskValue.TARGET_ENCODER_ONLY.value
                    ]
                    == 0
                ).all(), (
                    "Masked tokens should be 0 in output because mask is not overridden"
                )

    @torch.inference_mode()
    def test_fast_pass_is_the_same_as_no_masks(
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
        sentinel2_l2a_mask = torch.full(
            (B, H, W, T, C), MaskValue.ONLINE_ENCODER.value, dtype=torch.long
        )

        latlon = torch.randn(B, latlon_num_band_sets, D)
        latlon_mask = torch.full(
            (B, latlon_num_band_sets),
            MaskValue.ONLINE_ENCODER.value,
            dtype=torch.float32,
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

        encoder.eval()
        outputs = []
        for fast_pass in [True, False]:
            output, _, _ = encoder.apply_attn(
                x=x,
                timestamps=timestamps,
                patch_size=patch_size,
                input_res=input_res,
                fast_pass=fast_pass,
            )
            outputs.append(output)

        assert torch.allclose(outputs[0]["sentinel2_l2a"], outputs[1]["sentinel2_l2a"])
        assert torch.allclose(outputs[0]["latlon"], outputs[1]["latlon"])

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
        x = MaskedOlmoEarthSample(**masked_sample_dict)

        patch_size = 4
        input_res = 1

        # No early exit configuration is provided.
        output_dict = encoder.forward(x, patch_size, input_res, token_exit_cfg=None)
        output, _, _ = unpack_encoder_output(output_dict)

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
        assert output.sentinel2_l2a.shape == expected_shape, (
            f"Expected output sentinel2_l2a shape {expected_shape}, got {output.sentinel2_l2a.shape}"
        )

        expected_mask_shape = (
            B,
            expected_H,
            expected_W,
            T,
            sentinel2_l2a_num_band_sets,
        )
        assert output.sentinel2_l2a_mask.shape == expected_mask_shape, (
            f"Expected output sentinel2_l2a_mask shape {expected_mask_shape}, got {output.sentinel2_l2a_mask.shape}"
        )
        assert output.latlon.shape == (
            B,
            latlon_num_band_sets,
            expected_embedding_size,
        ), f"Expected output latlon shape {latlon.shape}, got {output.latlon.shape}"
        assert output.latlon_mask.shape == (
            B,
            latlon_num_band_sets,
        ), (
            f"Expected output latlon_mask shape {latlon_mask.shape}, got {output.latlon_mask.shape}"
        )

        # test the gradients are correct too
        output.sentinel2_l2a.sum().backward()

        for name, param in encoder.named_parameters():
            # the composite_encodings is a bug which will be fixed now
            if not any(
                ignore_param in name
                for ignore_param in [
                    "project_and_aggregate",
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
        x = MaskedOlmoEarthSample(**masked_sample_dict)

        patch_size = 2
        input_res = 1

        token_exit_cfg = {"sentinel2_l2a": 2, "latlon": 0}

        output_dict = encoder.forward(
            x,
            patch_size,
            input_res,
            token_exit_cfg=token_exit_cfg,
        )
        output, _, _ = unpack_encoder_output(output_dict)
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
        assert output.sentinel2_l2a.shape == expected_shape_sentinel2_l2a, (
            f"Expected output sentinel2_l2a shape {expected_shape_sentinel2_l2a}, got {output.sentinel2_l2a.shape}"
        )

        expected_mask_shape = (
            B,
            expected_H,
            expected_W,
            T,
            sentinel2_l2a_num_band_sets,
        )
        assert output.sentinel2_l2a_mask.shape == expected_mask_shape, (
            f"Expected output sentinel2_l2a_mask shape {expected_mask_shape}, got {output.sentinel2_l2a_mask.shape}"
        )
        expected_shape_latlon = (
            B,
            latlon_num_band_sets,
            expected_embedding_size,
        )
        assert output.latlon.shape == expected_shape_latlon, (
            f"Expected output latlon shape {expected_shape_latlon}, got {output.latlon.shape}"
        )

        output.sentinel2_l2a.sum().backward()
        for name, param in encoder.named_parameters():
            # the composite_encodings is a bug which will be fixed now
            if not (
                any(
                    ignore_param in name
                    for ignore_param in [
                        "project_and_aggregate",
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
        x = MaskedOlmoEarthSample(**masked_sample_dict)

        patch_size = 4
        input_res = 1

        output_dict = encoder.forward(x, patch_size, input_res, token_exit_cfg=None)
        output, _, _ = unpack_encoder_output(output_dict)

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
        assert output.sentinel2_l2a.shape == expected_shape, (
            f"Expected output sentinel2_l2a shape {expected_shape}, got {output.sentinel2_l2a.shape}"
        )

        expected_mask_shape = (
            B,
            expected_H,
            expected_W,
            T,
            sentinel2_l2a_num_band_sets,
        )
        assert output.sentinel2_l2a_mask.shape == expected_mask_shape, (
            f"Expected output sentinel2_l2a_mask shape {expected_mask_shape}, got {output.sentinel2_l2a_mask.shape}"
        )
        assert output.latlon.shape == (
            B,
            1,
            expected_embedding_size,
        ), f"Expected output latlon shape {latlon.shape}, got {output.latlon.shape}"
        assert output.latlon_mask.shape == (
            B,
            1,
        ), (
            f"Expected output latlon_mask shape {latlon_mask.shape}, got {output.latlon_mask.shape}"
        )

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
                        "project_and_aggregate",
                    ]
                )
                or ("block" in name)
            ):
                assert param.grad is not None, name

    def test_inference_fast_pass(
        self,
        encoder: Encoder,
        modality_band_set_len_and_total_bands: dict[str, tuple[int, int]],
    ) -> None:
        """Test the inference fast pass of the Encoder."""
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
        x = MaskedOlmoEarthSample(**masked_sample_dict)

        patch_size = 4
        input_res = 1

        # No early exit configuration is provided.
        with torch.inference_mode():
            output_dict = encoder.forward(
                x,
                patch_size,
                input_res,
                token_exit_cfg=None,
                fast_pass=True,
            )
        output, _, _ = unpack_encoder_output(output_dict)

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
        assert output.sentinel2_l2a.shape == expected_shape, (
            f"Expected output sentinel2_l2a shape {expected_shape}, got {output.sentinel2_l2a.shape}"
        )

        expected_mask_shape = (
            B,
            expected_H,
            expected_W,
            T,
            sentinel2_l2a_num_band_sets,
        )
        assert output.sentinel2_l2a_mask.shape == expected_mask_shape, (
            f"Expected output sentinel2_l2a_mask shape {expected_mask_shape}, got {output.sentinel2_l2a_mask.shape}"
        )
        assert output.latlon.shape == (
            B,
            latlon_num_band_sets,
            expected_embedding_size,
        ), f"Expected output latlon shape {latlon.shape}, got {output.latlon.shape}"
        assert output.latlon_mask.shape == (
            B,
            latlon_num_band_sets,
        ), (
            f"Expected output latlon_mask shape {latlon_mask.shape}, got {output.latlon_mask.shape}"
        )

    def test_output_embedding_size(
        self,
        supported_modalities: list[ModalitySpec],
        modality_band_set_len_and_total_bands: dict[str, tuple[int, int]],
    ) -> None:
        """Test that output_embedding_size projects tokens to the correct size."""
        # Temporarily disable deterministic mode for this test due to scatter_reduce
        torch.use_deterministic_algorithms(False)

        embedding_size = 16
        output_embedding_size = 32

        encoder = Encoder(
            embedding_size=embedding_size,
            max_patch_size=8,
            min_patch_size=1,
            num_heads=2,
            mlp_ratio=4.0,
            depth=2,
            drop_path=0.1,
            supported_modalities=supported_modalities,
            max_sequence_length=12,
            output_embedding_size=output_embedding_size,
        )

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
        timestamps = torch.cat([days, months, years], dim=-1)

        masked_sample_dict = {
            "sentinel2_l2a": sentinel2_l2a,
            "sentinel2_l2a_mask": sentinel2_l2a_mask,
            "latlon": latlon,
            "latlon_mask": latlon_mask,
            "timestamps": timestamps,
        }
        x = MaskedOlmoEarthSample(**masked_sample_dict)

        patch_size = 4
        input_res = 1

        output_dict = encoder.forward(x, patch_size, input_res, token_exit_cfg=None)
        output, project_aggregated, _ = unpack_encoder_output(output_dict)

        # Verify tokens have output_embedding_size, not embedding_size
        expected_H = H // patch_size
        expected_W = W // patch_size
        expected_shape = (
            B,
            expected_H,
            expected_W,
            T,
            sentinel2_l2a_num_band_sets,
            output_embedding_size,
        )
        assert output.sentinel2_l2a is not None
        assert output.sentinel2_l2a.shape == expected_shape, (
            f"Expected sentinel2_l2a shape {expected_shape}, got {output.sentinel2_l2a.shape}"
        )

        assert output.latlon is not None
        assert output.latlon.shape == (
            B,
            latlon_num_band_sets,
            output_embedding_size,
        ), (
            f"Expected latlon shape (B, {latlon_num_band_sets}, {output_embedding_size}), "
            f"got {output.latlon.shape}"
        )

        # Verify project_aggregated also has output_embedding_size
        assert project_aggregated is not None
        assert project_aggregated.shape == (B, output_embedding_size), (
            f"Expected project_aggregated shape (B, {output_embedding_size}), "
            f"got {project_aggregated.shape}"
        )


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
        assert output.sentinel2_l2a.shape == expected_token_shape, (
            f"Expected tokens shape {expected_token_shape}, got {output.sentinel2_l2a.shape}"
        )

        expected_mask_shape = (B, H, W, T, sentinel2_l2a_num_band_sets)
        assert output.sentinel2_l2a_mask.shape == expected_mask_shape, (
            f"Expected mask shape {expected_mask_shape}, got {output.sentinel2_l2a_mask.shape}"
        )
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
        assert output.sentinel2_l2a.shape == expected_token_shape, (
            f"Expected tokens shape {expected_token_shape}, got {output.sentinel2_l2a.shape}"
        )

        expected_mask_shape = (B, H, W, T, sentinel2_l2a_num_band_sets)
        assert output.sentinel2_l2a_mask.shape == expected_mask_shape, (
            f"Expected mask shape {expected_mask_shape}, got {output.sentinel2_l2a_mask.shape}"
        )
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
                    "project_and_aggregate",
                ]
            ):
                assert param.grad is not None, name


def test_encoder_rope_dynamic_patch_sizes(
    modality_band_set_len_and_total_bands: dict[str, tuple[int, int]],
) -> None:
    """2D RoPE should use runtime patch grid/size, not a fixed spatial table."""
    supported_modalities = [Modality.SENTINEL2_L2A, Modality.LATLON]
    sentinel2_l2a_num_bands = modality_band_set_len_and_total_bands["sentinel2_l2a"][1]
    latlon_num_bands = modality_band_set_len_and_total_bands["latlon"][1]
    encoder = Encoder(
        supported_modalities=supported_modalities,
        embedding_size=16,
        max_patch_size=4,
        min_patch_size=1,
        num_heads=2,
        mlp_ratio=2.0,
        max_sequence_length=12,
        depth=2,
        drop_path=0.0,
        spatial_pos_encoding="rope",
    )

    B, H, W, T = 1, 8, 8, 2
    timestamps = torch.tensor([[[1, 0, 2020], [2, 1, 2020]]], dtype=torch.long)
    for patch_size in (2, 4):
        sample = MaskedOlmoEarthSample(
            sentinel2_l2a=torch.randn(B, H, W, T, sentinel2_l2a_num_bands),
            sentinel2_l2a_mask=torch.zeros(
                B, H, W, T, sentinel2_l2a_num_bands, dtype=torch.long
            ),
            latlon=torch.randn(B, latlon_num_bands),
            latlon_mask=torch.zeros(B, latlon_num_bands, dtype=torch.long),
            timestamps=timestamps,
        )
        encoder.zero_grad()
        output_dict = encoder.forward(sample, patch_size=patch_size, input_res=10)
        output, _, _ = unpack_encoder_output(output_dict)

        assert output.sentinel2_l2a is not None
        assert output.sentinel2_l2a.shape[:3] == (
            B,
            H // patch_size,
            W // patch_size,
        )
        output.sentinel2_l2a.sum().backward()
        assert encoder.blocks[0].attn.q.weight.grad is not None


def test_encoder_register_bottleneck(
    modality_band_set_len_and_total_bands: dict[str, tuple[int, int]],
) -> None:
    """The Perceiver-style register bottleneck returns a fixed grid, decoupled from input size."""
    supported_modalities = [Modality.SENTINEL2_L2A, Modality.LATLON]
    sentinel2_l2a_num_bands = modality_band_set_len_and_total_bands["sentinel2_l2a"][1]
    latlon_num_bands = modality_band_set_len_and_total_bands["latlon"][1]
    grid_size, register_dim = 3, 8
    encoder = Encoder(
        supported_modalities=supported_modalities,
        embedding_size=16,
        max_patch_size=4,
        min_patch_size=1,
        num_heads=2,
        mlp_ratio=2.0,
        max_sequence_length=12,
        depth=2,
        drop_path=0.0,
        spatial_pos_encoding="rope",
        use_register_bottleneck=True,
        register_grid_size=grid_size,
        register_dim=register_dim,
        register_read_depth=1,
        register_latent_depth=2,
    )

    B, H, W, T = 2, 8, 8, 2
    timestamps = torch.tensor(
        [[[1, 0, 2020], [2, 1, 2020]], [[1, 0, 2020], [2, 1, 2020]]], dtype=torch.long
    )
    prev_max_coord = None
    for patch_size in (2, 4):
        s2_mask = torch.zeros(B, H, W, T, sentinel2_l2a_num_bands, dtype=torch.long)
        # Mark the top half decode-only so the read must exclude them via the mask.
        s2_mask[:, : H // 2] = MaskValue.DECODER.value
        sample = MaskedOlmoEarthSample(
            sentinel2_l2a=torch.randn(B, H, W, T, sentinel2_l2a_num_bands),
            sentinel2_l2a_mask=s2_mask,
            latlon=torch.randn(B, latlon_num_bands),
            latlon_mask=torch.zeros(B, latlon_num_bands, dtype=torch.long),
            timestamps=timestamps,
        )
        encoder.zero_grad()
        output_dict = encoder.forward(sample, patch_size=patch_size, input_res=10)

        registers = output_dict["registers"]
        register_positions = output_dict["register_positions"]
        # Register count is fixed regardless of patch grid; width is the bottleneck dim.
        assert registers.shape == (B, grid_size * grid_size, register_dim)
        assert register_positions.shape == (B, grid_size * grid_size, 2)
        # With a register bottleneck, project_and_aggregate pools the register tokens
        # (only), so the contrastive projection is sized to register_dim.
        project_aggregated = output_dict["project_aggregated"]
        assert project_aggregated.shape == (B, register_dim)
        # Coordinates are anchored and rescale with the input extent (variable input size).
        assert register_positions.amax() > 0
        if prev_max_coord is not None:
            assert register_positions.amax().item() != prev_max_coord
        prev_max_coord = register_positions.amax().item()

        registers.sum().backward()
        assert encoder.register_bottleneck is not None
        assert encoder.register_bottleneck.registers.grad is not None
        assert encoder.blocks[0].attn.q.weight.grad is not None


def test_encoder_register_bottleneck_dynamic_grid(
    modality_band_set_len_and_total_bands: dict[str, tuple[int, int]],
) -> None:
    """register_grid_size=None clones a single latent across the (dynamic) patch grid."""
    supported_modalities = [Modality.SENTINEL2_L2A, Modality.LATLON]
    sentinel2_l2a_num_bands = modality_band_set_len_and_total_bands["sentinel2_l2a"][1]
    latlon_num_bands = modality_band_set_len_and_total_bands["latlon"][1]
    register_dim = 8
    encoder = Encoder(
        supported_modalities=supported_modalities,
        embedding_size=16,
        max_patch_size=4,
        min_patch_size=1,
        num_heads=2,
        mlp_ratio=2.0,
        max_sequence_length=12,
        depth=2,
        drop_path=0.0,
        spatial_pos_encoding="rope",
        use_register_bottleneck=True,
        register_grid_size=None,
        register_dim=register_dim,
        register_read_depth=1,
        register_latent_depth=2,
    )
    # Single shared latent, not a per-cell grid of parameters.
    assert encoder.register_bottleneck is not None
    assert encoder.register_bottleneck.dynamic_grid
    assert encoder.register_bottleneck.register.shape == (1, register_dim)
    assert not hasattr(encoder.register_bottleneck, "registers")

    B, H, W, T = 2, 8, 8, 2
    timestamps = torch.tensor(
        [[[1, 0, 2020], [2, 1, 2020]], [[1, 0, 2020], [2, 1, 2020]]], dtype=torch.long
    )
    for patch_size in (2, 4):
        sample = MaskedOlmoEarthSample(
            sentinel2_l2a=torch.randn(B, H, W, T, sentinel2_l2a_num_bands),
            sentinel2_l2a_mask=torch.zeros(
                B, H, W, T, sentinel2_l2a_num_bands, dtype=torch.long
            ),
            latlon=torch.randn(B, latlon_num_bands),
            latlon_mask=torch.zeros(B, latlon_num_bands, dtype=torch.long),
            timestamps=timestamps,
        )
        encoder.zero_grad()
        output_dict = encoder.forward(sample, patch_size=patch_size, input_res=10)

        # The register grid tracks the patch grid (H//patch_size) instead of being fixed.
        expected_side = H // patch_size
        n_reg = expected_side * expected_side
        assert encoder.register_bottleneck.register_grid == (
            expected_side,
            expected_side,
        )
        assert output_dict["registers"].shape == (B, n_reg, register_dim)
        assert output_dict["register_positions"].shape == (B, n_reg, 2)

        output_dict["registers"].sum().backward()
        assert encoder.register_bottleneck.register.grad is not None


def test_encoder_register_bottleneck_interleave(
    modality_band_set_len_and_total_bands: dict[str, tuple[int, int]],
) -> None:
    """register_interleave pairs one read with each latent self-attention block."""
    supported_modalities = [Modality.SENTINEL2_L2A, Modality.LATLON]
    sentinel2_l2a_num_bands = modality_band_set_len_and_total_bands["sentinel2_l2a"][1]
    latlon_num_bands = modality_band_set_len_and_total_bands["latlon"][1]
    grid_size, register_dim, latent_depth = 3, 8, 3
    encoder = Encoder(
        supported_modalities=supported_modalities,
        embedding_size=16,
        max_patch_size=4,
        min_patch_size=1,
        num_heads=2,
        mlp_ratio=2.0,
        max_sequence_length=12,
        depth=2,
        drop_path=0.0,
        spatial_pos_encoding="rope",
        use_register_bottleneck=True,
        register_grid_size=grid_size,
        register_dim=register_dim,
        register_read_depth=1,
        register_latent_depth=latent_depth,
        register_interleave=True,
    )
    bottleneck = encoder.register_bottleneck
    assert bottleneck is not None
    assert bottleneck.interleave
    # One read per latent self-attention block (read_depth is ignored when interleaving).
    assert len(bottleneck.read_blocks) == latent_depth
    assert len(bottleneck.latent_blocks) == latent_depth

    B, H, W, T = 2, 8, 8, 2
    timestamps = torch.tensor(
        [[[1, 0, 2020], [2, 1, 2020]], [[1, 0, 2020], [2, 1, 2020]]], dtype=torch.long
    )
    sample = MaskedOlmoEarthSample(
        sentinel2_l2a=torch.randn(B, H, W, T, sentinel2_l2a_num_bands),
        sentinel2_l2a_mask=torch.zeros(
            B, H, W, T, sentinel2_l2a_num_bands, dtype=torch.long
        ),
        latlon=torch.randn(B, latlon_num_bands),
        latlon_mask=torch.zeros(B, latlon_num_bands, dtype=torch.long),
        timestamps=timestamps,
    )
    output_dict = encoder.forward(sample, patch_size=2, input_res=10)
    assert output_dict["registers"].shape == (B, grid_size * grid_size, register_dim)
    output_dict["registers"].sum().backward()
    # Gradients reach the last interleaved read (only reached if reads run between selves).
    assert bottleneck.read_blocks[-1].attn.q.weight.grad is not None


def test_encoder_register_bottleneck_multi_depth(
    modality_band_set_len_and_total_bands: dict[str, tuple[int, int]],
) -> None:
    """register_read_layers reads a different encoder depth at each interleaved step."""
    supported_modalities = [Modality.SENTINEL2_L2A, Modality.LATLON]
    sentinel2_l2a_num_bands = modality_band_set_len_and_total_bands["sentinel2_l2a"][1]
    latlon_num_bands = modality_band_set_len_and_total_bands["latlon"][1]
    grid_size, register_dim, depth = 3, 8, 4
    # Read only from depth 2 (an intermediate layer): blocks AFTER depth 2 then have no
    # path to the registers, which pins down that the read sources the intermediate layer
    # rather than the final one.
    read_layers = [2]
    encoder = Encoder(
        supported_modalities=supported_modalities,
        embedding_size=16,
        max_patch_size=4,
        min_patch_size=1,
        num_heads=2,
        mlp_ratio=2.0,
        max_sequence_length=12,
        depth=depth,
        drop_path=0.0,
        spatial_pos_encoding="rope",
        use_register_bottleneck=True,
        register_grid_size=grid_size,
        register_dim=register_dim,
        # read_depth / latent_depth are overridden by read_layers.
        register_read_depth=1,
        register_latent_depth=99,
        register_read_layers=read_layers,
    )
    bottleneck = encoder.register_bottleneck
    assert bottleneck is not None
    assert bottleneck.multi_depth
    assert bottleneck.interleave  # multi-depth forces the interleaved schedule
    assert bottleneck.read_layers == read_layers
    # One read + one latent block per source layer (read_depth / latent_depth ignored).
    assert len(bottleneck.read_blocks) == len(read_layers)
    assert len(bottleneck.latent_blocks) == len(read_layers)

    # Also build a stride-2 encoder to confirm the read/latent counts track len(read_layers).
    stride2 = Encoder(
        supported_modalities=supported_modalities,
        embedding_size=16,
        max_patch_size=4,
        min_patch_size=1,
        num_heads=2,
        mlp_ratio=2.0,
        max_sequence_length=12,
        depth=depth,
        drop_path=0.0,
        spatial_pos_encoding="rope",
        use_register_bottleneck=True,
        register_grid_size=grid_size,
        register_dim=register_dim,
        register_read_layers=[2, 4],
    )
    assert stride2.register_bottleneck is not None
    assert len(stride2.register_bottleneck.read_blocks) == 2
    assert len(stride2.register_bottleneck.latent_blocks) == 2

    B, H, W, T = 2, 8, 8, 2
    timestamps = torch.tensor(
        [[[1, 0, 2020], [2, 1, 2020]], [[1, 0, 2020], [2, 1, 2020]]], dtype=torch.long
    )
    sample = MaskedOlmoEarthSample(
        sentinel2_l2a=torch.randn(B, H, W, T, sentinel2_l2a_num_bands),
        sentinel2_l2a_mask=torch.zeros(
            B, H, W, T, sentinel2_l2a_num_bands, dtype=torch.long
        ),
        latlon=torch.randn(B, latlon_num_bands),
        latlon_mask=torch.zeros(B, latlon_num_bands, dtype=torch.long),
        timestamps=timestamps,
    )
    output_dict = encoder.forward(sample, patch_size=2, input_res=10)
    assert output_dict["registers"].shape == (B, grid_size * grid_size, register_dim)
    output_dict["registers"].sum().backward()
    # The read draws K/V from the depth-2 output, so the read block and blocks up to and
    # including depth 2 (indices 0, 1) receive gradient...
    assert bottleneck.read_blocks[0].attn.q.weight.grad is not None
    assert encoder.blocks[0].attn.q.weight.grad is not None
    assert encoder.blocks[1].attn.q.weight.grad is not None
    # ...while blocks AFTER the deepest read layer (depths 3, 4) do not -- they have no
    # path to the registers, proving the read sources the intermediate layer, not the final.
    assert encoder.blocks[2].attn.q.weight.grad is None
    assert encoder.blocks[3].attn.q.weight.grad is None


def test_encoder_register_bottleneck_per_depth_read_proj(
    modality_band_set_len_and_total_bands: dict[str, tuple[int, int]],
) -> None:
    """per_depth_read_proj gives each multi-depth read its own input_norm + kv_proj."""
    supported_modalities = [Modality.SENTINEL2_L2A, Modality.LATLON]
    sentinel2_l2a_num_bands = modality_band_set_len_and_total_bands["sentinel2_l2a"][1]
    latlon_num_bands = modality_band_set_len_and_total_bands["latlon"][1]
    grid_size, register_dim, depth = 3, 8, 4
    read_layers = [2, 4]
    encoder = Encoder(
        supported_modalities=supported_modalities,
        embedding_size=16,
        max_patch_size=4,
        min_patch_size=1,
        num_heads=2,
        mlp_ratio=2.0,
        max_sequence_length=12,
        depth=depth,
        drop_path=0.0,
        spatial_pos_encoding="rope",
        use_register_bottleneck=True,
        register_grid_size=grid_size,
        register_dim=register_dim,
        register_read_layers=read_layers,
        register_per_depth_read_proj=True,
    )
    bottleneck = encoder.register_bottleneck
    assert bottleneck is not None
    assert bottleneck.per_depth_read_proj
    # One norm + one projection per read layer; no shared pair.
    assert len(bottleneck.input_norms) == len(read_layers)
    assert len(bottleneck.kv_projs) == len(read_layers)
    assert not hasattr(bottleneck, "input_norm")
    assert not hasattr(bottleneck, "kv_proj")

    B, H, W = 2, 8, 8
    timestamps = torch.tensor(
        [[[1, 0, 2020], [2, 1, 2020]], [[1, 0, 2020], [2, 1, 2020]]], dtype=torch.long
    )
    sample = MaskedOlmoEarthSample(
        sentinel2_l2a=torch.randn(B, H, W, 2, sentinel2_l2a_num_bands),
        sentinel2_l2a_mask=torch.zeros(
            B, H, W, 2, sentinel2_l2a_num_bands, dtype=torch.long
        ),
        latlon=torch.randn(B, latlon_num_bands),
        latlon_mask=torch.zeros(B, latlon_num_bands, dtype=torch.long),
        timestamps=timestamps,
    )
    output_dict = encoder.forward(sample, patch_size=2, input_res=10)
    assert output_dict["registers"].shape == (B, grid_size * grid_size, register_dim)
    output_dict["registers"].sum().backward()
    # Each per-depth norm + projection receives gradient.
    for norm in bottleneck.input_norms:
        assert norm.weight.grad is not None
    for proj in bottleneck.kv_projs:
        assert proj.weight.grad is not None


def test_encoder_register_bottleneck_contrastive_source(
    modality_band_set_len_and_total_bands: dict[str, tuple[int, int]],
) -> None:
    """register_contrastive_source toggles the contrastive head between latents/tokens."""
    supported_modalities = [Modality.SENTINEL2_L2A, Modality.LATLON]
    sentinel2_l2a_num_bands = modality_band_set_len_and_total_bands["sentinel2_l2a"][1]
    latlon_num_bands = modality_band_set_len_and_total_bands["latlon"][1]
    embedding_size, register_dim = 16, 8

    def build(source: str) -> Encoder:
        return Encoder(
            supported_modalities=supported_modalities,
            embedding_size=embedding_size,
            max_patch_size=4,
            min_patch_size=1,
            num_heads=2,
            mlp_ratio=2.0,
            max_sequence_length=12,
            depth=2,
            drop_path=0.0,
            spatial_pos_encoding="rope",
            use_register_bottleneck=True,
            register_grid_size=3,
            register_dim=register_dim,
            register_contrastive_source=source,
        )

    B, H, W = 2, 8, 8
    timestamps = torch.tensor(
        [[[1, 0, 2020], [2, 1, 2020]], [[1, 0, 2020], [2, 1, 2020]]], dtype=torch.long
    )

    def sample() -> MaskedOlmoEarthSample:
        return MaskedOlmoEarthSample(
            sentinel2_l2a=torch.randn(B, H, W, 2, sentinel2_l2a_num_bands),
            sentinel2_l2a_mask=torch.zeros(
                B, H, W, 2, sentinel2_l2a_num_bands, dtype=torch.long
            ),
            latlon=torch.randn(B, latlon_num_bands),
            latlon_mask=torch.zeros(B, latlon_num_bands, dtype=torch.long),
            timestamps=timestamps,
        )

    # Default: project from the register latents (sized to register_dim).
    reg_encoder = build("registers")
    assert reg_encoder.contrastive_from_registers
    reg_out = reg_encoder.forward(sample(), patch_size=2, input_res=10)
    assert reg_out["project_aggregated"].shape == (B, register_dim)

    # Opt-in: project from the encoder patch tokens (sized to the final embedding size),
    # the pre-bottleneck behaviour.
    tok_encoder = build("encoder_tokens")
    assert not tok_encoder.contrastive_from_registers
    tok_out = tok_encoder.forward(sample(), patch_size=2, input_res=10)
    assert tok_out["project_aggregated"].shape == (B, embedding_size)


def test_predictor_forward_rope(
    modality_band_set_len_and_total_bands: dict[str, tuple[int, int]],
) -> None:
    """Predictor cross-attention should pass separate 2D RoPE positions for Q/K."""
    supported_modalities = [Modality.SENTINEL2_L2A, Modality.LATLON]
    sentinel2_l2a_num_band_sets = modality_band_set_len_and_total_bands[
        "sentinel2_l2a"
    ][0]
    latlon_num_band_sets = modality_band_set_len_and_total_bands["latlon"][0]
    predictor = Predictor(
        supported_modalities=supported_modalities,
        encoder_embedding_size=16,
        decoder_embedding_size=16,
        depth=2,
        mlp_ratio=2.0,
        num_heads=2,
        max_sequence_length=12,
        drop_path=0.0,
        spatial_pos_encoding="rope",
    )

    B, H, W, T = 1, 2, 2, 2
    sentinel2_l2a_tokens = torch.randn(
        B, H, W, T, sentinel2_l2a_num_band_sets, 16, requires_grad=True
    )
    sentinel2_l2a_mask = torch.zeros(
        B, H, W, T, sentinel2_l2a_num_band_sets, dtype=torch.float32
    )
    sentinel2_l2a_mask[:, 0, 0, :, :] = MaskValue.DECODER.value
    latlon = torch.randn(B, latlon_num_band_sets, 16, requires_grad=True)
    latlon_mask = torch.zeros(B, latlon_num_band_sets, dtype=torch.float32)
    encoded_tokens = TokensAndMasks(
        sentinel2_l2a=sentinel2_l2a_tokens,
        sentinel2_l2a_mask=sentinel2_l2a_mask,
        latlon=latlon,
        latlon_mask=latlon_mask,
    )
    timestamps = torch.tensor([[[1, 0, 2020], [2, 1, 2020]]], dtype=torch.long)

    output = predictor.forward(encoded_tokens, timestamps, patch_size=4, input_res=10)

    assert output.sentinel2_l2a is not None
    assert output.sentinel2_l2a.shape == (
        B,
        H,
        W,
        T,
        sentinel2_l2a_num_band_sets,
        16,
    )
    output.sentinel2_l2a.sum().backward()
    assert predictor.blocks[0].attn.q.weight.grad is not None


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
    x = MaskedOlmoEarthSample(**masked_sample_dict)

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
    )
    output_dict = encoder.forward(
        x,
        patch_size,
        input_res,
        token_exit_cfg=token_exit_cfg,
    )
    output, _, decoder_kwargs = unpack_encoder_output(output_dict)
    output = predictor.forward(
        output, x.timestamps, patch_size, input_res, **decoder_kwargs
    )
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
                "project_and_aggregate",
            ]
        ):
            assert param.grad is not None, name

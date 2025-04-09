"""Test LatentMIM with loss."""

import logging

import pytest
import torch

from helios.data.constants import Modality, ModalitySpec
from helios.nn.flexihelios import Encoder, Predictor, Reconstructor, TokensAndMasks
from helios.nn.mae import MAE
from helios.train.loss import MAELoss
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


def test_mae_with_loss(
    modality_band_set_len_and_total_bands: dict[str, tuple[int, int]],
) -> None:
    """Test the full end to end forward pass of the model with an exit configuration and loss."""
    supported_modalities = [
        Modality.SENTINEL2_L2A,
        Modality.WORLDCOVER,
    ]
    sentinel2_l2a_num_band_sets, sentinel2_l2a_num_bands = (
        modality_band_set_len_and_total_bands["sentinel2_l2a"]
    )
    B, H, W, T, C = (
        16,
        32,
        32,
        2,
        sentinel2_l2a_num_bands,
    )
    # Create dummy sentinel2_l2a data: shape (B, H, W, T, C)
    sentinel2_l2a = torch.randn(B, H, W, T, C)
    # Here we assume 0 (ONLINE_ENCODER) means the token is visible.
    sentinel2_l2a_mask = torch.zeros(
        B, H, W, T, sentinel2_l2a_num_band_sets, dtype=torch.long
    )

    worldcover = torch.randn(B, H, W, 1, 1)
    worldcover_mask = (
        torch.ones(B, H, W, 1, 1, dtype=torch.float32) * MaskValue.DECODER.value
    )
    # Generate valid timestamps:
    # - days: range 1..31,
    # - months: range 1..13,
    # - years: e.g. 2018-2019.
    days = torch.randint(0, 25, (B, T, 1), dtype=torch.long)
    months = torch.randint(0, 12, (B, T, 1), dtype=torch.long)
    years = torch.randint(2018, 2020, (B, T, 1), dtype=torch.long)
    timestamps = torch.cat([days, months, years], dim=-1)  # Shape: (B, T, 3)

    masked_sample_dict = {
        "sentinel2_l2a": sentinel2_l2a,
        "sentinel2_l2a_mask": sentinel2_l2a_mask,
        "worldcover": worldcover,
        "worldcover_mask": worldcover_mask,
        "timestamps": timestamps,
    }
    x = MaskedHeliosSample(**masked_sample_dict)

    patch_size = 2
    # input_res = 1
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
    reconstructor = Reconstructor(
        supported_modalities=supported_modalities,
        max_patch_size=MAX_PATCH_SIZE,
        embedding_size=ENCODER_EMBEDDING_SIZE,
    )
    mae = MAE(encoder, predictor, reconstructor)
    _, output = mae.forward(x, patch_size)
    assert output.sentinel2_l2a is not None
    assert output.sentinel2_l2a_mask is not None
    assert x.sentinel2_l2a is not None
    assert x.sentinel2_l2a_mask is not None
    assert output.sentinel2_l2a.shape == x.sentinel2_l2a.shape
    assert output.sentinel2_l2a_mask.shape == x.sentinel2_l2a_mask.shape

    assert output.worldcover is not None
    assert output.worldcover_mask is not None
    assert x.worldcover is not None
    assert x.worldcover_mask is not None
    assert output.worldcover.shape == x.worldcover.shape
    assert output.worldcover_mask.shape == x.worldcover_mask.shape

    assert (output.worldcover_mask == x.worldcover_mask).all()
    assert (output.sentinel2_l2a_mask == x.sentinel2_l2a_mask).all()

    # this reflects the forward_model function in mae
    loss_fn = MAELoss()
    reconstructed = output
    labels = x.as_dict()
    labels.pop("timestamps")
    target_output = TokensAndMasks(**labels)
    loss = loss_fn.compute(reconstructed, target_output)
    loss.backward()

    for name, param in mae.encoder.named_parameters():
        # worldcover and latlons are masked from the encoder
        if not any(
            ignore_param in name
            for ignore_param in [
                "pos_embed",
                "month_embed",
                "composite_encodings.per_modality_channel_embeddings.worldcover",
                "patch_embeddings.per_modality_embeddings.worldcover",
            ]
        ):
            assert param.grad is not None, name
    for name, param in mae.decoder.named_parameters():
        # sentinel2_l2a is "masked" from the decoder
        if not any(
            ignore_param in name
            for ignore_param in [
                "pos_embed",
                "month_embed",
            ]
        ):
            assert param.grad is not None, name

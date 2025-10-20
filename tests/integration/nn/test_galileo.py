"""Test the Galileo model."""

import logging

import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.nn.flexi_vit import Encoder, Predictor
from olmoearth_pretrain.nn.galileo import Galileo
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

logger = logging.getLogger(__name__)


def test_galileo_forward_pass(
    modality_band_set_len_and_total_bands: dict[str, tuple[int, int]],
    masked_sample_dict: dict[str, torch.Tensor],
) -> None:
    """Test the forward pass of the Galileo model."""
    # Define supported modalities
    supported_modalities = [
        Modality.SENTINEL2_L2A,
        Modality.LATLON,
        Modality.WORLDCOVER,
    ]
    sentinel2_l2a_num_band_sets, sentinel2_l2a_num_bands = (
        modality_band_set_len_and_total_bands["sentinel2_l2a"]
    )
    latlon_num_band_sets, latlon_num_bands = modality_band_set_len_and_total_bands[
        "latlon"
    ]
    B, H, W, T, C = masked_sample_dict["sentinel2_l2a"].shape
    x = MaskedOlmoEarthSample(**masked_sample_dict)

    patch_size = 4
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
    galileo = Galileo(encoder=encoder, decoder=predictor)

    output = galileo.forward(x, x, patch_size)
    output_a, output_b = output["a"], output["b"]
    _, decoded_a, _, _ = output_a
    _, decoded_b, _, _ = output_b
    patched_H = H // patch_size
    patched_W = W // patch_size

    # Check decoded_a shapes
    assert decoded_a.sentinel2_l2a is not None
    assert decoded_a.sentinel2_l2a_mask is not None
    assert decoded_a.latlon is not None
    assert decoded_a.latlon_mask is not None
    assert decoded_a.sentinel2_l2a.shape == (
        B,
        patched_H,
        patched_W,
        T,
        sentinel2_l2a_num_band_sets,
        predictor.output_embedding_size,
    )
    assert decoded_a.sentinel2_l2a_mask.shape == (
        B,
        patched_H,
        patched_W,
        T,
        sentinel2_l2a_num_band_sets,
    )
    assert decoded_a.latlon.shape == (
        B,
        latlon_num_band_sets,
        predictor.output_embedding_size,
    )
    assert decoded_a.latlon_mask.shape == (
        B,
        latlon_num_band_sets,
    )
    assert decoded_a.worldcover is not None
    assert decoded_a.worldcover_mask is not None
    assert decoded_a.worldcover.shape == (
        B,
        patched_H,
        patched_W,
        1,
        1,
        predictor.output_embedding_size,
    )
    assert decoded_a.worldcover_mask.shape == (
        B,
        patched_H,
        patched_W,
        1,
        1,
    )

    # Check decoded_b shapes
    assert decoded_b.sentinel2_l2a is not None
    assert decoded_b.sentinel2_l2a_mask is not None
    assert decoded_b.latlon is not None
    assert decoded_b.latlon_mask is not None
    assert decoded_b.sentinel2_l2a.shape == (
        B,
        patched_H,
        patched_W,
        T,
        sentinel2_l2a_num_band_sets,
        predictor.output_embedding_size,
    )
    assert decoded_b.sentinel2_l2a_mask.shape == (
        B,
        patched_H,
        patched_W,
        T,
        sentinel2_l2a_num_band_sets,
    )
    assert decoded_b.latlon.shape == (
        B,
        latlon_num_band_sets,
        predictor.output_embedding_size,
    )
    assert decoded_b.latlon_mask.shape == (
        B,
        latlon_num_band_sets,
    )
    assert decoded_b.worldcover is not None
    assert decoded_b.worldcover_mask is not None
    assert decoded_b.worldcover.shape == (
        B,
        patched_H,
        patched_W,
        1,
        1,
        predictor.output_embedding_size,
    )
    assert decoded_b.worldcover_mask.shape == (
        B,
        patched_H,
        patched_W,
        1,
        1,
    )

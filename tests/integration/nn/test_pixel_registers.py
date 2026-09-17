"""Tests for pixel-resolution registers (``register_pixel_grid``).

Covers the pixreg run (see ``scripts/official/v1_3/experiments/pixreg_pixrecon.py``):

* pixel-center register coordinate math;
* one register per pixel regardless of patch size, evenly spaced positions, and a
  finite gradient into the cloned latent;
* the pixel spacing as a ground property, identical at every patch size;
* the decoupled latent self-attention width/heads and the affine-free block norms;
* the whole pixel-register LatentMIM forward (decoder over the larger grid, per-cell
  map supervision, time-conditioned raw-band reconstruction heads).
"""

import pytest
import torch
from torch import nn

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.nn.flexi_vit import Encoder, EncoderConfig, PredictorConfig
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.nn.supervision_head import (
    SupervisionHeadConfig,
    SupervisionModalityConfig,
    SupervisionTaskType,
    compute_supervision_loss,
)
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample, MaskValue

REGISTER_DIM = 16
LATENT_ATTN_DIM = 16
LATENT_NUM_HEADS = 2
B, H, W, T = 2, 8, 8, 2


def _build_encoder() -> Encoder:
    """Small pixel-grid register encoder mirroring the pixreg run config."""
    return Encoder(
        supported_modalities=[Modality.SENTINEL2_L2A, Modality.LATLON],
        embedding_size=32,
        max_patch_size=4,
        min_patch_size=1,
        num_heads=2,
        mlp_ratio=2.0,
        max_sequence_length=12,
        depth=2,
        drop_path=0.0,
        position_encoding="rope",
        use_register_bottleneck=True,
        register_dim=REGISTER_DIM,
        register_latent_depth=2,
        register_per_depth_read_proj=True,
        # Wideread: reads at encoder width; latent self-attention at register width.
        register_attn_dim=32,
        register_pixel_grid=True,
        register_latent_attn_dim=LATENT_ATTN_DIM,
        register_latent_num_heads=LATENT_NUM_HEADS,
        register_norm_affine=False,
    )


def _make_sample() -> MaskedOlmoEarthSample:
    """Sample with unit-granularity mixed masking on 4x4 pixel blocks.

    The 4x4 blocks align with every patch size in {1, 2, 4}, so each
    ``(patch, timestep)`` unit carries a single mask value at any tested ps.
    """
    torch.manual_seed(1234)
    num_bands = Modality.SENTINEL2_L2A.num_bands
    latlon_bands = Modality.LATLON.num_bands
    mask = torch.zeros(B, H, W, T, num_bands, dtype=torch.long)
    mask[:, 0:4, 0:4, 0, :] = MaskValue.DECODER.value
    mask[:, 4:8, 0:4, 1, :] = MaskValue.TARGET_ENCODER_ONLY.value
    return MaskedOlmoEarthSample(
        sentinel2_l2a=torch.randn(B, H, W, T, num_bands),
        sentinel2_l2a_mask=mask,
        latlon=torch.randn(B, latlon_bands),
        latlon_mask=torch.zeros(B, latlon_bands, dtype=torch.long),
        timestamps=torch.tensor(
            [[[1, 0, 2020], [2, 1, 2020]]], dtype=torch.long
        ).expand(B, -1, -1),
    )


def _forward(
    encoder: Encoder, sample: MaskedOlmoEarthSample, patch_size: int
) -> dict[str, torch.Tensor]:
    """Deterministic (eval, no-grad) forward pass."""
    encoder.eval()
    with torch.no_grad():
        return encoder.forward(sample, patch_size=patch_size, input_res=10)


def test_pixel_register_positions_math() -> None:
    """Pixel-center register coordinates follow ((p + 0.5) / ps - 0.5) * scale."""
    encoder = _build_encoder()
    bottleneck = encoder.register_bottleneck
    assert bottleneck is not None
    scale = 3.0
    positions = bottleneck.build_pixel_register_positions(
        batch_size=2,
        register_grid=(4, 6),
        patch_size=2,
        patch_coordinate_scale=scale,
        device=torch.device("cpu"),
    )
    assert positions.shape == (2, 24, 2)
    expected_rows = torch.tensor([((p + 0.5) / 2 - 0.5) * scale for p in range(4)])
    grid = positions[0].view(4, 6, 2)
    torch.testing.assert_close(grid[:, 0, 0], expected_rows)
    # All rows share the column coordinates, all columns share the row coordinates.
    assert torch.equal(grid[:, :1, 0].expand(-1, 6), grid[..., 0])
    assert torch.equal(grid[:1, :, 1].expand(4, -1), grid[..., 1])
    # At patch_size=1 the pixel positions reduce to the patch positions p * scale.
    ps1 = bottleneck.build_pixel_register_positions(
        1, (4, 4), 1, scale, torch.device("cpu")
    )
    torch.testing.assert_close(
        ps1[0].view(4, 4, 2)[:, 0, 0], torch.arange(4, dtype=torch.float32) * scale
    )


@pytest.mark.parametrize("patch_size", [1, 2, 4])
def test_pixel_grid_register_shapes(patch_size: int) -> None:
    """One register per pixel regardless of patch size; pixel-spacing positions."""
    torch.manual_seed(0)
    encoder = _build_encoder()
    sample = _make_sample()
    encoder.zero_grad()
    output = encoder.forward(sample, patch_size=patch_size, input_res=10)

    n_reg = H * W
    # The bottleneck returns the grid shaped; positions stay flat (row-major).
    assert output["registers"].shape == (B, H, W, REGISTER_DIM)
    assert output["register_positions"].shape == (B, n_reg, 2)
    # Consecutive pixels are evenly spaced along each axis.
    grid = output["register_positions"][0].view(H, W, 2)
    row_steps = grid[1:, 0, 0] - grid[:-1, 0, 0]
    col_steps = grid[0, 1:, 1] - grid[0, :-1, 1]
    torch.testing.assert_close(row_steps, row_steps[:1].expand(H - 1))
    torch.testing.assert_close(col_steps, col_steps[:1].expand(W - 1))

    output["registers"].sum().backward()
    bottleneck = encoder.register_bottleneck
    assert bottleneck is not None
    assert bottleneck.register.grad is not None
    assert torch.isfinite(bottleneck.register.grad).all()


def test_pixel_register_spacing_invariant_across_patch_sizes() -> None:
    """The pixel register spacing is a ground property: identical at every ps."""
    torch.manual_seed(0)
    encoder = _build_encoder()
    sample = _make_sample()
    spacings = []
    for patch_size in (1, 2, 4):
        output = _forward(encoder, sample, patch_size)
        grid = output["register_positions"][0].view(H, W, 2)
        spacings.append(grid[1, 0, 0] - grid[0, 0, 0])
    torch.testing.assert_close(spacings[0], spacings[1])
    torch.testing.assert_close(spacings[0], spacings[2])


def test_pixel_grid_matches_patch_grid_at_ps1() -> None:
    """At ps=1 the pixel-grid bottleneck lays down the same frame as the patch grid."""
    encoder = _build_encoder()
    bottleneck = encoder.register_bottleneck
    assert bottleneck is not None
    scale = 2.5
    patch_positions = torch.stack(
        torch.meshgrid(
            torch.arange(4, dtype=torch.float32) * scale,
            torch.arange(4, dtype=torch.float32) * scale,
            indexing="ij",
        ),
        dim=-1,
    ).reshape(1, 16, 2)
    patch_grid_positions = bottleneck.build_register_positions(patch_positions, (4, 4))
    pixel_positions = bottleneck.build_pixel_register_positions(
        1, (4, 4), 1, scale, torch.device("cpu")
    )
    torch.testing.assert_close(pixel_positions, patch_grid_positions)


def test_latent_attention_decoupled_and_norms_affine_free() -> None:
    """Reads at encoder width; latent blocks at the register width; no block affine."""
    encoder = _build_encoder()
    bottleneck = encoder.register_bottleneck
    assert bottleneck is not None
    for read_blk in bottleneck.read_blocks:
        assert read_blk.attn.num_heads == 2
        # Wideread: q/k/v run at the encoder width (32), not the register width.
        assert read_blk.attn.q.out_features == 32
    for latent_blk in bottleneck.latent_blocks:
        assert latent_blk.attn.num_heads == LATENT_NUM_HEADS
        # latent_attn_dim == register_dim -> classic tied-width attention.
        assert latent_blk.attn.q.out_features == REGISTER_DIM
    for blk in (*bottleneck.read_blocks, *bottleneck.latent_blocks):
        for module in blk.modules():
            if isinstance(module, nn.LayerNorm):
                assert not module.elementwise_affine
    # The K/V input norms and the output norm keep their affine.
    for norm in bottleneck.input_norms:
        assert norm.elementwise_affine
    assert bottleneck.norm.elementwise_affine


def test_pixel_grid_requires_bottleneck() -> None:
    """register_pixel_grid without the bottleneck is a config error."""
    config = EncoderConfig(
        supported_modality_names=[Modality.SENTINEL2_L2A.name],
        embedding_size=16,
        num_heads=2,
        depth=1,
        position_encoding="rope",
        register_pixel_grid=True,
    )
    with pytest.raises(ValueError, match="register_pixel_grid requires"):
        config.validate()


def test_latent_head_shape_validated() -> None:
    """A latent width that does not split into RoPE-sized heads is rejected."""
    config = EncoderConfig(
        supported_modality_names=[Modality.SENTINEL2_L2A.name],
        embedding_size=16,
        num_heads=2,
        depth=1,
        position_encoding="rope",
        use_register_bottleneck=True,
        register_dim=8,
        register_latent_attn_dim=8,
        register_latent_num_heads=3,
    )
    with pytest.raises(ValueError, match="register_latent_num_heads"):
        config.validate()


def _pixreg_pixrecon_model_config() -> LatentMIMConfig:
    """Small version of the pixreg_pixrecon run: pixel registers + recon heads."""
    modalities = [Modality.SENTINEL2_L2A.name, Modality.LATLON.name]
    encoder_config = EncoderConfig(
        supported_modality_names=modalities,
        embedding_size=32,
        num_heads=2,
        depth=2,
        mlp_ratio=2.0,
        max_patch_size=4,
        min_patch_size=1,
        max_sequence_length=12,
        drop_path=0.0,
        position_encoding="rope",
        use_register_bottleneck=True,
        register_dim=REGISTER_DIM,
        register_latent_depth=2,
        register_per_depth_read_proj=True,
        register_attn_dim=32,
        register_pixel_grid=True,
        register_latent_attn_dim=LATENT_ATTN_DIM,
        register_latent_num_heads=LATENT_NUM_HEADS,
        register_norm_affine=False,
    )
    decoder_config = PredictorConfig(
        supported_modality_names=modalities,
        encoder_embedding_size=32,
        decoder_embedding_size=16,
        num_heads=2,
        depth=1,
        mlp_ratio=2.0,
        max_sequence_length=12,
        drop_path=0.0,
        position_encoding="rope",
        use_register_bottleneck=True,
        register_dim=REGISTER_DIM,
    )
    supervision_config = SupervisionHeadConfig(
        modality_configs={
            # A static per-cell head (regression so the random target is valid).
            "latlon": SupervisionModalityConfig(
                task_type=SupervisionTaskType.REGRESSION,
                num_output_channels=Modality.LATLON.num_bands,
                weight=0.1,
            ),
            # The pixrecon head: time-conditioned reconstruction of the S2 inputs.
            "sentinel2_l2a": SupervisionModalityConfig(
                task_type=SupervisionTaskType.REGRESSION,
                num_output_channels=Modality.SENTINEL2_L2A.num_bands,
                weight=0.05,
                regression_loss_type="mse",
                time_conditioned=True,
                time_harmonics=4,
            ),
        },
        spatial_unfold=1,
    )
    return LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        supervision_head_config=supervision_config,
    )


@pytest.mark.parametrize("patch_size", [1, 2, 4])
def test_pixreg_pixrecon_latent_mim_forward(patch_size: int) -> None:
    """The full model runs at every patch size and reconstructs per (pixel, timestep)."""
    torch.manual_seed(0)
    model = _pixreg_pixrecon_model_config().build()
    sample = _make_sample()
    model.train()
    (latent, decoded, _pooled, _recon, _metrics, supervision_preds, _proj) = (
        model.forward(sample, patch_size=patch_size)
    )
    assert supervision_preds is not None
    # Per-cell recon at every observed timestep, at the target's pixel resolution.
    assert supervision_preds["sentinel2_l2a"].shape == (
        B,
        H,
        W,
        T,
        Modality.SENTINEL2_L2A.num_bands,
    )
    assert supervision_preds["latlon"].shape == (B, Modality.LATLON.num_bands)
    # Decoder targets keep the input layout at this patch size.
    assert decoded.sentinel2_l2a is not None
    assert decoded.sentinel2_l2a.shape[:3] == (B, H // patch_size, W // patch_size)

    assert model.supervision_head is not None
    loss, per_modality = compute_supervision_loss(
        supervision_preds, sample, model.supervision_head
    )
    assert torch.isfinite(loss)
    assert set(per_modality) == {"sentinel2_l2a", "latlon"}
    loss.backward()
    # The reconstruction loss reaches the register latent through the pixel grid.
    assert model.encoder.register_bottleneck is not None
    grad = model.encoder.register_bottleneck.register.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0

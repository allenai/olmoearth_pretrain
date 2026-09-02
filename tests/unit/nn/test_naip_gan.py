"""Unit tests for the NAIP GAN generator and discriminator."""

import pytest
import torch

from olmoearth_pretrain.nn.naip_gan import (
    NaipDiscriminator,
    NaipGenerator,
    discriminator_adversarial_loss,
    generator_adversarial_loss,
)

torch.set_default_device("cpu")


def test_generator_output_shape() -> None:
    """Generator upsamples the token grid by patch_size * upsample_factor."""
    batch, height, width, dim = 2, 4, 5, 16
    patch_size, upsample_factor = 4, 4
    gen = NaipGenerator(
        embedding_size=dim,
        patch_size=patch_size,
        hidden_sizes=[32, 32, 32],
        out_channels=4,
        upsample_factor=upsample_factor,
        num_res_blocks=1,
    )
    pooled = torch.randn(batch, height, width, dim)
    out = gen(pooled, patch_size=patch_size)
    factor = patch_size * upsample_factor
    assert out.shape == (batch, 4, height * factor, width * factor)


def test_generator_upsample_factor_must_be_power_of_two() -> None:
    """A non-power-of-2 upsample factor is rejected."""
    with pytest.raises(ValueError):
        NaipGenerator(
            embedding_size=8, patch_size=4, hidden_sizes=[8, 8, 8], upsample_factor=3
        )


@pytest.mark.parametrize("in_height,in_width", [(4, 5), (2, 2), (7, 3)])
def test_generator_resamples_up_for_large_patch_size(
    in_height: int, in_width: int
) -> None:
    """A larger encoder patch size upsamples the token grid before unpatchify."""
    dim = 16
    unpatchify, upsample = 4, 4
    gen = NaipGenerator(
        embedding_size=dim,
        patch_size=unpatchify,  # canonical unpatchify factor U
        hidden_sizes=[32, 32, 32],
        out_channels=4,
        upsample_factor=upsample,
        num_res_blocks=1,
    )
    pooled = torch.randn(2, in_height, in_width, dim)
    # patch_size == U -> no resample.
    out = gen(pooled, patch_size=unpatchify)
    assert out.shape == (
        2,
        4,
        in_height * unpatchify * upsample,
        in_width * unpatchify * upsample,
    )
    # patch_size 8 (> U=4) -> resample the token grid by 8/4 = 2.
    out2 = gen(pooled, patch_size=8)
    assert out2.shape == (
        2,
        4,
        in_height * 2 * unpatchify * upsample,
        in_width * 2 * unpatchify * upsample,
    )


def test_generator_resamples_down_for_small_patch_size() -> None:
    """A smaller encoder patch size downsamples the token grid before unpatchify."""
    dim = 8
    gen = NaipGenerator(
        embedding_size=dim,
        patch_size=4,
        hidden_sizes=[16, 16],
        upsample_factor=2,
        num_res_blocks=1,
    )
    pooled = torch.randn(1, 8, 8, dim)
    # patch_size 2 (< U=4) -> resample by 2/4 = 0.5 -> 4x4 tokens.
    out = gen(pooled, patch_size=2)
    factor = 4 * 2  # U * upsample_factor
    assert out.shape == (1, 4, 4 * factor, 4 * factor)


def test_generator_canonical_patch_size_uses_token_grid() -> None:
    """When patch_size equals the unpatchify factor the token grid is used as-is."""
    dim = 8
    gen = NaipGenerator(
        embedding_size=dim,
        patch_size=2,
        hidden_sizes=[16, 16],
        upsample_factor=2,
        num_res_blocks=1,
    )
    out = gen(torch.randn(1, 4, 4, dim), patch_size=2)
    factor = 2 * 2
    assert out.shape == (1, 4, 4 * factor, 4 * factor)


def test_discriminator_output_shape() -> None:
    """Discriminator returns per-patch logits at the conditioning grid size."""
    batch, height, width, dim = 2, 4, 5, 16
    disc = NaipDiscriminator(
        embedding_size=dim,
        in_channels=4,
        image_strided_conv_channels=[16, 32],
        feature_channels=32,
    )
    image = torch.randn(batch, 4, height * 4, width * 4)
    cond = torch.randn(batch, height, width, dim)
    logits = disc(image, cond, patch_size=1)
    assert logits.shape == (batch, 1, height, width)


def test_generator_casts_input_to_param_dtype() -> None:
    """Generator handles inputs whose dtype differs from its params (mixed precision)."""
    gen = NaipGenerator(
        embedding_size=8,
        patch_size=2,
        hidden_sizes=[16, 16],
        upsample_factor=2,
        num_res_blocks=1,
    ).double()
    pooled = torch.randn(1, 4, 4, 8)  # float32 input, float64 params
    out = gen(pooled, patch_size=2)
    assert out.dtype == torch.float64


def test_discriminator_casts_inputs_to_param_dtype() -> None:
    """Discriminator handles inputs whose dtype differs from its params."""
    disc = NaipDiscriminator(
        embedding_size=8,
        in_channels=4,
        image_strided_conv_channels=[8, 16],
        feature_channels=16,
    ).double()
    image = torch.randn(1, 4, 16, 16)  # float32 inputs, float64 params
    cond = torch.randn(1, 4, 4, 8)
    logits = disc(image, cond, patch_size=1)
    assert logits.dtype == torch.float64


def test_discriminator_image_cond_output_shape() -> None:
    """Image-conditioned discriminator handles a Sentinel-2 temporal stack."""
    batch, t, grid_h, grid_w = 2, 5, 4, 5
    disc = NaipDiscriminator(
        embedding_size=16,  # unused in image mode
        in_channels=4,
        image_strided_conv_channels=[16, 32],
        feature_channels=32,
        cond_mode="image",
        cond_in_channels=6,
    )
    image = torch.randn(batch, 4, grid_h * 4, grid_w * 4)
    # Temporal stack [B, T, C, H, W] at a Sentinel-2-like resolution; fusion
    # happens at that image resolution.
    cond_h, cond_w = grid_h * 2, grid_w * 2
    cond = torch.randn(batch, t, 6, cond_h, cond_w)
    time_mask = torch.ones(batch, t, dtype=torch.bool)
    logits = disc(image, cond, patch_size=1, cond_time_mask=time_mask)
    assert logits.shape == (batch, 1, cond_h, cond_w)


def test_discriminator_unknown_cond_mode_raises() -> None:
    """An unsupported cond_mode is rejected."""
    with pytest.raises(ValueError):
        NaipDiscriminator(embedding_size=8, cond_mode="nope")


def test_discriminator_cond_pre_post_pool_channels() -> None:
    """The S2 cond stem uses non-strided pre-pool and post-pool conv stacks."""
    batch, t, grid_h, grid_w = 2, 4, 4, 5
    disc = NaipDiscriminator(
        embedding_size=8,  # unused in image mode
        in_channels=4,
        image_strided_conv_channels=[16, 32, 64],
        feature_channels=64,
        cond_mode="image",
        cond_in_channels=6,
        cond_image_pre_pool_channels=[16, 24],
        cond_image_post_pool_channels=[24],
    )
    image = torch.randn(batch, 4, grid_h * 8, grid_w * 8)
    cond_h, cond_w = grid_h * 2, grid_w * 2
    cond = torch.randn(batch, t, 6, cond_h, cond_w)
    logits = disc(image, cond, patch_size=1)
    assert logits.shape == (batch, 1, cond_h, cond_w)


def test_discriminator_image_cond_no_time_mask() -> None:
    """Without a time mask the condition averages over all timesteps."""
    batch, t, grid_h, grid_w = 2, 3, 4, 4
    disc = NaipDiscriminator(
        embedding_size=8,
        in_channels=4,
        image_strided_conv_channels=[16, 32],
        feature_channels=32,
        cond_mode="image",
        cond_in_channels=6,
    )
    image = torch.randn(batch, 4, grid_h * 4, grid_w * 4)
    cond_h, cond_w = grid_h * 3, grid_w * 3
    cond = torch.randn(batch, t, 6, cond_h, cond_w)
    logits = disc(image, cond, patch_size=1)
    assert logits.shape == (batch, 1, cond_h, cond_w)


def test_discriminator_cond_embedding_channels() -> None:
    """cond_embedding_channels adds post-unpatchify convs to the condition path."""
    disc = NaipDiscriminator(
        embedding_size=16,
        in_channels=4,
        image_strided_conv_channels=[16, 32],
        feature_channels=32,
        cond_embedding_channels=[64, 32],
    )
    # cond_proj is a single linear (the unpatchify); the convs live in
    # cond_post_unpatchify.
    assert isinstance(disc.cond_proj, torch.nn.Linear)
    num_convs = sum(
        1 for m in disc.cond_post_unpatchify if isinstance(m, torch.nn.Conv2d)
    )
    assert num_convs == 3  # two 3x3 convs (64, 32) plus the final 1x1 projection
    image = torch.randn(2, 4, 16, 20)
    cond = torch.randn(2, 4, 5, 16)
    logits = disc(image, cond, patch_size=1)
    assert logits.shape == (2, 1, 4, 5)


def test_discriminator_no_cond_embedding_channels_is_identity() -> None:
    """Without cond_embedding_channels there are no post-unpatchify convs."""
    disc = NaipDiscriminator(
        embedding_size=16,
        in_channels=4,
        image_strided_conv_channels=[16, 32],
        feature_channels=32,
    )
    num_convs = sum(
        1 for m in disc.cond_post_unpatchify if isinstance(m, torch.nn.Conv2d)
    )
    assert num_convs == 0


def test_discriminator_convs_per_resolution() -> None:
    """Refinement convs add layers without changing the output shape."""
    batch, height, width, dim = 2, 4, 5, 16
    disc = NaipDiscriminator(
        embedding_size=dim,
        in_channels=4,
        image_strided_conv_channels=[16, 32],
        feature_channels=32,
        num_convs_per_resolution=2,
    )
    # stem + 1 strided conv, each followed by 2 refinement convs (6), plus the
    # final projection conv to feature_channels (1) -> 7 convs total.
    num_convs = sum(1 for m in disc.from_image if isinstance(m, torch.nn.Conv2d))
    assert num_convs == 7
    image = torch.randn(batch, 4, height * 4, width * 4)
    cond = torch.randn(batch, height, width, dim)
    logits = disc(image, cond, patch_size=1)
    assert logits.shape == (batch, 1, height, width)


def test_discriminator_cond_unpatchify_factor_embedding() -> None:
    """Learned unpatchify fuses at the base token_grid * patch_size grid."""
    batch, height, width, dim = 2, 4, 5, 16
    f = 4
    disc = NaipDiscriminator(
        embedding_size=dim,
        in_channels=4,
        image_strided_conv_channels=[16, 32],
        feature_channels=32,
        use_projection=True,
        cond_unpatchify_factor=f,
    )
    image = torch.randn(batch, 4, height * 4 * f, width * 4 * f)
    cond = torch.randn(batch, height, width, dim)
    # patch_size == f: tokens already canonical, fusion grid = token_grid * f.
    logits = disc(image, cond, patch_size=f)
    assert logits.shape == (batch, 1, height * f, width * f)


def test_discriminator_cond_unpatchify_factor_image() -> None:
    """Image cond mode fuses at the Sentinel-2 image resolution."""
    batch, t, grid_h, grid_w = 2, 4, 3, 4
    disc = NaipDiscriminator(
        embedding_size=8,
        in_channels=4,
        image_strided_conv_channels=[16, 32],
        feature_channels=32,
        cond_mode="image",
        cond_in_channels=6,
    )
    image = torch.randn(batch, 4, grid_h * 8, grid_w * 8)
    cond_h, cond_w = grid_h * 4, grid_w * 4
    cond = torch.randn(batch, t, 6, cond_h, cond_w)
    time_mask = torch.ones(batch, t, dtype=torch.bool)
    logits = disc(image, cond, patch_size=1, cond_time_mask=time_mask)
    assert logits.shape == (batch, 1, cond_h, cond_w)


def test_discriminator_invalid_cond_unpatchify_factor() -> None:
    """cond_unpatchify_factor must be >= 1."""
    with pytest.raises(ValueError):
        NaipDiscriminator(embedding_size=8, cond_unpatchify_factor=0)


def test_discriminator_resamples_cond_for_patch_size() -> None:
    """A patch size below the unpatchify factor resamples the condition down."""
    batch, height, width, dim = 2, 8, 8, 16
    f = 4
    disc = NaipDiscriminator(
        embedding_size=dim,
        in_channels=4,
        image_strided_conv_channels=[16, 32],
        feature_channels=32,
        cond_unpatchify_factor=f,
    )
    cond = torch.randn(batch, height, width, dim)
    image = torch.randn(batch, 4, 64, 64)
    # patch_size 2 (< f=4): tokens -> round(8 * 2 / 4) = 4, fusion grid = 4 * f.
    logits = disc(image, cond, patch_size=2)
    assert logits.shape == (batch, 1, 4 * f, 4 * f)


@pytest.mark.parametrize("loss_type", ["hinge", "bce"])
def test_adversarial_losses_are_scalars(loss_type: str) -> None:
    """Both adversarial losses reduce to scalars for each variant."""
    real_logits = torch.randn(2, 1, 4, 4, requires_grad=True)
    fake_logits = torch.randn(2, 1, 4, 4, requires_grad=True)
    d_loss = discriminator_adversarial_loss(real_logits, fake_logits, loss_type)
    g_loss = generator_adversarial_loss(fake_logits, loss_type)
    assert d_loss.ndim == 0
    assert g_loss.ndim == 0
    # Losses should be differentiable.
    d_loss.backward(retain_graph=True)
    g_loss.backward()


def test_unknown_loss_type_raises() -> None:
    """An unsupported loss type is rejected."""
    logits = torch.randn(2, 1, 4, 4)
    with pytest.raises(ValueError):
        generator_adversarial_loss(logits, "nope")
    with pytest.raises(ValueError):
        discriminator_adversarial_loss(logits, logits, "nope")

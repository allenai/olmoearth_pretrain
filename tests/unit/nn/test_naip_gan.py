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
        hidden_size=32,
        out_channels=4,
        upsample_factor=upsample_factor,
        num_res_blocks=1,
    )
    pooled = torch.randn(batch, height, width, dim)
    out = gen(pooled)
    factor = patch_size * upsample_factor
    assert out.shape == (batch, 4, height * factor, width * factor)


def test_generator_upsample_factor_must_be_power_of_two() -> None:
    """A non-power-of-2 upsample factor is rejected."""
    with pytest.raises(ValueError):
        NaipGenerator(embedding_size=8, patch_size=4, upsample_factor=3)


def test_discriminator_output_shape() -> None:
    """Discriminator returns per-patch logits at the conditioning grid size."""
    batch, height, width, dim = 2, 4, 5, 16
    disc = NaipDiscriminator(
        embedding_size=dim,
        in_channels=4,
        hidden_size=16,
        num_image_layers=2,
    )
    image = torch.randn(batch, 4, height * 4, width * 4)
    cond = torch.randn(batch, height, width, dim)
    logits = disc(image, cond)
    assert logits.shape == (batch, 1, height, width)


def test_generator_casts_input_to_param_dtype() -> None:
    """Generator handles inputs whose dtype differs from its params (mixed precision)."""
    gen = NaipGenerator(
        embedding_size=8,
        patch_size=2,
        hidden_size=16,
        upsample_factor=2,
        num_res_blocks=1,
    ).double()
    pooled = torch.randn(1, 4, 4, 8)  # float32 input, float64 params
    out = gen(pooled)
    assert out.dtype == torch.float64


def test_discriminator_casts_inputs_to_param_dtype() -> None:
    """Discriminator handles inputs whose dtype differs from its params."""
    disc = NaipDiscriminator(
        embedding_size=8, in_channels=4, hidden_size=8, num_image_layers=2
    ).double()
    image = torch.randn(1, 4, 16, 16)  # float32 inputs, float64 params
    cond = torch.randn(1, 4, 4, 8)
    logits = disc(image, cond)
    assert logits.dtype == torch.float64


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

"""Conditional pix2pix-GAN branch for predicting NAIP from the pooled embedding.

The generator upsamples the encoder's pooled spatial embedding ``[B, H, W, D]``
into a NAIP image, and a conditional PatchGAN discriminator distinguishes real
vs generated NAIP conditioned on the same pooled embedding.

The generator lives inside the model (so it is trained by the main optimizer
alongside the encoder). The discriminator is held by the train module with its
own optimizer so its gradients never reach the encoder.
"""

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, TokensAndMasks
from olmoearth_pretrain.nn.latent_mim import LatentMIM
from olmoearth_pretrain.nn.pooling import PoolingType, pool_unmasked_tokens

logger = logging.getLogger(__name__)


def _num_groups(num_groups: int, channels: int) -> int:
    """Return a valid GroupNorm group count that divides ``channels``."""
    return math.gcd(num_groups, channels)


class _ResBlock(nn.Module):
    """A simple pre-activation residual block."""

    def __init__(self, channels: int, num_groups: int = 8):
        """Initialize the residual block.

        Args:
            channels: Number of input/output channels.
            num_groups: Target number of groups for GroupNorm.
        """
        super().__init__()
        groups = _num_groups(num_groups, channels)
        self.block = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the residual block."""
        return x + self.block(x)


class NaipGenerator(nn.Module):
    """Generator that upsamples a pooled spatial embedding into a NAIP image."""

    def __init__(
        self,
        embedding_size: int,
        patch_size: int,
        hidden_size: int = 128,
        out_channels: int = 4,
        upsample_factor: int = 4,
        num_res_blocks: int = 2,
        num_groups: int = 8,
        hidden_sizes: list[int] | None = None,
    ):
        """Initialize the generator.

        The total upsampling from the token grid is ``patch_size * upsample_factor``:
        a learned per-patch unpatchify (``patch_size``) brings the token grid to the
        base pixel resolution, then the conv trunk upsamples by ``upsample_factor``.
        For NAIP set ``upsample_factor = naip_10.image_tile_size_factor``.

        Args:
            embedding_size: Channel dim ``D`` of the pooled embedding ``[B, H, W, D]``.
            patch_size: Encoder patch size; the token is unpatchified into a
                ``patch_size x patch_size`` block of features.
            hidden_size: Width of the conv trunk when ``hidden_sizes`` is not given
                (a single width shared by every resolution stage).
            out_channels: Number of NAIP output bands (R, G, B, IR -> 4).
            upsample_factor: Conv-trunk spatial upsampling factor (power of 2).
            num_res_blocks: Residual blocks applied at the base resolution and
                after each upsampling step.
            num_groups: Target GroupNorm groups.
            hidden_sizes: Optional per-stage channel widths, one for the base
                (post-unpatchify) resolution followed by one for each upsampling
                stage (length ``log2(upsample_factor) + 1``). Lets capacity be
                concentrated at the coarse resolution, e.g. ``[256, 128, 128]``.
                Overrides ``hidden_size`` when provided.
        """
        super().__init__()
        n_up = int(round(math.log2(upsample_factor)))
        if 2**n_up != upsample_factor:
            raise ValueError(
                f"upsample_factor must be a power of 2, got {upsample_factor}"
            )
        # Per-stage channel widths: one for the base (post-unpatchify) resolution
        # plus one for each upsampling stage. Defaults to a flat ``hidden_size``.
        if hidden_sizes is None:
            channels = [hidden_size] * (n_up + 1)
        else:
            if len(hidden_sizes) != n_up + 1:
                raise ValueError(
                    f"hidden_sizes must have length log2(upsample_factor) + 1 = "
                    f"{n_up + 1} (base resolution plus one per upsampling stage), "
                    f"got {len(hidden_sizes)}"
                )
            channels = list(hidden_sizes)
        self.patch_size = patch_size
        self.hidden_size = channels[0]
        self.upsample_factor = upsample_factor
        self.out_channels = out_channels

        # Learned unpatchify: each token -> a patch_size x patch_size block of
        # base-resolution features, i.e. the token grid is expanded to the base
        # pixel grid.
        self.unpatchify = nn.Linear(
            embedding_size, patch_size * patch_size * channels[0]
        )

        layers: list[nn.Module] = [
            _ResBlock(channels[0], num_groups) for _ in range(num_res_blocks)
        ]
        for stage in range(n_up):
            c_in = channels[stage]
            c_out = channels[stage + 1]
            layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
            # This conv both refines and (optionally) changes the channel width.
            layers.append(nn.Conv2d(c_in, c_out, kernel_size=3, padding=1))
            layers.append(nn.GroupNorm(_num_groups(num_groups, c_out), c_out))
            layers.append(nn.SiLU())
            for _ in range(num_res_blocks):
                layers.append(_ResBlock(c_out, num_groups))
        self.upsample = nn.Sequential(*layers)

        self.to_image = nn.Conv2d(channels[-1], out_channels, kernel_size=3, padding=1)

    def forward(self, pooled: Tensor) -> Tensor:
        """Generate a NAIP image.

        Args:
            pooled: Pooled spatial embedding of shape ``[B, H, W, D]``.

        Returns:
            NAIP image of shape
            ``[B, out_channels, H * patch_size * upsample_factor, W * patch_size * upsample_factor]``.
        """
        # Match the (possibly mixed-precision) param dtype; pooling can upcast to
        # float32 even when the model runs in bf16 under FSDP mixed precision.
        pooled = pooled.to(self.unpatchify.weight.dtype)
        x = self.unpatchify(pooled)  # [B, H, W, patch^2 * hidden]
        x = rearrange(
            x,
            "b h w (ph pw c) -> b c (h ph) (w pw)",
            ph=self.patch_size,
            pw=self.patch_size,
            c=self.hidden_size,
        )  # [B, hidden, H * patch_size, W * patch_size] (base pixel grid)
        x = self.upsample(x)
        return self.to_image(x)


class NaipDiscriminator(nn.Module):
    """Conditional PatchGAN discriminator over NAIP images.

    The image is downsampled with strided convs, adaptively pooled to the token
    grid, then concatenated with the projected pooled embedding before a small
    conv head produces per-patch real/fake logits.
    """

    def __init__(
        self,
        embedding_size: int,
        in_channels: int = 4,
        hidden_size: int = 64,
        num_image_layers: int = 3,
        max_channel_multiple: int = 4,
    ):
        """Initialize the discriminator.

        Args:
            embedding_size: Channel dim ``D`` of the conditioning embedding.
            in_channels: Number of NAIP input bands.
            hidden_size: Base channel width.
            num_image_layers: Number of strided conv layers on the image.
            max_channel_multiple: Cap on the channel growth as a multiple of
                ``hidden_size``.
        """
        super().__init__()
        layers: list[nn.Module] = []
        c_in = in_channels
        c_out = hidden_size
        for _ in range(num_image_layers):
            layers.append(nn.Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            c_in = c_out
            c_out = min(c_out * 2, hidden_size * max_channel_multiple)
        self.from_image = nn.Sequential(*layers)
        self.feature_channels = c_in

        self.cond_proj = nn.Linear(embedding_size, self.feature_channels)
        self.head = nn.Sequential(
            nn.Conv2d(
                self.feature_channels * 2,
                self.feature_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.feature_channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, image: Tensor, cond: Tensor) -> Tensor:
        """Classify each patch as real or fake.

        Args:
            image: NAIP image of shape ``[B, in_channels, Hf, Wf]``.
            cond: Conditioning embedding of shape ``[B, H, W, D]``.

        Returns:
            Per-patch logits of shape ``[B, 1, H, W]``.
        """
        # The discriminator is kept in its own (fp32) precision, separate from the
        # FSDP mixed-precision model, so cast inputs to its param dtype.
        param_dtype = self.cond_proj.weight.dtype
        image = image.to(param_dtype)
        cond = cond.to(param_dtype)
        feats = self.from_image(image)
        feats = F.adaptive_avg_pool2d(feats, output_size=cond.shape[1:3])
        cond_feats = self.cond_proj(cond).permute(0, 3, 1, 2).contiguous()
        combined = torch.cat([feats, cond_feats], dim=1)
        return self.head(combined)


def discriminator_adversarial_loss(
    real_logits: Tensor, fake_logits: Tensor, loss_type: str = "hinge"
) -> Tensor:
    """Adversarial loss for the discriminator (real should score high)."""
    if loss_type == "hinge":
        return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()
    if loss_type == "bce":
        return F.binary_cross_entropy_with_logits(
            real_logits, torch.ones_like(real_logits)
        ) + F.binary_cross_entropy_with_logits(
            fake_logits, torch.zeros_like(fake_logits)
        )
    raise ValueError(f"Unknown GAN loss type: {loss_type}")


def generator_adversarial_loss(fake_logits: Tensor, loss_type: str = "hinge") -> Tensor:
    """Adversarial loss for the generator (fake should fool the discriminator)."""
    if loss_type == "hinge":
        return -fake_logits.mean()
    if loss_type == "bce":
        return F.binary_cross_entropy_with_logits(
            fake_logits, torch.ones_like(fake_logits)
        )
    raise ValueError(f"Unknown GAN loss type: {loss_type}")


@dataclass
class NaipGeneratorConfig(Config):
    """Configuration for :class:`NaipGenerator`.

    Total upsampling from the token grid is ``patch_size * upsample_factor``; set
    ``upsample_factor`` to the NAIP ``image_tile_size_factor`` so the output lands
    at native NAIP resolution.

    Set ``hidden_sizes`` to a per-stage list (base resolution plus one per
    upsampling stage) to concentrate capacity at the coarse resolution, e.g.
    ``[256, 128, 128]``; otherwise ``hidden_size`` is shared by every stage.
    """

    embedding_size: int
    patch_size: int
    hidden_size: int = 128
    out_channels: int = 4
    upsample_factor: int = 4
    num_res_blocks: int = 2
    num_groups: int = 8
    hidden_sizes: list[int] | None = None

    def build(self) -> NaipGenerator:
        """Build the generator."""
        return NaipGenerator(
            embedding_size=self.embedding_size,
            patch_size=self.patch_size,
            hidden_size=self.hidden_size,
            out_channels=self.out_channels,
            upsample_factor=self.upsample_factor,
            num_res_blocks=self.num_res_blocks,
            num_groups=self.num_groups,
            hidden_sizes=self.hidden_sizes,
        )


@dataclass
class NaipDiscriminatorConfig(Config):
    """Configuration for :class:`NaipDiscriminator`."""

    embedding_size: int
    in_channels: int = 4
    hidden_size: int = 64
    num_image_layers: int = 3
    max_channel_multiple: int = 4

    def build(self) -> NaipDiscriminator:
        """Build the discriminator."""
        return NaipDiscriminator(
            embedding_size=self.embedding_size,
            in_channels=self.in_channels,
            hidden_size=self.hidden_size,
            num_image_layers=self.num_image_layers,
            max_channel_multiple=self.max_channel_multiple,
        )


class NaipGanModel(LatentMIM):
    """LatentMIM plus a NAIP generator on top of the pooled spatial embedding."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        generator: nn.Module,
        reconstructor: torch.nn.Module | None = None,
    ):
        """Initialize the model.

        Args:
            encoder: The online encoder.
            decoder: The latent-MIM decoder (predictor).
            generator: The NAIP generator applied to the pooled embedding.
            reconstructor: Optional MAE reconstructor.
        """
        super().__init__(encoder=encoder, decoder=decoder, reconstructor=reconstructor)
        self.generator = generator

    def forward(  # type: ignore[override]
        self, x: MaskedOlmoEarthSample, patch_size: int
    ) -> tuple[
        TokensAndMasks,
        TokensAndMasks,
        torch.Tensor,
        TokensAndMasks | None,
        dict,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Forward pass returning the standard LatentMIM outputs plus GAN tensors.

        Returns:
            The five LatentMIM outputs, followed by the pooled spatial embedding
            ``[B, H, W, D]`` and the generated NAIP image ``[B, C, Hf, Wf]``.
        """
        latent, decoded, latent_projected_and_pooled, reconstructed, extra_metrics = (
            super().forward(x, patch_size)
        )
        pooled = pool_unmasked_tokens(
            latent, PoolingType.MEAN, spatial_pooling=True
        )  # [B, H, W, D]
        fake_naip = self.generator(pooled)
        return (
            latent,
            decoded,
            latent_projected_and_pooled,
            reconstructed,
            extra_metrics,
            pooled,
            fake_naip,
        )


@dataclass
class NaipGanModelConfig(Config):
    """Configuration for :class:`NaipGanModel`.

    Mirrors ``LatentMIMConfig`` (encoder + decoder + optional reconstructor) and
    adds the generator. Kept as a standalone config (rather than subclassing
    ``LatentMIMConfig``) so the ``generator_config`` field is required.
    """

    encoder_config: Config
    decoder_config: Config
    generator_config: Config
    reconstructor_config: Config | None = None

    def validate(self) -> None:
        """Validate encoder/decoder compatibility (mirrors LatentMIMConfig)."""
        if (
            self.encoder_config.supported_modalities
            != self.decoder_config.supported_modalities
        ):
            raise ValueError("Encoder and decoder must support the same modalities")
        if (
            self.encoder_config.max_sequence_length
            != self.decoder_config.max_sequence_length
        ):
            raise ValueError(
                "Encoder and decoder must have the same max sequence length"
            )
        encoder_output_size = (
            self.encoder_config.output_embedding_size
            or self.encoder_config.embedding_size
        )
        if encoder_output_size != self.decoder_config.encoder_embedding_size:
            raise ValueError("Encoder embedding size must be consistent!")
        if encoder_output_size != self.generator_config.embedding_size:
            raise ValueError(
                "Generator embedding_size must match the encoder output size "
                f"({encoder_output_size} != {self.generator_config.embedding_size})"
            )
        # The generator's unpatchify factor must equal the encoder's (fixed) patch
        # size so the generated image lands at native NAIP resolution.
        if (
            self.encoder_config.min_patch_size != self.encoder_config.max_patch_size
            or self.encoder_config.min_patch_size != self.generator_config.patch_size
        ):
            raise ValueError(
                "Generator patch_size must equal a fixed encoder patch size "
                f"(encoder min={self.encoder_config.min_patch_size}, "
                f"max={self.encoder_config.max_patch_size}, "
                f"generator={self.generator_config.patch_size})"
            )

    def build(self) -> NaipGanModel:
        """Build the NAIP GAN model."""
        self.validate()
        encoder = self.encoder_config.build()
        decoder = self.decoder_config.build()
        reconstructor = (
            self.reconstructor_config.build()
            if self.reconstructor_config is not None
            else None
        )
        generator = self.generator_config.build()
        return NaipGanModel(
            encoder=encoder,
            decoder=decoder,
            generator=generator,
            reconstructor=reconstructor,
        )

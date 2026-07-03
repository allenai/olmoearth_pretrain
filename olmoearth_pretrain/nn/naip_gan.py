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
        hidden_sizes: list[int],
        out_channels: int = 4,
        upsample_factor: int = 4,
        num_res_blocks: int = 2,
        num_groups: int = 8,
    ):
        """Initialize the generator.

        The total upsampling from the token grid is ``patch_size * upsample_factor``:
        a learned per-patch unpatchify (``patch_size``) brings the token grid to the
        base pixel resolution, then the conv trunk upsamples by ``upsample_factor``.
        For NAIP set ``upsample_factor = naip_10.image_tile_size_factor``.

        To support a flexible / varying encoder patch size, pass the encoder
        ``patch_size`` to :meth:`forward`: the pooled token grid is first bilinearly
        resampled so each token spans the canonical ``self.patch_size`` base pixels
        (keeping the physical extent fixed), and the single learned unpatchify then
        always lands at the base resolution regardless of the encoder patch size.

        Args:
            embedding_size: Channel dim ``D`` of the pooled embedding ``[B, H, W, D]``.
            patch_size: Learned unpatchify block factor (the canonical patch
                size); each (resampled) token is expanded into a
                ``patch_size x patch_size`` block of features, e.g. 4 -> 40 m/px
                tokens unpatchified to the 10 m/px base grid.
            hidden_sizes: Per-stage channel widths, one for the base
                (post-unpatchify) resolution followed by one for each upsampling
                stage (length ``log2(upsample_factor) + 1``). Lets capacity be
                concentrated at the coarse resolution, e.g. ``[256, 128, 128]``.
            out_channels: Number of NAIP output bands (R, G, B, IR -> 4).
            upsample_factor: Conv-trunk spatial upsampling factor (power of 2).
            num_res_blocks: Residual blocks applied at the base resolution and
                after each upsampling step.
            num_groups: Target GroupNorm groups.
        """
        super().__init__()
        n_up = int(round(math.log2(upsample_factor)))
        if 2**n_up != upsample_factor:
            raise ValueError(
                f"upsample_factor must be a power of 2, got {upsample_factor}"
            )
        # Per-stage channel widths: one for the base (post-unpatchify) resolution
        # plus one for each upsampling stage.
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

    def forward(self, pooled: Tensor, patch_size: int) -> Tensor:
        """Generate a NAIP image.

        Args:
            pooled: Pooled spatial embedding of shape ``[B, H, W, D]``.
            patch_size: Encoder patch size for this batch. When it differs from
                the unpatchify factor ``self.patch_size`` the token grid is
                bilinearly resampled by ``patch_size / self.patch_size`` so each
                token becomes canonical before the (fixed) unpatchify, keeping
                the output resolution consistent across patch sizes.

        Returns:
            NAIP image of shape ``[B, out_channels, Hc * U * upsample_factor,
            Wc * U * upsample_factor]`` where ``U`` is the unpatchify factor
            (``self.patch_size``) and ``(Hc, Wc)`` the (resampled) token grid.
        """
        # Match the (possibly mixed-precision) param dtype; pooling can upcast to
        # float32 even when the model runs in bf16 under FSDP mixed precision.
        pooled = pooled.to(self.unpatchify.weight.dtype)
        # Resample the token grid so each token spans the canonical
        # ``self.patch_size`` base pixels (fixed extent), letting one learned
        # unpatchify land at the base resolution for any encoder patch size.
        h, w = pooled.shape[1], pooled.shape[2]
        target = (
            max(1, round(h * patch_size / self.patch_size)),
            max(1, round(w * patch_size / self.patch_size)),
        )
        if target != (h, w):
            pooled = (
                F.interpolate(
                    pooled.permute(0, 3, 1, 2),
                    size=target,
                    mode="bilinear",
                    align_corners=False,
                )
                .permute(0, 2, 3, 1)
                .contiguous()
            )
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

    The image is downsampled with strided convs and adaptively pooled to the
    fusion grid (the base 10 m/px grid in embedding mode, or the Sentinel-2 image
    resolution in image mode), then combined with the projected condition before
    a conv head produces per-patch real/fake logits.

    Because the condition here is a rich Sentinel-2-derived embedding (not a
    downscaled NAIP image), the discriminator can be pushed to actually judge
    image-vs-condition *consistency* rather than unconditional realism via:

    * post-unpatchify convs on the condition (``cond_embedding_channels``),
    * a deeper reasoning head (``num_head_res_blocks``), and
    * a projection-discriminator inner-product term (``use_projection``;
      Miyato & Koyama, "cGANs with Projection Discriminator"), which only
      rewards logits when image features align with the projected condition.
    """

    def __init__(
        self,
        embedding_size: int,
        in_channels: int = 4,
        image_strided_conv_channels: list[int] | None = None,
        feature_channels: int = 256,
        cond_embedding_channels: list[int] | None = None,
        num_head_res_blocks: int = 0,
        use_projection: bool = False,
        num_groups: int = 8,
        cond_mode: str = "embedding",
        cond_in_channels: int = 4,
        cond_image_pre_pool_channels: list[int] | None = None,
        cond_image_post_pool_channels: list[int] | None = None,
        num_convs_per_resolution: int = 0,
        cond_unpatchify_factor: int = 1,
    ):
        """Initialize the discriminator.

        Args:
            embedding_size: Channel dim ``D`` of the conditioning embedding
                (used only when ``cond_mode == 'embedding'``).
            in_channels: Number of NAIP input bands.
            image_strided_conv_channels: Per-conv output widths of the NAIP image
                stack. The first entry is a stride-1 stem and each subsequent
                entry a stride-2 conv, e.g. ``[64, 128, 256]`` is a stem to 64
                then two strided convs to 128 and 256. Defaults to
                ``[64, 128, 256]``.
            feature_channels: Fused feature width. A final conv projects the image
                stack to this width, the condition is produced at this width, and
                the head fuses the two (so its input is ``2 * feature_channels``).
            cond_embedding_channels: Optional per-conv output widths of the
                non-strided convs applied to the condition *after* the learned
                unpatchify (a final 1x1 conv projects back to
                ``feature_channels``), e.g. ``[256]`` gives
                ``feature_channels -> 256 -> feature_channels``. ``None``/empty
                applies no post-unpatchify convs. Only used when
                ``cond_mode == 'embedding'``.
            num_head_res_blocks: Number of residual blocks inserted into the
                fusion head so it has depth to compare image features against the
                condition (0 keeps the original shallow head).
            use_projection: If True, add a projection-discriminator term: the
                spatial inner product between the image features and the
                projected condition is added to the logits, so the discriminator
                cannot score well by ignoring the condition.
            num_groups: Target GroupNorm groups for the head residual blocks.
            cond_mode: How the condition is provided. ``'embedding'`` (default)
                takes a pooled embedding ``[B, H, W, D]`` and projects it;
                ``'image'`` takes a raw image temporal stack
                ``[B, T, cond_in_channels, Hc, Wc]`` (the Sentinel-2 time series)
                and embeds it (see ``cond_image_pre_pool_channels``).
            cond_in_channels: Number of channels (bands) of the raw image
                condition when ``cond_mode == 'image'``.
            cond_image_pre_pool_channels: Per-conv output widths of the
                non-strided convs applied to each timestep of the raw image
                condition (at the image resolution) before the mean over time.
                Defaults to ``[feature_channels]``.
            cond_image_post_pool_channels: Per-conv output widths of the extra
                non-strided convs applied after the mean over time (before the
                final 1x1 projection to ``feature_channels``). Defaults to none.
            num_convs_per_resolution: Number of stride-1 refinement convs (3x3 +
                LeakyReLU) inserted after each conv of the NAIP image stack, at
                that resolution (0 disables them).
            cond_unpatchify_factor: In ``'embedding'`` mode, the factor ``f`` of
                the learned unpatchify: tokens are first resampled so each spans
                ``f`` canonical base pixels, then a single linear expands each
                into an ``f x f`` block of features, mirroring the generator.
                Fusion always happens at the base ``token_grid * patch_size``
                (10 m/px) grid regardless of ``f``; ``f`` only sets the
                unpatchify block factor (set it to the generator ``patch_size``,
                e.g. 4, to resample tokens to 40 m/px before a learned 4x4
                unpatchify to 10 m/px).
        """
        super().__init__()
        if cond_mode not in ("embedding", "image"):
            raise ValueError(f"Unknown cond_mode: {cond_mode}")
        if cond_unpatchify_factor < 1:
            raise ValueError(
                f"cond_unpatchify_factor must be >= 1, got {cond_unpatchify_factor}"
            )
        self.cond_mode = cond_mode
        self.use_projection = use_projection
        self.cond_unpatchify_factor = cond_unpatchify_factor
        self.feature_channels = feature_channels

        # Downsample the NAIP image: a stride-1 stem (channels[0]) then stride-2
        # convs (channels[1:]), each optionally followed by num_convs_per_resolution
        # stride-1 refinement convs, and a final conv to feature_channels.
        image_channels = image_strided_conv_channels or [64, 128, 256]
        if not image_channels:
            raise ValueError("image_strided_conv_channels must be non-empty")
        image_layers: list[nn.Module] = []
        prev = in_channels
        for i, width in enumerate(image_channels):
            if i == 0:
                image_layers.append(
                    nn.Conv2d(prev, width, kernel_size=3, stride=1, padding=1)
                )
            else:
                image_layers.append(
                    nn.Conv2d(prev, width, kernel_size=4, stride=2, padding=1)
                )
            image_layers.append(nn.LeakyReLU(0.2, inplace=True))
            for _ in range(num_convs_per_resolution):
                image_layers.append(
                    nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1)
                )
                image_layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev = width
        image_layers.append(nn.Conv2d(prev, feature_channels, kernel_size=3, padding=1))
        image_layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.from_image = nn.Sequential(*image_layers)

        if cond_mode == "embedding":
            # Learned unpatchify of the pooled embedding [B, H, W, D]: a single
            # linear expands each token into an f x f block of features
            # (f = cond_unpatchify_factor), mirroring the generator.
            cond_out = (
                cond_unpatchify_factor * cond_unpatchify_factor * feature_channels
            )
            self.cond_proj = nn.Linear(embedding_size, cond_out)
            # Optional non-strided convs applied after the unpatchify, with a
            # final 1x1 projection back to feature_channels. Identity when unset.
            unpatchify_layers: list[nn.Module] = []
            prev = feature_channels
            for width in cond_embedding_channels or []:
                unpatchify_layers.append(
                    nn.Conv2d(prev, width, kernel_size=3, padding=1)
                )
                unpatchify_layers.append(nn.LeakyReLU(0.2, inplace=True))
                prev = width
            if cond_embedding_channels:
                unpatchify_layers.append(
                    nn.Conv2d(prev, feature_channels, kernel_size=1)
                )
            self.cond_post_unpatchify = nn.Sequential(*unpatchify_layers)
        else:
            # Raw image condition as a temporal stack [B, T, cond_in_channels, H, W].
            # Non-strided per-timestep convs keep it at the image resolution; after
            # the mean over time, more non-strided convs and a 1x1 projection to
            # feature_channels. The result is resampled to the fusion grid in
            # forward (a no-op when it already matches).
            pre_channels = cond_image_pre_pool_channels or [feature_channels]
            pre_layers: list[nn.Module] = []
            prev = cond_in_channels
            for width in pre_channels:
                pre_layers.append(nn.Conv2d(prev, width, kernel_size=3, padding=1))
                pre_layers.append(nn.LeakyReLU(0.2, inplace=True))
                prev = width
            self.cond_pre_pool = nn.Sequential(*pre_layers)
            post_layers: list[nn.Module] = []
            for width in cond_image_post_pool_channels or []:
                post_layers.append(nn.Conv2d(prev, width, kernel_size=3, padding=1))
                post_layers.append(nn.LeakyReLU(0.2, inplace=True))
                prev = width
            post_layers.append(nn.Conv2d(prev, feature_channels, kernel_size=1))
            self.cond_post_pool = nn.Sequential(*post_layers)

        head_layers: list[nn.Module] = [
            nn.Conv2d(
                self.feature_channels * 2,
                self.feature_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for _ in range(num_head_res_blocks):
            head_layers.append(_ResBlock(self.feature_channels, num_groups))
        head_layers.append(
            nn.Conv2d(self.feature_channels, 1, kernel_size=3, padding=1)
        )
        self.head = nn.Sequential(*head_layers)

    def forward(
        self,
        image: Tensor,
        cond: Tensor,
        patch_size: int,
        cond_time_mask: Tensor | None = None,
    ) -> Tensor:
        """Classify each patch as real or fake.

        Args:
            image: NAIP image of shape ``[B, in_channels, Hf, Wf]``.
            cond: Conditioning input. When ``cond_mode == 'embedding'`` this is a
                pooled embedding ``[B, H, W, D]`` (whose ``[H, W]`` is the token
                grid); when ``cond_mode == 'image'`` this is a raw image temporal
                stack ``[B, T, cond_in_channels, Hc, Wc]`` (the Sentinel-2 time
                series at its native 10 m/px resolution).
            patch_size: Encoder patch size for this batch. Only used in
                ``'embedding'`` mode: the token grid is resampled by
                ``patch_size / cond_unpatchify_factor`` so each token spans
                ``cond_unpatchify_factor`` canonical base pixels before the
                learned unpatchify, landing fusion at the base
                ``token_grid * patch_size`` (10 m/px) grid regardless of patch
                size. Ignored in ``'image'`` mode.
            cond_time_mask: Optional ``[B, T]`` boolean mask of valid timesteps for
                the ``'image'`` condition; the mean over time uses only valid
                timesteps. When ``None`` all timesteps are averaged.

        Returns:
            Per-patch logits. In ``'embedding'`` mode of shape
            ``[B, 1, H * patch_size, W * patch_size]`` (the base 10 m/px grid);
            in ``'image'`` mode of shape ``[B, 1, Hc, Wc]`` (the Sentinel-2 image
            resolution).
        """
        # The discriminator is kept in its own (fp32) precision, separate from the
        # FSDP mixed-precision model, so cast inputs to its param dtype. Reference
        # the first image conv, which exists regardless of the cond variant.
        param_dtype = self.from_image[0].weight.dtype
        image = image.to(param_dtype)
        cond = cond.to(param_dtype)
        feats = self.from_image(image)
        f = self.cond_unpatchify_factor
        if self.cond_mode == "embedding":
            token_grid = (int(cond.shape[1]), int(cond.shape[2]))
            # Resample tokens so each spans the canonical ``f`` base pixels, so the
            # learned unpatchify lands at the base (10 m/px) grid for any encoder
            # patch size.
            canonical = (
                max(1, round(token_grid[0] * patch_size / f)),
                max(1, round(token_grid[1] * patch_size / f)),
            )
            if canonical != token_grid:
                cond = (
                    F.interpolate(
                        cond.permute(0, 3, 1, 2),
                        size=canonical,
                        mode="bilinear",
                        align_corners=False,
                    )
                    .permute(0, 2, 3, 1)
                    .contiguous()
                )
                token_grid = canonical
            fusion_grid = (token_grid[0] * f, token_grid[1] * f)
            feats = F.adaptive_avg_pool2d(feats, output_size=fusion_grid)
            # Learned unpatchify: [B, H, W, f*f*C] -> [B, C, H*f, W*f].
            cond_feats = rearrange(
                self.cond_proj(cond),
                "b h w (ph pw c) -> b c (h ph) (w pw)",
                ph=f,
                pw=f,
                c=self.feature_channels,
            ).contiguous()
            cond_feats = self.cond_post_unpatchify(cond_feats)
        else:
            # cond is a temporal stack [B, T, cond_in_channels, H, W] at the
            # Sentinel-2 (10 m/px) resolution. Apply the per-timestep convs, mean
            # over the (valid) timesteps, then the post-pool convs; fusion happens
            # at that native image resolution.
            b, t = cond.shape[0], cond.shape[1]
            x = self.cond_pre_pool(cond.flatten(0, 1)).unflatten(0, (b, t))
            if cond_time_mask is not None:
                m = cond_time_mask.to(x.dtype).reshape(b, t, 1, 1, 1)
                x = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
            else:
                x = x.mean(dim=1)
            cond_feats = self.cond_post_pool(x)  # [B, feature_channels, Hc, Wc]
            feats = F.adaptive_avg_pool2d(feats, output_size=cond_feats.shape[-2:])
        combined = torch.cat([feats, cond_feats], dim=1)
        logits = self.head(combined)
        if self.use_projection:
            # Projection-discriminator term: reward image/condition agreement.
            proj = (feats * cond_feats).sum(dim=1, keepdim=True)
            logits = logits + proj
        return logits


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

    Set ``hidden_sizes`` to a per-stage list of channel widths (base resolution
    plus one per upsampling stage, length ``log2(upsample_factor) + 1``) to
    concentrate capacity at the coarse resolution, e.g. ``[256, 128, 128]``.

    Pass the encoder ``patch_size`` to the generator's ``forward`` to support a
    flexible encoder patch size: the token grid is resampled to the canonical
    ``patch_size`` (unpatchify factor) before the unpatchify, so the output stays
    consistent across patch sizes.
    """

    embedding_size: int
    patch_size: int
    hidden_sizes: list[int]
    out_channels: int = 4
    upsample_factor: int = 4
    num_res_blocks: int = 2
    num_groups: int = 8

    def build(self) -> NaipGenerator:
        """Build the generator."""
        return NaipGenerator(
            embedding_size=self.embedding_size,
            patch_size=self.patch_size,
            hidden_sizes=self.hidden_sizes,
            out_channels=self.out_channels,
            upsample_factor=self.upsample_factor,
            num_res_blocks=self.num_res_blocks,
            num_groups=self.num_groups,
        )


@dataclass
class NaipDiscriminatorConfig(Config):
    """Configuration for :class:`NaipDiscriminator`.

    Set ``cond_embedding_channels``, ``num_head_res_blocks`` and ``use_projection``
    to make the discriminator reason more about the conditioning embedding (rather
    than just the realism of the input image).

    Set ``cond_mode='image'`` (with ``cond_in_channels``) to condition on a raw
    image (the full Sentinel-2 temporal stack) instead of a pooled embedding;
    ``embedding_size`` is then unused. ``cond_image_pre_pool_channels`` are
    non-strided convs applied to each timestep at the S2 resolution;
    ``cond_image_post_pool_channels`` are non-strided convs applied after the mean
    over time. In embedding mode ``cond_embedding_channels`` are non-strided convs
    applied after the learned unpatchify of the condition.

    ``image_strided_conv_channels`` is the per-conv width list for the NAIP image
    stack (first entry a stride-1 stem, the rest stride-2 convs); ``feature_channels``
    is the fused width the image and condition are projected to before the head.

    Set ``cond_unpatchify_factor`` to the generator ``patch_size`` so the condition
    tokens are resampled to that (coarse) resolution and a learned unpatchify
    expands them back to the base 10 m/px fusion grid, mirroring the generator
    (embedding mode only; image mode fuses at the Sentinel-2 image resolution).
    """

    embedding_size: int
    in_channels: int = 4
    image_strided_conv_channels: list[int] | None = None
    feature_channels: int = 256
    cond_embedding_channels: list[int] | None = None
    num_head_res_blocks: int = 0
    use_projection: bool = False
    num_groups: int = 8
    cond_mode: str = "embedding"
    cond_in_channels: int = 4
    cond_image_pre_pool_channels: list[int] | None = None
    cond_image_post_pool_channels: list[int] | None = None
    num_convs_per_resolution: int = 0
    cond_unpatchify_factor: int = 1

    def build(self) -> NaipDiscriminator:
        """Build the discriminator."""
        return NaipDiscriminator(
            embedding_size=self.embedding_size,
            in_channels=self.in_channels,
            image_strided_conv_channels=self.image_strided_conv_channels,
            feature_channels=self.feature_channels,
            cond_embedding_channels=self.cond_embedding_channels,
            num_head_res_blocks=self.num_head_res_blocks,
            use_projection=self.use_projection,
            num_groups=self.num_groups,
            cond_mode=self.cond_mode,
            cond_in_channels=self.cond_in_channels,
            cond_image_pre_pool_channels=self.cond_image_pre_pool_channels,
            cond_image_post_pool_channels=self.cond_image_post_pool_channels,
            num_convs_per_resolution=self.num_convs_per_resolution,
            cond_unpatchify_factor=self.cond_unpatchify_factor,
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
        fake_naip = self.generator(pooled, patch_size)
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
        # The generator (and the discriminator, in the train module) resamples the
        # token grid to the canonical unpatchify factor using the per-batch encoder
        # patch size, so a fixed encoder patch size is no longer required.

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

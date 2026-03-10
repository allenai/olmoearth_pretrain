"""Simple set up of latent predictor."""

import logging
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import (
    MixedPrecisionPolicy,
    fully_shard,
    register_fsdp_forward_method,
)

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.nn.flexi_vit import TokensAndMasks
from olmoearth_pretrain.nn.utils import DistributedMixins, unpack_encoder_output

logger = logging.getLogger(__name__)


class RandomFourierProjection(nn.Module):
    """cos(Wx + b) * sqrt(2/D) — approximates a Gaussian RBF kernel.

    Captures non-linear interactions between input dimensions via random
    Fourier feature approximation. sigma controls kernel bandwidth:
    larger = more sensitive to small input differences.
    """

    _skip_custom_init = True

    def __init__(self, in_features: int, out_features: int, sigma: float = 1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scale = math.sqrt(2.0 / out_features)
        nn.init.normal_(self.linear.weight, std=sigma)
        nn.init.uniform_(self.linear.bias, 0, 2 * math.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.cos(self.linear(x))


class PolynomialProjection(nn.Module):
    """Degree-2 polynomial expansion + orthogonal random projection.

    Explicitly forms all pairwise product features [x, x_i * x_j] then
    projects with an orthogonal random matrix. Captures second-order
    interactions (e.g., band ratios like NDVI ~ NIR * Red).
    """

    _skip_custom_init = True

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        n_poly = in_features * (in_features + 1) // 2
        self.proj = nn.Linear(in_features + n_poly, out_features)
        nn.init.orthogonal_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        idx = torch.triu_indices(in_features, in_features)
        self.register_buffer("_triu_row", idx[0])
        self.register_buffer("_triu_col", idx[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2 = x[..., self._triu_row] * x[..., self._triu_col]
        return self.proj(torch.cat([x, x2], dim=-1))


class LatentMIM(nn.Module, DistributedMixins):
    """Latent MIM Style."""

    supports_multiple_modalities_at_once = True

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        reconstructor: torch.nn.Module | None = None,
        target_projection_init: str = "default",
        target_projection_sigma: float = 1.0,
    ):
        """Initialize the Latent MIM Style.

        Args:
            encoder: The encoder to use.
            decoder: The decoder to use.
            reconstructor: Optional reconstructor for auto-encoding.
            target_projection_init: How to initialize the target encoder's patch
                embeddings. One of "default" (keep encoder init), "orthogonal"
                (orthogonal matrix), "random_fourier" (RFF with cos non-linearity),
                or "polynomial" (degree-2 expansion + orthogonal projection).
            target_projection_sigma: Bandwidth for random_fourier init. Larger
                values = more sensitive to fine-grained input differences.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.reconstructor = reconstructor
        self.target_encoder = deepcopy(self.encoder)
        if hasattr(self.target_encoder, "disable_band_dropout"):
            self.target_encoder.disable_band_dropout()
        if target_projection_init != "default":
            self._reinit_target_projections(
                target_projection_init, sigma=target_projection_sigma
            )
        # Freeze after reinit so new modules are also frozen.
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def _reinit_target_projections(self, init_type: str, sigma: float = 1.0) -> None:
        """Reinitialize target encoder patch embeddings for frozen random projection targets."""
        from olmoearth_pretrain.nn.flexi_patch_embed import FlexiPatchEmbed

        count = 0
        for module in self.target_encoder.modules():
            if not isinstance(module, FlexiPatchEmbed):
                continue
            proj = module.proj
            if not isinstance(proj, nn.Linear):
                logger.warning(
                    f"Skipping non-Linear patch embed proj ({type(proj).__name__})"
                )
                continue
            in_f, out_f = proj.in_features, proj.out_features
            if init_type == "orthogonal":
                nn.init.orthogonal_(proj.weight)
                if proj.bias is not None:
                    nn.init.zeros_(proj.bias)
            elif init_type == "random_fourier":
                module.proj = RandomFourierProjection(in_f, out_f, sigma=sigma)
            elif init_type == "polynomial":
                module.proj = PolynomialProjection(in_f, out_f)
            else:
                raise ValueError(f"Unknown target_projection_init: {init_type!r}")
            count += 1
        logger.info(
            f"Reinitialized {count} target encoder patch embeddings with {init_type!r}"
        )

    def forward(
        self, x: MaskedOlmoEarthSample, patch_size: int
    ) -> tuple[
        TokensAndMasks,
        TokensAndMasks,
        torch.Tensor,
        TokensAndMasks | None,
        dict[str, Any],
    ]:
        """Forward pass for the Latent MIM Style.

        Returns:
            latent: embeddings from encoder
            decoded: predictions from decoder for masked tokens
            latent_projected_and_pooled: pooled tokens for contrastive loss
            reconstructed: MAE predictions if enabled
        """
        # TODO: Input And outputs here are not consistent between encoder and decoder need a tokensandmaks++
        output_dict = self.encoder(x, patch_size=patch_size)
        token_norm_stats = output_dict.pop("token_norm_stats", None)
        latent, latent_projected_and_pooled, decoder_kwargs = unpack_encoder_output(
            output_dict
        )
        extra_metrics = {}
        if token_norm_stats is not None:
            extra_metrics["token_norm_stats"] = token_norm_stats
        reconstructed = None
        if self.reconstructor:
            reconstructed = self.reconstructor(latent, x.timestamps, patch_size)
        decoded = self.decoder(
            latent, timestamps=x.timestamps, patch_size=patch_size, **decoder_kwargs
        )
        return (
            latent,
            decoded,
            latent_projected_and_pooled,
            reconstructed,
            extra_metrics,
        )

    def apply_fsdp(
        self,
        dp_mesh: DeviceMesh | None = None,
        param_dtype: torch.dtype | None = None,
        reduce_dtype: torch.dtype = torch.float32,
        prefetch_factor: int = 0,
    ) -> None:
        """Apply FSDP to the model."""
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype, reduce_dtype=reduce_dtype
        )
        fsdp_config = dict(mesh=dp_mesh, mp_policy=mp_policy)

        self.encoder.apply_fsdp(**fsdp_config)
        self.decoder.apply_fsdp(**fsdp_config)
        self.target_encoder.apply_fsdp(**fsdp_config)
        if self.reconstructor:
            self.reconstructor.apply_fsdp(**fsdp_config)
        # TODO: More finegrained wrapping of the encoder transformer layers next time
        fully_shard(self, **fsdp_config)
        register_fsdp_forward_method(self.target_encoder, "forward")

    def apply_compile(self) -> None:
        """Apply torch.compile to the model."""
        logger.info("Applying torch.compile to the model")
        self.encoder.apply_compile()
        logger.info("Applied torch.compile to the encoder")
        self.decoder.apply_compile()
        logger.info("Applied torch.compile to the decoder")
        self.target_encoder.apply_compile()
        logger.info("Applied torch.compile to the target encoder")


@dataclass
class LatentMIMConfig(Config):
    """Configuration for the Latent Predictor."""

    encoder_config: Config
    decoder_config: Config
    reconstructor_config: Config | None = None
    target_projection_init: str = "default"
    target_projection_sigma: float = 1.0

    def validate(self) -> None:
        """Validate the configuration."""
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
        if (
            self.encoder_config.embedding_size
            != self.decoder_config.encoder_embedding_size
        ):
            raise ValueError("Encoder embedding size must be consistent!")

    def build(self) -> "LatentMIM":
        """Build the Latent Predictor."""
        self.validate()
        encoder = self.encoder_config.build()
        decoder = self.decoder_config.build()
        reconstructor = (
            self.reconstructor_config.build()
            if self.reconstructor_config is not None
            else None
        )
        return LatentMIM(
            encoder=encoder,
            decoder=decoder,
            reconstructor=reconstructor,
            target_projection_init=self.target_projection_init,
            target_projection_sigma=self.target_projection_sigma,
        )

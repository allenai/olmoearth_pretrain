"""Latent MIM with a separate auxiliary encoder for one modality (e.g. NAIP).

The main encoder processes a subset of modalities (e.g. Sentinel-2 / Landsat /
Sentinel-1) and is the transferable backbone. A separate ``naip_encoder`` (its own
parameters) encodes only the *unmasked* patches of an auxiliary high-resolution
modality (NAIP). The latent-MIM decoder is trained on the main modalities only (it
never sees NAIP context), while a reconstructor predicts the masked NAIP pixels by
cross-attending to BOTH the main-encoder features and the separate NAIP-encoder
features.

This keeps NAIP out of the main backbone (the main encoder spends no parameters on
NAIP) while still letting the model see a little NAIP so it can reconstruct it.
"""

import logging
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
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, TokensAndMasks
from olmoearth_pretrain.nn.utils import DistributedMixins, unpack_encoder_output

logger = logging.getLogger(__name__)


class DualEncoderLatentMIM(nn.Module, DistributedMixins):
    """Latent MIM with a separate auxiliary (NAIP) encoder.

    The main encoder is exposed as ``self.encoder`` so the existing train module
    machinery (EMA target update, band dropout, patch-size checks, downstream eval)
    operates on the transferable backbone unchanged.
    """

    supports_multiple_modalities_at_once = True

    def __init__(
        self,
        encoder: nn.Module,
        naip_encoder: nn.Module,
        decoder: nn.Module,
        reconstructor: nn.Module | None = None,
        naip_modality_name: str = "naip_10",
        detach_main_context_for_reconstruction: bool = False,
    ):
        """Initialize the dual-encoder latent MIM model.

        Args:
            encoder: The main encoder (transferable backbone), excludes the NAIP modality.
            naip_encoder: The separate encoder for the NAIP modality only.
            decoder: The latent-MIM predictor, trained on the main modalities only.
            reconstructor: Reconstructor that predicts NAIP pixels, cross-attending to the
                merged (main + NAIP) encoder features.
            naip_modality_name: Name of the auxiliary modality handled by ``naip_encoder``.
            detach_main_context_for_reconstruction: If True, detach the main-encoder
                features before they are used as context for NAIP reconstruction, so the
                NAIP reconstruction loss does not backprop into the main encoder (the main
                backbone is then shaped purely by latent MIM). If False (default), the main
                encoder additionally receives gradient from NAIP reconstruction (a standard
                auxiliary-task setup); either way the main encoder has no NAIP-specific
                parameters.
        """
        super().__init__()
        self.encoder = encoder
        self.naip_encoder = naip_encoder
        self.decoder = decoder
        self.reconstructor = reconstructor
        self.naip_modality_name = naip_modality_name
        self.detach_main_context_for_reconstruction = (
            detach_main_context_for_reconstruction
        )
        # Target encoder is an EMA copy of the MAIN encoder only (NAIP is reconstructed via
        # pixels, not latent-MIM, so it needs no target).
        self.target_encoder = deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def _merge_latents(
        self, main_latent: TokensAndMasks, naip_latent: TokensAndMasks
    ) -> TokensAndMasks:
        """Merge the main-encoder and NAIP-encoder outputs into one TokensAndMasks.

        The main and NAIP encoders process disjoint modalities, so their token/mask
        fields are non-overlapping and can simply be unioned. Optionally detaches the
        main-encoder token features so NAIP reconstruction does not backprop into the
        main encoder.
        """
        main_dict = dict(main_latent.as_dict(include_nones=False))
        if self.detach_main_context_for_reconstruction:
            main_dict = {
                key: (val.detach() if not key.endswith("_mask") else val)
                for key, val in main_dict.items()
            }
        merged = main_dict
        merged.update(naip_latent.as_dict(include_nones=False))
        return TokensAndMasks(**merged)

    def forward(
        self, x: MaskedOlmoEarthSample, patch_size: int
    ) -> tuple[
        TokensAndMasks,
        TokensAndMasks,
        torch.Tensor,
        TokensAndMasks | None,
        dict[str, Any],
    ]:
        """Forward pass.

        Returns:
            latent: main-encoder embeddings (the transferable backbone output).
            decoded: latent-MIM predictions for masked main-modality tokens.
            latent_projected_and_pooled: pooled main-encoder tokens for contrastive loss.
            reconstructed: reconstructed NAIP pixels (or None if no reconstructor).
            extra_metrics: extra metrics (e.g. token norm stats).
        """
        # Main encoder over the non-NAIP modalities.
        main_output = self.encoder(x, patch_size=patch_size)
        token_norm_stats = main_output.pop("token_norm_stats", None)
        main_latent, latent_projected_and_pooled, decoder_kwargs = (
            unpack_encoder_output(main_output)
        )

        # Separate NAIP encoder over only the (unmasked) NAIP patches.
        naip_output = self.naip_encoder(x, patch_size=patch_size)
        naip_latent, _, _ = unpack_encoder_output(naip_output)

        extra_metrics: dict[str, Any] = {}
        if token_norm_stats is not None:
            extra_metrics["token_norm_stats"] = token_norm_stats

        # Latent-MIM decoder: main-modality context ONLY (no NAIP context).
        decoded = self.decoder(
            main_latent,
            timestamps=x.timestamps,
            patch_size=patch_size,
            **decoder_kwargs,
        )

        # Reconstructor: predicts NAIP pixels, cross-attending to BOTH the main-encoder
        # features and the separate NAIP-encoder features.
        reconstructed = None
        if self.reconstructor is not None:
            merged_latent = self._merge_latents(main_latent, naip_latent)
            reconstructed = self.reconstructor(merged_latent, x.timestamps, patch_size)

        return (
            main_latent,
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
        self.naip_encoder.apply_fsdp(**fsdp_config)
        self.decoder.apply_fsdp(**fsdp_config)
        self.target_encoder.apply_fsdp(**fsdp_config)
        if self.reconstructor:
            self.reconstructor.apply_fsdp(**fsdp_config)
        fully_shard(self, **fsdp_config)
        register_fsdp_forward_method(self.target_encoder, "forward")

    def apply_compile(self) -> None:
        """Apply torch.compile to the model."""
        logger.info("Applying torch.compile to the model")
        self.encoder.apply_compile()
        self.naip_encoder.apply_compile()
        self.decoder.apply_compile()
        self.target_encoder.apply_compile()
        if self.reconstructor:
            self.reconstructor.apply_compile()


@dataclass
class DualEncoderLatentMIMConfig(Config):
    """Configuration for :class:`DualEncoderLatentMIM`."""

    encoder_config: Config
    naip_encoder_config: Config
    decoder_config: Config
    reconstructor_config: Config | None = None
    naip_modality_name: str = "naip_10"
    detach_main_context_for_reconstruction: bool = False

    def _encoder_output_size(self, encoder_config: Config) -> int:
        """Get the token width an encoder emits (accounts for output projection)."""
        return encoder_config.output_embedding_size or encoder_config.embedding_size

    def validate(self) -> None:
        """Validate the configuration."""
        main_modalities = self.encoder_config.supported_modality_names
        naip = self.naip_modality_name

        # The main encoder must NOT process NAIP (that is what the separate encoder is for).
        if naip in main_modalities:
            raise ValueError(
                f"Main encoder must not support the NAIP modality '{naip}'; it is handled "
                "by the separate naip_encoder."
            )
        # The NAIP encoder must process exactly the NAIP modality.
        if self.naip_encoder_config.supported_modality_names != [naip]:
            raise ValueError(
                f"naip_encoder_config.supported_modality_names must be ['{naip}'], got "
                f"{self.naip_encoder_config.supported_modality_names}"
            )
        # The latent-MIM decoder is trained on the main modalities only (no NAIP context).
        if naip in self.decoder_config.supported_modality_names:
            raise ValueError(
                f"Latent-MIM decoder must not include the NAIP modality '{naip}' "
                "(NAIP is reconstructed via the reconstructor, not latent MIM)."
            )
        if self.decoder_config.supported_modality_names != main_modalities:
            raise ValueError(
                "Latent-MIM decoder and main encoder must support the same modalities."
            )
        if self.encoder_config.max_sequence_length != (
            self.decoder_config.max_sequence_length
        ):
            raise ValueError(
                "Main encoder and decoder must have the same max sequence length."
            )

        main_out = self._encoder_output_size(self.encoder_config)
        naip_out = self._encoder_output_size(self.naip_encoder_config)
        # Merged tokens are consumed uniformly by the decoder input projection, so both
        # encoders must emit the same token width.
        if naip_out != main_out:
            raise ValueError(
                f"NAIP encoder output width ({naip_out}) must match main encoder output "
                f"width ({main_out})."
            )
        if main_out != self.decoder_config.encoder_embedding_size:
            raise ValueError(
                "Main encoder output width must match decoder.encoder_embedding_size."
            )

        if self.reconstructor_config is not None:
            recon_predictor = self.reconstructor_config.decoder_config
            if naip not in recon_predictor.supported_modality_names:
                raise ValueError(
                    "Reconstructor's predictor must include the NAIP modality "
                    f"'{naip}' so it can decode NAIP."
                )
            if recon_predictor.encoder_embedding_size != main_out:
                raise ValueError(
                    "Reconstructor predictor.encoder_embedding_size must match the "
                    "encoder output width."
                )
            if naip not in self.reconstructor_config.supported_modality_names:
                raise ValueError(
                    f"Reconstructor must reconstruct the NAIP modality '{naip}'."
                )

    def build(self) -> "DualEncoderLatentMIM":
        """Build the dual-encoder latent MIM model."""
        self.validate()
        encoder = self.encoder_config.build()
        naip_encoder = self.naip_encoder_config.build()
        decoder = self.decoder_config.build()
        reconstructor = (
            self.reconstructor_config.build()
            if self.reconstructor_config is not None
            else None
        )
        return DualEncoderLatentMIM(
            encoder=encoder,
            naip_encoder=naip_encoder,
            decoder=decoder,
            reconstructor=reconstructor,
            naip_modality_name=self.naip_modality_name,
            detach_main_context_for_reconstruction=(
                self.detach_main_context_for_reconstruction
            ),
        )

"""LatentMIMLITE: Latent MIM with independently configured target encoder."""

import logging
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


class LatentMIMLITE(nn.Module, DistributedMixins):
    """Latent MIM with an independently configured frozen target encoder.

    Unlike LatentMIM which deepcopies the encoder to create the target encoder,
    LatentMIMLITE accepts a separately built target encoder. This allows the target
    encoder to have a completely different architecture (depth, width, heads,
    embedding size, bandsets, etc.).

    No reconstructor -- just encoder + decoder + frozen target encoder.
    """

    supports_multiple_modalities_at_once = True

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        target_encoder: nn.Module,
    ):
        """Initialize LatentMIMLITE.

        Args:
            encoder: The online encoder.
            decoder: The predictor/decoder.
            target_encoder: Independently configured target encoder (will be frozen).
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_encoder = target_encoder
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def forward(
        self, x: MaskedOlmoEarthSample, patch_size: int
    ) -> tuple[
        TokensAndMasks,
        TokensAndMasks,
        torch.Tensor,
        dict[str, Any],
    ]:
        """Forward pass.

        Returns:
            latent: Embeddings from encoder.
            decoded: Predictions from decoder for masked tokens.
            latent_projected_and_pooled: Pooled tokens for contrastive loss.
            extra_metrics: Additional metrics (e.g. token norm stats).
        """
        output_dict = self.encoder(x, patch_size=patch_size)
        token_norm_stats = output_dict.pop("token_norm_stats", None)
        latent, latent_projected_and_pooled, decoder_kwargs = unpack_encoder_output(
            output_dict
        )
        extra_metrics: dict[str, Any] = {}
        if token_norm_stats is not None:
            extra_metrics["token_norm_stats"] = token_norm_stats
        decoded = self.decoder(
            latent, timestamps=x.timestamps, patch_size=patch_size, **decoder_kwargs
        )
        return (
            latent,
            decoded,
            latent_projected_and_pooled,
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
        fully_shard(self, **fsdp_config)
        register_fsdp_forward_method(self.target_encoder, "forward")

    def apply_compile(self) -> None:
        """Apply torch.compile to the model."""
        logger.info("Applying torch.compile to LatentMIMLITE")
        self.encoder.apply_compile()
        logger.info("Applied torch.compile to the encoder")
        self.decoder.apply_compile()
        logger.info("Applied torch.compile to the decoder")
        self.target_encoder.apply_compile()
        logger.info("Applied torch.compile to the target encoder")


@dataclass
class LatentMIMLITEConfig(Config):
    """Configuration for LatentMIMLITE.

    Args:
        encoder_config: Config for the online encoder.
        decoder_config: Config for the predictor/decoder.
        target_encoder_config: Config for the independent target encoder.
    """

    encoder_config: Config
    decoder_config: Config
    target_encoder_config: Config

    def validate(self) -> None:
        """Validate the configuration."""
        # Encoder-decoder compatibility (same as LatentMIMConfig)
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
            raise ValueError("Encoder embedding size must be consistent with decoder!")

        # Target encoder dimension alignment:
        # The decoder's output_embedding_size must match the target encoder's
        # embedding_size so the loss can compare them.
        decoder_output_size = getattr(
            self.decoder_config, "output_embedding_size", None
        )
        if decoder_output_size is None:
            # PredictorConfig defaults output_embedding_size to encoder_embedding_size
            decoder_output_size = self.decoder_config.encoder_embedding_size
        if decoder_output_size != self.target_encoder_config.embedding_size:
            raise ValueError(
                f"Decoder output_embedding_size ({decoder_output_size}) must match "
                f"target_encoder embedding_size ({self.target_encoder_config.embedding_size}) "
                f"for loss computation."
            )

    def build(self) -> "LatentMIMLITE":
        """Build the LatentMIMLITE model."""
        self.validate()
        encoder = self.encoder_config.build()
        decoder = self.decoder_config.build()
        target_encoder = self.target_encoder_config.build()
        return LatentMIMLITE(
            encoder=encoder,
            decoder=decoder,
            target_encoder=target_encoder,
        )

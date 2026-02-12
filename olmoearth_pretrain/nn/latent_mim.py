"""Simple set up of latent predictor."""

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
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.nn.flexi_vit import TokensAndMasks
from olmoearth_pretrain.nn.tokenization import TokenizationConfig
from olmoearth_pretrain.nn.utils import DistributedMixins, unpack_encoder_output

logger = logging.getLogger(__name__)


class LatentMIM(nn.Module, DistributedMixins):
    """Latent MIM Style."""

    supports_multiple_modalities_at_once = True

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        reconstructor: torch.nn.Module | None = None,
        target_encoder: nn.Module | None = None,
    ):
        """Initialize the Latent MIM Style.

        Args:
            encoder: The encoder to use.
            decoder: The decoder to use.
            reconstructor: Optional reconstructor for auto-encoding.
            target_encoder: Optional separate target encoder with different
                tokenization. If None, a frozen deepcopy of the encoder is used.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.reconstructor = reconstructor

        if target_encoder is not None:
            self.target_encoder = target_encoder
            self.has_separate_target_encoder = True
        else:
            self.target_encoder = deepcopy(self.encoder)
            self.has_separate_target_encoder = False
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Cache tokenization configs for batch adaptation
        encoder_tok = getattr(self.encoder, "tokenization_config", None)
        target_tok = getattr(self.target_encoder, "tokenization_config", None)
        self._encoder_tokenization_config = encoder_tok or TokenizationConfig()
        self._target_tokenization_config = target_tok or TokenizationConfig()

    def prepare_target_batch(
        self, batch: MaskedOlmoEarthSample
    ) -> MaskedOlmoEarthSample:
        """Prepare a batch for the target encoder.

        If the target encoder uses the same tokenization as the online encoder,
        this is equivalent to ``batch.unmask()``. Otherwise, masks are adapted
        to the target tokenization config and unmasked.

        Args:
            batch: The masked batch from the dataloader.

        Returns:
            An unmasked batch with masks shaped for the target encoder's
            tokenization config.
        """
        if not self.has_separate_target_encoder:
            return batch.unmask()

        enc_tok = self._encoder_tokenization_config
        tgt_tok = self._target_tokenization_config

        updates: dict[str, Any] = {}
        for modality in batch.modalities:
            mask_name = MaskedOlmoEarthSample.get_masked_modality_name(modality)
            mask = getattr(batch, mask_name)
            if mask is None:
                continue

            num_enc_bs = enc_tok.get_num_bandsets(modality)
            num_tgt_bs = tgt_tok.get_num_bandsets(modality)

            if num_enc_bs == num_tgt_bs:
                # Same bandset count: standard unmask
                updates[mask_name] = mask * (mask == MaskValue.MISSING.value)
            else:
                # Different bandset count: detect missing, create new mask
                # Check if any bandset at each position is MISSING
                is_missing = (mask == MaskValue.MISSING.value).any(
                    dim=-1, keepdim=True
                )  # [..., 1]
                # Create new mask: ONLINE_ENCODER (0) everywhere, MISSING where missing
                new_mask = torch.where(
                    is_missing,
                    torch.tensor(
                        MaskValue.MISSING.value,
                        dtype=mask.dtype,
                        device=mask.device,
                    ),
                    torch.tensor(
                        MaskValue.ONLINE_ENCODER.value,
                        dtype=mask.dtype,
                        device=mask.device,
                    ),
                )
                # Expand to target bandset count
                updates[mask_name] = new_mask.expand(
                    *mask.shape[:-1], num_tgt_bs
                ).contiguous()

        return batch._replace(**updates)

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
    target_encoder_config: Config | None = None

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

        if self.target_encoder_config is not None:
            if (
                self.encoder_config.supported_modalities
                != self.target_encoder_config.supported_modalities
            ):
                raise ValueError(
                    "Encoder and target encoder must support the same modalities"
                )
            if (
                self.encoder_config.embedding_size
                != self.target_encoder_config.embedding_size
            ):
                raise ValueError(
                    "Encoder and target encoder must have the same embedding size"
                )

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
        target_encoder = (
            self.target_encoder_config.build()
            if self.target_encoder_config is not None
            else None
        )
        return LatentMIM(
            encoder=encoder,
            decoder=decoder,
            reconstructor=reconstructor,
            target_encoder=target_encoder,
        )

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
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.nn.flexi_vit import TokensAndMasks
from olmoearth_pretrain.nn.utils import DistributedMixins, unpack_encoder_output

logger = logging.getLogger(__name__)


class FrozenTargetProjection(nn.Module):
    """Frozen projection-only target encoder.

    When every modality exits at depth 0 and the target is never EMA-updated
    (``ema_decay=(1.0, 1.0)``), the full target-encoder copy is dead weight: the
    encoder's ``forward`` skips ``apply_attn`` entirely and the target is just the
    frozen initial projection. This module deepcopies only the pieces that
    exit-0 actually runs (``patch_embeddings`` + optional ``embedding_projector``),
    so the transformer blocks are never copied, sharded, all-gathered, or saved.

    ``project_aggregated`` is intentionally not computed: both latent-MIM train
    modules only consume ``tokens_and_masks`` from the target output.
    """

    def __init__(self, encoder: nn.Module):
        """Copy and freeze the projection submodules of ``encoder``."""
        super().__init__()
        self.patch_embeddings = deepcopy(encoder.patch_embeddings)
        self.embedding_projector = deepcopy(encoder.embedding_projector)
        for p in self.parameters():
            p.requires_grad = False

    def forward(
        self,
        x: MaskedOlmoEarthSample,
        patch_size: int,
        token_exit_cfg: dict[str, int] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Compute exit-0 targets: patch embeddings + optional projector."""
        if token_exit_cfg is not None and any(
            exit_depth > 0 for exit_depth in token_exit_cfg.values()
        ):
            raise ValueError(
                "FrozenTargetProjection only supports token_exit_cfg with all "
                f"exit depths 0, got {token_exit_cfg}. Use the full target "
                "encoder (projection_only_target=False) for deeper exits."
            )
        patchified_tokens_and_masks = self.patch_embeddings.forward(x, patch_size)
        output = TokensAndMasks(**patchified_tokens_and_masks)
        if self.embedding_projector is not None:
            output = self.embedding_projector(output)
        return {"tokens_and_masks": output}


class LatentMIM(nn.Module, DistributedMixins):
    """Latent MIM Style."""

    supports_multiple_modalities_at_once = True

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        reconstructor: torch.nn.Module | None = None,
        projection_only_target: bool = False,
    ):
        """Initialize the Latent MIM Style.

        Args:
            encoder: The encoder to use.
            decoder: The decoder to use.
            reconstructor: Optional reconstructor for auto-encoding.
            projection_only_target: If True, the target encoder is only the frozen
                initial projection (patch embeddings + optional embedding projector)
                instead of a full copy of the encoder. Only valid when all token
                exit depths are 0 and the target is never EMA-updated
                (ema_decay=(1.0, 1.0)).
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.reconstructor = reconstructor
        if projection_only_target:
            self.target_encoder: nn.Module = FrozenTargetProjection(self.encoder)
        else:
            self.target_encoder = deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

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
        if isinstance(self.target_encoder, FrozenTargetProjection):
            # Tiny frozen module: shard as a single unit (one all-gather per step)
            # instead of the per-block wrapping a full encoder copy would get.
            fully_shard(self.target_encoder, **fsdp_config)
        else:
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
        if hasattr(self.target_encoder, "apply_compile"):
            self.target_encoder.apply_compile()
            logger.info("Applied torch.compile to the target encoder")


@dataclass
class LatentMIMConfig(Config):
    """Configuration for the Latent Predictor."""

    encoder_config: Config
    decoder_config: Config
    reconstructor_config: Config | None = None
    projection_only_target: bool = False

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
        encoder_output_size = (
            self.encoder_config.output_embedding_size
            or self.encoder_config.embedding_size
        )
        if encoder_output_size != self.decoder_config.encoder_embedding_size:
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
            projection_only_target=self.projection_only_target,
        )

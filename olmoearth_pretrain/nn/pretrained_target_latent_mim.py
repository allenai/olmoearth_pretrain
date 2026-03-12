"""PretrainedTargetLatentMIM: LatentMIM with a frozen pre-trained target encoder."""

import logging
from dataclasses import dataclass
from typing import Any, cast

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
from olmoearth_pretrain.model_loader import (
    ModelID,
    load_model_from_id,
    load_model_from_path,
)
from olmoearth_pretrain.nn.flexi_vit import TokensAndMasks
from olmoearth_pretrain.nn.pretrained_target_encoder import PretrainedTargetEncoder
from olmoearth_pretrain.nn.tokenization import TokenizationConfig
from olmoearth_pretrain.nn.utils import DistributedMixins, unpack_encoder_output

logger = logging.getLogger(__name__)


def _compute_bandset_expansion(
    encoder_tokenization_config: TokenizationConfig,
    decoder_tokenization_config: TokenizationConfig,
    modality_names: list[str],
) -> dict[str, int]:
    """Compute how many bandsets each modality expands from encoder to decoder.

    Args:
        encoder_tokenization_config: The online encoder's tokenization (single bandset).
        decoder_tokenization_config: The decoder's tokenization (target encoder's, multi-bandset).
        modality_names: Modalities to compute expansion for.

    Returns:
        Dict mapping modality name to expansion factor (decoder_bandsets // encoder_bandsets).
    """
    expansion = {}
    for modality in modality_names:
        enc_n = encoder_tokenization_config.get_num_bandsets(modality)
        dec_n = decoder_tokenization_config.get_num_bandsets(modality)
        if dec_n % enc_n != 0:
            raise ValueError(
                f"Decoder bandsets ({dec_n}) must be divisible by encoder bandsets "
                f"({enc_n}) for modality {modality}"
            )
        expansion[modality] = dec_n // enc_n
    return expansion


def _expand_bandsets_tokens_and_masks(
    latent: TokensAndMasks,
    expansion: dict[str, int],
) -> TokensAndMasks:
    """Expand bandsets in TokensAndMasks from encoder to decoder dimensions.

    For each modality, repeats tokens along the bandset dimension:
    [B, H, W, T, 1, D] -> [B, H, W, T, N, D]
    And masks similarly:
    [B, H, W, T, 1] -> [B, H, W, T, N]
    """
    output_dict: dict[str, Any] = {}
    for modality in latent.modalities:
        tokens = getattr(latent, modality)
        mask_name = latent.get_masked_modality_name(modality)
        mask = getattr(latent, mask_name)

        n = expansion.get(modality, 1)
        if n > 1:
            # tokens: [..., bandsets, D] -> repeat bandsets
            tokens = tokens.repeat_interleave(n, dim=-2)
            # mask: [..., bandsets] -> repeat bandsets
            mask = mask.repeat_interleave(n, dim=-1)

        output_dict[modality] = tokens
        output_dict[mask_name] = mask

    return TokensAndMasks(**output_dict)


class PretrainedTargetLatentMIM(nn.Module, DistributedMixins):
    """LatentMIM variant using a frozen pre-trained model as the target encoder.

    Unlike LatentMIM, the target encoder is NOT a deepcopy of the online encoder —
    it's a separately loaded pre-trained model. This enables the online encoder to
    learn representations that align with a strong teacher.
    """

    supports_multiple_modalities_at_once = True

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        target_encoder: PretrainedTargetEncoder,
        encoder_to_decoder_bandset_expansion: dict[str, int],
        reconstructor: nn.Module | None = None,
    ):
        """Initialize the PretrainedTargetLatentMIM.

        Args:
            encoder: The online encoder (single bandset tokenization).
            decoder: The decoder (target encoder's multi-bandset tokenization).
            target_encoder: Frozen PretrainedTargetEncoder wrapper.
            encoder_to_decoder_bandset_expansion: Per-modality bandset expansion factors.
            reconstructor: Optional reconstructor for auto-encoding.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_encoder = target_encoder
        self.reconstructor = reconstructor
        self.encoder_to_decoder_bandset_expansion = encoder_to_decoder_bandset_expansion

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
            latent: embeddings from encoder
            decoded: predictions from decoder for masked tokens
            latent_projected_and_pooled: pooled tokens for contrastive loss
            reconstructed: MAE predictions if enabled
            extra_metrics: additional metrics dict
        """
        output_dict = self.encoder(x, patch_size=patch_size)
        token_norm_stats = output_dict.pop("token_norm_stats", None)
        latent, latent_projected_and_pooled, decoder_kwargs = unpack_encoder_output(
            output_dict
        )
        extra_metrics: dict[str, Any] = {}
        if token_norm_stats is not None:
            extra_metrics["token_norm_stats"] = token_norm_stats

        # Expand bandsets before passing to decoder
        latent_expanded = _expand_bandsets_tokens_and_masks(
            latent, self.encoder_to_decoder_bandset_expansion
        )

        reconstructed = None
        if self.reconstructor:
            reconstructed = self.reconstructor(latent, x.timestamps, patch_size)

        decoded = self.decoder(
            latent_expanded,
            timestamps=x.timestamps,
            patch_size=patch_size,
            **decoder_kwargs,
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
        # Target encoder is frozen — cast to param_dtype upfront so its parameters
        # match the mixed-precision compute dtype.  This avoids dtype mismatches
        # (e.g. bf16 input vs fp32 conv bias) that arise when nested FSDP units
        # don't propagate mixed-precision casting to child modules.
        if param_dtype is not None:
            self.target_encoder.to(dtype=param_dtype)
        # Shard the pretrained encoder (its own apply_fsdp calls fully_shard on itself),
        # then shard the wrapper so register_fsdp_forward_method works correctly.
        self.target_encoder.pretrained_encoder.apply_fsdp(**fsdp_config)
        if self.target_encoder.random_projections is not None:
            fully_shard(self.target_encoder.random_projections, **fsdp_config)
        fully_shard(self.target_encoder, **fsdp_config)
        if self.reconstructor:
            self.reconstructor.apply_fsdp(**fsdp_config)
        fully_shard(self, **fsdp_config)
        register_fsdp_forward_method(self.target_encoder, "forward")

    def apply_compile(self) -> None:
        """Apply torch.compile to the model."""
        logger.info("Applying torch.compile to the model")
        self.encoder.apply_compile()
        logger.info("Applied torch.compile to the encoder")
        self.decoder.apply_compile()
        logger.info("Applied torch.compile to the decoder")
        self.target_encoder.pretrained_encoder.apply_compile()
        logger.info("Applied torch.compile to the target encoder")


@dataclass
class PretrainedTargetLatentMIMConfig(Config):
    """Configuration for PretrainedTargetLatentMIM.

    Args:
        encoder_config: Online encoder config (single bandset).
        decoder_config: Decoder config (target encoder's tokenization).
        target_encoder_model_id: HuggingFace model ID (e.g. "OlmoEarth-v1-Base").
        target_encoder_model_path: Alternative: local/distributed checkpoint path.
        projection_only: If True, target encoder uses only patch embeddings.
        per_modality_forward: If True, separate forward pass per encodable modality.
        encodable_modality_names: Modalities processed by pretrained target encoder.
        reconstructor_config: Optional reconstructor config.
    """

    encoder_config: Config
    decoder_config: Config
    target_encoder_model_id: str | None = None
    target_encoder_model_path: str | None = None
    projection_only: bool = False
    per_modality_forward: bool = False
    encodable_modality_names: list[str] | None = None
    reconstructor_config: Config | None = None

    def validate(self) -> None:
        """Validate the configuration."""
        # Exactly one target source
        has_id = self.target_encoder_model_id is not None
        has_path = self.target_encoder_model_path is not None
        if has_id == has_path:
            raise ValueError(
                "Exactly one of target_encoder_model_id or "
                "target_encoder_model_path must be set"
            )

        # per_modality_forward requires full depth
        if self.per_modality_forward and self.projection_only:
            raise ValueError("per_modality_forward requires projection_only=False")

        # Decoder encoder_embedding_size must match online encoder
        if (
            self.encoder_config.embedding_size
            != self.decoder_config.encoder_embedding_size
        ):
            raise ValueError(
                "decoder_config.encoder_embedding_size must match "
                "encoder_config.embedding_size"
            )

    def _load_pretrained_model(self) -> nn.Module:
        """Load the pretrained model from ID or path."""
        if self.target_encoder_model_id is not None:
            model_id = ModelID(self.target_encoder_model_id)
            return load_model_from_id(model_id)
        else:
            assert self.target_encoder_model_path is not None
            return load_model_from_path(self.target_encoder_model_path)

    def _determine_modality_splits(
        self, pretrained_encoder: nn.Module
    ) -> tuple[list[str], list[str]]:
        """Determine which modalities are encodable vs decode-only.

        Returns:
            (encodable_modality_names, random_projection_modality_names)
        """
        pretrained_modalities = set(pretrained_encoder.supported_modality_names)
        all_training_modalities = set(self.encoder_config.supported_modality_names)

        if self.encodable_modality_names is not None:
            encodable = list(self.encodable_modality_names)
        else:
            # Default: modalities supported by both online encoder and pretrained encoder
            encodable = sorted(all_training_modalities & pretrained_modalities)

        # Decode-only: training modalities not handled by pretrained encoder
        decode_only = sorted(all_training_modalities - set(encodable))
        return encodable, decode_only

    def build(self) -> "PretrainedTargetLatentMIM":
        """Build the PretrainedTargetLatentMIM model."""
        self.validate()

        # Load pretrained model and extract encoder
        pretrained_model = self._load_pretrained_model()
        pretrained_encoder = pretrained_model.encoder

        # Determine modality splits
        encodable, decode_only = self._determine_modality_splits(pretrained_encoder)
        logger.info(f"Encodable modalities (pretrained target): {encodable}")
        logger.info(f"Decode-only modalities (random projections): {decode_only}")

        # Get pretrained encoder's tokenization config for random projections and decoder
        pretrained_tokenization_config = cast(
            TokenizationConfig,
            getattr(pretrained_encoder, "tokenization_config", TokenizationConfig()),
        )

        # Build PretrainedTargetEncoder wrapper
        target_encoder = PretrainedTargetEncoder(
            pretrained_encoder=pretrained_encoder,
            projection_only=self.projection_only,
            per_modality_forward=self.per_modality_forward,
            encodable_modality_names=encodable,
            random_projection_modality_names=decode_only,
            random_projection_embedding_size=pretrained_encoder.embedding_size,
            random_projection_tokenization_config=pretrained_tokenization_config,
        )

        # Build online encoder
        encoder = self.encoder_config.build()

        # Build decoder
        decoder = self.decoder_config.build()

        # Compute bandset expansion
        encoder_tokenization = cast(
            TokenizationConfig,
            getattr(encoder, "tokenization_config", TokenizationConfig()),
        )
        decoder_tokenization = cast(
            TokenizationConfig,
            getattr(decoder, "tokenization_config", pretrained_tokenization_config),
        )
        all_modalities = self.encoder_config.supported_modality_names
        expansion = _compute_bandset_expansion(
            encoder_tokenization, decoder_tokenization, all_modalities
        )
        logger.info(f"Bandset expansion mapping: {expansion}")

        # Build reconstructor
        reconstructor = (
            self.reconstructor_config.build()
            if self.reconstructor_config is not None
            else None
        )

        return PretrainedTargetLatentMIM(
            encoder=encoder,
            decoder=decoder,
            target_encoder=target_encoder,
            encoder_to_decoder_bandset_expansion=expansion,
            reconstructor=reconstructor,
        )

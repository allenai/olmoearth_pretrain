"""Retokenizing predictor for cross-tokenization latent MIM.

Wraps a standard Predictor to bridge between two different tokenization configs:
the online encoder's tokenization (config A) and the target encoder's tokenization
(config B). This enables training where the encoder uses collapsed bandsets while
the target uses the original per-resolution bandsets.
"""

import logging
from dataclasses import dataclass
from typing import Any

import torch.nn as nn
from torch import Tensor
from torch.distributed.fsdp import fully_shard

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.nn.flexi_vit import (
    Predictor,
    PredictorConfig,
    TokensAndMasks,
    get_modalities_to_process,
)
from olmoearth_pretrain.nn.tokenization import TokenizationConfig

logger = logging.getLogger(__name__)


class RetokenizingPredictor(nn.Module):
    """Predictor that re-tokenizes encoder output before decoding.

    Bridges between two tokenization configs by applying per-modality linear
    projections to convert tokens from the encoder's bandset grouping to the
    target's bandset grouping, then delegates to a standard Predictor.

    For modalities where bandset counts match, tokens pass through unchanged.
    """

    def __init__(
        self,
        predictor: Predictor,
        input_tokenization_config: TokenizationConfig,
        encoder_embedding_size: int,
        supported_modality_names: list[str],
    ):
        """Initialize the retokenizing predictor.

        Args:
            predictor: Inner predictor configured with target tokenization (config B).
            input_tokenization_config: Encoder's tokenization config (config A).
            encoder_embedding_size: Embedding dimension of the encoder output.
            supported_modality_names: Names of modalities this model supports.
        """
        super().__init__()
        self.predictor = predictor
        self.input_tokenization_config = input_tokenization_config
        self.target_tokenization_config = predictor.tokenization_config
        self.supported_modality_names = supported_modality_names
        self.encoder_embedding_size = encoder_embedding_size

        # Build per-modality retokenization projections where bandset counts differ
        self.retokenizers = nn.ModuleDict()
        self._retokenize_info: dict[str, tuple[int, int]] = {}
        for modality in supported_modality_names:
            num_input = input_tokenization_config.get_num_bandsets(modality)
            num_target = self.target_tokenization_config.get_num_bandsets(modality)
            if num_input != num_target:
                self.retokenizers[modality] = nn.Linear(
                    num_input * encoder_embedding_size,
                    num_target * encoder_embedding_size,
                )
                self._retokenize_info[modality] = (num_input, num_target)
                logger.info(
                    f"RetokenizingPredictor: {modality} "
                    f"{num_input} bandsets -> {num_target} bandsets"
                )

    def _retokenize(self, x: TokensAndMasks) -> TokensAndMasks:
        """Re-tokenize from input config A to target config B.

        For each modality needing retokenization:
        - Tokens: [..., num_input_bs, D] -> [..., num_target_bs, D]
        - Masks: [..., num_input_bs] -> [..., num_target_bs] via broadcast
        """
        if not self._retokenize_info:
            return x

        updates: dict[str, Tensor] = {}
        available_modalities = x.modalities
        modalities_to_process = get_modalities_to_process(
            available_modalities, self.supported_modality_names
        )

        for modality in modalities_to_process:
            if modality not in self._retokenize_info:
                continue

            num_input, num_target = self._retokenize_info[modality]
            tokens = getattr(x, modality)  # [..., num_input_bs, D]
            mask_name = x.get_masked_modality_name(modality)
            mask = getattr(x, mask_name)  # [..., num_input_bs]

            # Re-tokenize tokens: flatten bandset+embedding dims, project, reshape
            spatial_shape = tokens.shape[:-2]  # everything except (num_bs, D)
            d = tokens.shape[-1]
            tokens_flat = tokens.flatten(-2, -1)  # [..., num_input_bs * D]
            tokens_retok = self.retokenizers[modality](tokens_flat)
            tokens_retok = tokens_retok.view(*spatial_shape, num_target, d)
            updates[modality] = tokens_retok

            # Expand masks: [..., num_input_bs] -> [..., num_target_bs]
            # When going from fewer bandsets to more (e.g. 1 -> 3), broadcast
            if num_input == 1:
                # Simple broadcast: all target bandsets get the same mask value
                updates[mask_name] = mask.expand(
                    *mask.shape[:-1], num_target
                ).contiguous()
            else:
                # General case: repeat each input bandset mask value for its
                # corresponding target bandsets. For simplicity, use repeat_interleave
                # which works when num_target is divisible by num_input, or fall back
                # to taking the first mask value (they should be uniform per position).
                if num_target % num_input == 0:
                    factor = num_target // num_input
                    updates[mask_name] = mask.repeat_interleave(factor, dim=-1)
                else:
                    # Fall back: take the most common mask value per position
                    # For masked training, all bandsets at a spatial position typically
                    # share the same mask value, so taking index 0 is safe
                    base_mask = mask[..., :1]  # [..., 1]
                    updates[mask_name] = base_mask.expand(
                        *mask.shape[:-1], num_target
                    ).contiguous()

        # Build new TokensAndMasks with updates
        new_dict = x.as_dict(return_none=False)
        new_dict.update(updates)
        return TokensAndMasks(**new_dict)

    def forward(
        self,
        x: TokensAndMasks,
        timestamps: Tensor,
        patch_size: int,
        **kwargs: Any,
    ) -> TokensAndMasks:
        """Re-tokenize encoder output and decode.

        Args:
            x: TokensAndMasks from encoder (tokenization config A).
            timestamps: Timestamps for the data.
            patch_size: Patch size used.
            **kwargs: Additional kwargs passed to inner predictor.

        Returns:
            TokensAndMasks with target tokenization (config B).
        """
        x_retokenized = self._retokenize(x)
        return self.predictor(x_retokenized, timestamps, patch_size, **kwargs)

    def apply_fsdp(self, **fsdp_kwargs: Any) -> None:
        """Apply FSDP to the model."""
        self.predictor.apply_fsdp(**fsdp_kwargs)
        fully_shard(self, **fsdp_kwargs)

    def apply_compile(self) -> None:
        """Apply torch.compile to the model."""
        self.predictor.apply_compile()


@dataclass
class RetokenizingPredictorConfig(Config):
    """Configuration for a RetokenizingPredictor.

    The inner predictor_config should use the **target** tokenization (config B).
    The input_tokenization_config is the **encoder's** tokenization (config A).
    """

    predictor_config: PredictorConfig
    input_tokenization_config: TokenizationConfig

    def validate(self) -> None:
        """Validate the configuration."""
        self.predictor_config.validate()
        self.input_tokenization_config.validate()

    @property
    def supported_modalities(self) -> list:
        """Get the supported modalities (delegates to inner predictor)."""
        return self.predictor_config.supported_modalities

    @property
    def encoder_embedding_size(self) -> int:
        """Get the encoder embedding size (delegates to inner predictor)."""
        return self.predictor_config.encoder_embedding_size

    @property
    def max_sequence_length(self) -> int:
        """Get the max sequence length (delegates to inner predictor)."""
        return self.predictor_config.max_sequence_length

    def build(self) -> RetokenizingPredictor:
        """Build the retokenizing predictor."""
        self.validate()
        predictor = self.predictor_config.build()
        return RetokenizingPredictor(
            predictor=predictor,
            input_tokenization_config=self.input_tokenization_config,
            encoder_embedding_size=self.predictor_config.encoder_embedding_size,
            supported_modality_names=self.predictor_config.supported_modality_names,
        )

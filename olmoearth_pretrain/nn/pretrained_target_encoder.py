"""Pretrained target encoder wrapper for LatentMIM with a frozen teacher model."""

import logging
from typing import Any

import torch
import torch.nn as nn

from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, TokensAndMasks
from olmoearth_pretrain.nn.flexi_vit import (
    Encoder,
    MultiModalPatchEmbeddings,
)
from olmoearth_pretrain.nn.tokenization import TokenizationConfig

logger = logging.getLogger(__name__)


class PretrainedTargetEncoder(nn.Module):
    """Frozen wrapper around a pre-trained Encoder used as a static teacher.

    Supports three forward modes for encodable modalities:
    - Full depth, single pass (default): runs pretrained_encoder.forward() with all modalities
    - Projection only: runs only patch_embeddings.forward() (no transformer, no norm)
    - Per-modality forward: separate encoder forward per modality, then merge

    Decode-only modalities get random (but frozen) projections via a separate
    MultiModalPatchEmbeddings module.
    """

    def __init__(
        self,
        pretrained_encoder: Encoder,
        projection_only: bool = False,
        per_modality_forward: bool = False,
        encodable_modality_names: list[str] | None = None,
        random_projection_modality_names: list[str] | None = None,
        random_projection_embedding_size: int | None = None,
        random_projection_tokenization_config: TokenizationConfig | None = None,
    ):
        """Initialize the PretrainedTargetEncoder.

        Args:
            pretrained_encoder: The full pre-trained Encoder, will be frozen.
            projection_only: If True, skip transformer blocks (patch embeddings only).
            per_modality_forward: If True, separate forward pass per encodable modality.
            encodable_modality_names: Modalities that go through the pretrained encoder.
                If None, all modalities supported by the pretrained encoder are used.
            random_projection_modality_names: Decode-only modalities that get random projections.
            random_projection_embedding_size: Embedding size for random projections.
                Must match pretrained encoder embedding_size.
            random_projection_tokenization_config: Tokenization config for random projections.
        """
        super().__init__()

        self.pretrained_encoder = pretrained_encoder
        self.projection_only = projection_only
        self.per_modality_forward = per_modality_forward

        # Freeze pretrained encoder
        for p in self.pretrained_encoder.parameters():
            p.requires_grad = False
        # Disable band dropout on target encoder
        if hasattr(self.pretrained_encoder, "disable_band_dropout"):
            self.pretrained_encoder.disable_band_dropout()

        self.encodable_modality_names = (
            encodable_modality_names
            if encodable_modality_names is not None
            else list(self.pretrained_encoder.supported_modality_names)
        )

        self.random_projection_modality_names = random_projection_modality_names or []

        # Build random projections for decode-only modalities
        self.random_projections: MultiModalPatchEmbeddings | None = None
        if self.random_projection_modality_names:
            embedding_size = (
                random_projection_embedding_size
                or self.pretrained_encoder.embedding_size
            )
            tok_config = random_projection_tokenization_config or TokenizationConfig()
            self.random_projections = MultiModalPatchEmbeddings(
                supported_modality_names=self.random_projection_modality_names,
                max_patch_size=self.pretrained_encoder.max_patch_size,
                embedding_size=embedding_size,
                tokenization_config=tok_config,
                use_linear_patch_embed=self.pretrained_encoder.use_linear_patch_embed,
            )
            # Freeze random projections too
            for p in self.random_projections.parameters():
                p.requires_grad = False

    def _split_sample(self, x: MaskedOlmoEarthSample) -> tuple[list[str], list[str]]:
        """Split available modalities into encodable and decode-only."""
        available = x.modalities
        encodable = [m for m in available if m in self.encodable_modality_names]
        decode_only = [
            m for m in available if m in self.random_projection_modality_names
        ]
        return encodable, decode_only

    def _make_sub_sample(
        self, x: MaskedOlmoEarthSample, modality_names: list[str]
    ) -> MaskedOlmoEarthSample:
        """Create a sub-sample containing only the specified modalities."""
        updates: dict[str, Any] = {}
        for name in x._fields:
            if name == "timestamps":
                continue
            # Zero out modalities that aren't in the list
            base_name = name.replace("_mask", "") if name.endswith("_mask") else name
            if base_name not in modality_names and getattr(x, name) is not None:
                updates[name] = None
        return x._replace(**updates)

    def _forward_encodable_projection_only(
        self, x: MaskedOlmoEarthSample, patch_size: int
    ) -> dict[str, torch.Tensor]:
        """Run only patch embeddings (no transformer, no encodings, no norm)."""
        # Use __call__ to trigger FSDP hooks for mixed precision casting.
        # patch_embeddings is not individually FSDP-sharded, but the pretrained
        # encoder is, so we go through the encoder with projection_only logic.
        return self.pretrained_encoder.patch_embeddings(x, patch_size)

    def _forward_encodable_full(
        self,
        x: MaskedOlmoEarthSample,
        patch_size: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run full pretrained encoder forward."""
        # Use __call__ (not .forward()) to trigger FSDP hooks.
        return self.pretrained_encoder(x, patch_size=patch_size, **kwargs)

    def _forward_encodable_per_modality(
        self,
        x: MaskedOlmoEarthSample,
        encodable_modalities: list[str],
        patch_size: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run separate forward pass per encodable modality, then merge results."""
        merged_output: dict[str, Any] = {}
        for modality in encodable_modalities:
            sub_sample = self._make_sub_sample(x, [modality])
            # Use __call__ (not .forward()) to trigger FSDP hooks.
            output_dict = self.pretrained_encoder(
                sub_sample, patch_size=patch_size, **kwargs
            )
            tokens_and_masks = output_dict.get("tokens_and_masks")
            if tokens_and_masks is not None:
                tam_dict = tokens_and_masks.as_dict(include_nones=False)
                merged_output.update(tam_dict)
        return {"tokens_and_masks": TokensAndMasks(**merged_output)}

    def _forward_decode_only(
        self, x: MaskedOlmoEarthSample, patch_size: int
    ) -> dict[str, torch.Tensor]:
        """Apply random projections to decode-only modalities."""
        if self.random_projections is None:
            return {}
        # Use __call__ (not .forward()) to trigger FSDP hooks.
        return self.random_projections(x, patch_size)

    def forward(
        self,
        x: MaskedOlmoEarthSample,
        patch_size: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Forward pass matching Encoder.forward() signature.

        Args:
            x: Input sample (typically unmasked).
            patch_size: Patch size for patchification.
            **kwargs: Additional kwargs passed to encoder (e.g. token_exit_cfg).

        Returns:
            Dict with "tokens_and_masks" key containing TokensAndMasks.
        """
        encodable_modalities, decode_only_modalities = self._split_sample(x)

        # Forward for encodable modalities
        encodable_output: dict[str, Any] = {}
        if encodable_modalities:
            if self.projection_only:
                encodable_output = self._forward_encodable_projection_only(
                    x, patch_size
                )
            elif self.per_modality_forward:
                result = self._forward_encodable_per_modality(
                    x, encodable_modalities, patch_size, **kwargs
                )
                # Return directly if no decode-only modalities
                if not decode_only_modalities:
                    return result
                tam = result.get("tokens_and_masks")
                if tam is not None:
                    encodable_output = tam.as_dict(include_nones=False)
            else:
                result = self._forward_encodable_full(x, patch_size, **kwargs)
                if not decode_only_modalities:
                    return result
                tam = result.get("tokens_and_masks")
                if tam is not None:
                    encodable_output = tam.as_dict(include_nones=False)

        # Forward for decode-only modalities
        decode_only_output: dict[str, torch.Tensor] = {}
        if decode_only_modalities:
            decode_only_output = self._forward_decode_only(x, patch_size)

        # Merge all results
        merged = {}
        merged.update(encodable_output)
        merged.update(decode_only_output)

        return {"tokens_and_masks": TokensAndMasks(**merged)}

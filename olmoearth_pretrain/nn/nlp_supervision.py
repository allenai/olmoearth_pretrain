"""Text-conditioned NLP supervision decoder for map modality prediction.

Combines a ``CrossAttnDecoder`` (image tokens attend to SigLIP text
embeddings) with a per-pixel prediction head and task-aware loss (BCE for
classification, MSE for regression).

Map modalities are never tokenized by the main encoder — they are purely
ground-truth targets.  The decoder takes encoder output tokens from
encode-decode modalities (S2, S1, Landsat) and predicts map values
conditioned on a text class query.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.data.constants import MISSING_VALUE, Modality
from olmoearth_pretrain.datatypes import (
    MaskedOlmoEarthSample,
    MaskValue,
    TokensAndMasks,
)
from olmoearth_pretrain.open_set.catalog.registry import ClassEntry
from olmoearth_pretrain.open_set.model.cross_attn_decoder import (
    CrossAttnDecoder,
    CrossAttnDecoderConfig,
)

logger = logging.getLogger(__name__)

DEFAULT_REFERENCE_MODALITIES: tuple[str, ...] = (
    "sentinel2_l2a",
    "sentinel1",
    "landsat",
    "naip_10",
    "naip",
)


# ---------------------------------------------------------------------------
# Token flattening utilities (extracted from FrozenOlmoEarthEncoder)
# ---------------------------------------------------------------------------


def flatten_encoder_tokens(
    tokens_and_masks: TokensAndMasks,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, tuple[int, ...]]]:
    """Flatten per-modality ``TokensAndMasks`` into a single sequence.

    Args:
        tokens_and_masks: Encoder output with per-modality token tensors.

    Returns:
        Tuple of:
            - tokens: ``[B, N, D]`` — flat token sequence.
            - context_mask: ``[B, N]`` — True for tokens the decoder should
              attend to (ONLINE_ENCODER positions, not MISSING).
            - shapes: dict mapping modality name → per-modality token shape
              (excluding batch and embedding dims).
    """
    flat_tokens: list[torch.Tensor] = []
    flat_masks: list[torch.Tensor] = []
    shapes: dict[str, tuple[int, ...]] = {}

    for modality_name in tokens_and_masks.modalities:
        tokens = getattr(tokens_and_masks, modality_name)
        mask_name = TokensAndMasks.get_masked_modality_name(modality_name)
        mask = getattr(tokens_and_masks, mask_name)
        if tokens is None or mask is None:
            continue
        shapes[modality_name] = tuple(tokens.shape[1:-1])
        flat_tokens.append(rearrange(tokens, "b ... d -> b (...) d"))
        flat_masks.append(rearrange(mask, "b ... -> b (...)"))

    if not flat_tokens:
        raise RuntimeError("Encoder produced no tokens for this batch.")

    tokens = torch.cat(flat_tokens, dim=1)  # [B, N, D]
    masks = torch.cat(flat_masks, dim=1)  # [B, N]
    context_mask = masks == MaskValue.ONLINE_ENCODER.value
    return tokens, context_mask, shapes


def _select_reference_modality(
    shapes: dict[str, tuple[int, ...]],
    reference_modalities: tuple[str, ...] = DEFAULT_REFERENCE_MODALITIES,
) -> str:
    """Pick the first preferred spatial modality that is present."""
    for name in reference_modalities:
        if name in shapes and Modality.get(name).is_spatial:
            return name
    for name in shapes:
        if Modality.get(name).is_spatial:
            return name
    raise RuntimeError(
        f"No spatial modality present — cannot produce predictions. "
        f"Available: {list(shapes)}"
    )


def _slice_modality(
    flat_tokens: torch.Tensor,
    shapes: dict[str, tuple[int, ...]],
    modality: str,
) -> torch.Tensor:
    """Slice ``flat_tokens`` to recover the named modality's tokens.

    Returns ``[B, *shapes[modality], D]``.
    """
    offset = 0
    d = flat_tokens.shape[-1]
    for name, shape in shapes.items():
        n = 1
        for s in shape:
            n *= s
        if name == modality:
            slice_ = flat_tokens[:, offset : offset + n]
            return slice_.reshape(flat_tokens.shape[0], *shape, d)
        offset += n
    raise KeyError(f"Modality {modality!r} not found in shapes")


def _to_pixel_grid(modality_tokens: torch.Tensor) -> torch.Tensor:
    """Collapse temporal and bandset dims → ``[B, P_H, P_W, D]``."""
    if modality_tokens.ndim == 6:
        return modality_tokens.mean(dim=(3, 4))
    if modality_tokens.ndim == 5:
        return modality_tokens.mean(dim=3)
    if modality_tokens.ndim == 4:
        return modality_tokens
    raise ValueError(f"Unexpected token shape {tuple(modality_tokens.shape)}")


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------


def _compute_classification_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """BCE loss for a single (image, class) pair.

    Args:
        pred: ``[H, W]`` logits.
        target: ``[H, W]`` binary mask in {0, 1}.

    Returns:
        Scalar loss (zero-dim tensor).
    """
    # Mask out missing pixels.
    valid = target != MISSING_VALUE
    if not valid.any():
        return pred.new_zeros([])
    return F.binary_cross_entropy_with_logits(pred[valid], target[valid])


def _compute_regression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """MSE loss for a single (image, regression-entry) pair.

    Args:
        pred: ``[H, W]`` predicted values.
        target: ``[H, W]`` continuous ground-truth values (MISSING_VALUE for missing).

    Returns:
        Scalar loss.
    """
    valid = target != MISSING_VALUE
    if not valid.any():
        return pred.new_zeros([])
    return F.mse_loss(pred[valid], target[valid])


# ---------------------------------------------------------------------------
# NLPSupervisionDecoder
# ---------------------------------------------------------------------------


@dataclass
class NLPSupervisionDecoderConfig(Config):
    """Configuration for :class:`NLPSupervisionDecoder`.

    Attributes:
        decoder_config: Cross-attention decoder configuration.
        text_dim: Dimensionality of the text encoder's embeddings (SigLIP default: 1152).
        reference_modalities: Preference order for which encoder modality's
            spatial grid to use for predictions.
        text_condition_regression: If True (mode b), regression targets also go
            through the cross-attention decoder.  If False (mode a), regression
            targets use direct per-modality linear heads on encoder output.
        supervision_weight: Global weight applied to the total NLP supervision loss.
        upsample_mode: Interpolation mode for spatial downsampling.
        regression_modality_names: Names of regression modalities for mode (a)
            direct heads.  Only needed when ``text_condition_regression=False``.
    """

    decoder_config: CrossAttnDecoderConfig = field(
        default_factory=CrossAttnDecoderConfig
    )
    text_dim: int = 1152
    reference_modalities: tuple[str, ...] = DEFAULT_REFERENCE_MODALITIES
    text_condition_regression: bool = True
    supervision_weight: float = 1.0
    upsample_mode: str = "bilinear"
    regression_modality_names: list[str] = field(default_factory=list)

    def build(
        self,
        encoder_dim: int,
        max_patch_size: int,
    ) -> NLPSupervisionDecoder:
        """Build the NLP supervision decoder."""
        cross_attn_decoder = self.decoder_config.build(
            image_dim=encoder_dim,
            text_dim=self.text_dim,
        )

        regression_heads: nn.ModuleDict | None = None
        if not self.text_condition_regression and self.regression_modality_names:
            regression_heads = nn.ModuleDict(
                {
                    name: nn.Linear(encoder_dim, max_patch_size**2)
                    for name in self.regression_modality_names
                }
            )

        return NLPSupervisionDecoder(
            cross_attn_decoder=cross_attn_decoder,
            max_patch_size=max_patch_size,
            reference_modalities=self.reference_modalities,
            text_condition_regression=self.text_condition_regression,
            supervision_weight=self.supervision_weight,
            upsample_mode=self.upsample_mode,
            regression_heads=regression_heads,
        )


class NLPSupervisionDecoder(nn.Module):
    """Text-conditioned decoder for supervised map modality prediction.

    For classification entries (and regression when ``text_condition_regression=True``):
        1. Flatten encoder TokensAndMasks → ``[B, N, D]``
        2. Replicate across class queries → ``[C*B, N, D]``
        3. Run CrossAttnDecoder (Q=image, K/V=text) → ``[C*B, N, D_dec]``
        4. Extract reference modality → spatial grid
        5. Linear prediction head → per-pixel scalar predictions
        6. BCE loss (classification) or MSE loss (regression)

    For regression when ``text_condition_regression=False`` (mode a):
        Per-modality linear heads operate directly on encoder output.
    """

    def __init__(
        self,
        cross_attn_decoder: CrossAttnDecoder,
        max_patch_size: int,
        reference_modalities: tuple[str, ...] = DEFAULT_REFERENCE_MODALITIES,
        text_condition_regression: bool = True,
        supervision_weight: float = 1.0,
        upsample_mode: str = "bilinear",
        regression_heads: nn.ModuleDict | None = None,
    ) -> None:
        """Initialize the NLP supervision decoder."""
        super().__init__()
        self.cross_attn_decoder = cross_attn_decoder
        self.max_patch_size = max_patch_size
        self.reference_modalities = reference_modalities
        self.text_condition_regression = text_condition_regression
        self.supervision_weight = supervision_weight
        self.upsample_mode = upsample_mode

        # Shared prediction head for text-conditioned path.
        # Each class query produces a single-channel spatial prediction.
        self.pixel_pred_head = nn.Linear(
            cross_attn_decoder.output_dim, max_patch_size**2
        )

        # Per-modality regression heads for mode (a).
        self.regression_heads = regression_heads

    def _predict_text_conditioned(
        self,
        encoder_tokens: torch.Tensor,
        context_mask: torch.Tensor,
        shapes: dict[str, tuple[int, ...]],
        text_tokens: torch.Tensor,
        text_attn_mask: torch.Tensor | None,
        n_classes: int,
        target_size: tuple[int, int] | None,
    ) -> torch.Tensor:
        """Run the text-conditioned path for classification/regression.

        Args:
            encoder_tokens: ``[B, N, D]`` flat encoder output.
            context_mask: ``[B, N]`` bool mask.
            shapes: Per-modality token shapes.
            text_tokens: ``[C, L, D_text]`` per-token text embeddings.
            text_attn_mask: ``[C, L]`` text padding mask.
            n_classes: Number of class queries (C).
            target_size: ``(H_out, W_out)`` for output resolution.

        Returns:
            ``[C, B, H_out, W_out]`` per-pixel predictions.
        """
        b = encoder_tokens.shape[0]

        # Replicate image tokens across class queries.
        image_rep = encoder_tokens.unsqueeze(0).expand(n_classes, -1, -1, -1)
        image_rep = rearrange(image_rep, "c b n d -> (c b) n d")
        mask_rep = context_mask.unsqueeze(0).expand(n_classes, -1, -1)
        mask_rep = rearrange(mask_rep, "c b n -> (c b) n")

        # Replicate text tokens across batch.
        text_rep = text_tokens.unsqueeze(1).expand(-1, b, -1, -1)
        text_rep = rearrange(text_rep, "c b l d -> (c b) l d")
        if text_attn_mask is not None:
            text_mask_rep = text_attn_mask.bool().unsqueeze(1).expand(-1, b, -1)
            text_mask_rep = rearrange(text_mask_rep, "c b l -> (c b) l")
        else:
            text_mask_rep = None

        # Cross-attention decoder.
        refined = self.cross_attn_decoder(
            image_tokens=image_rep,
            text_tokens=text_rep,
            text_attn_mask=text_mask_rep,
            image_attn_mask=mask_rep,
        )  # [C*B, N, D_dec]

        # Extract reference modality → spatial grid.
        ref_name = _select_reference_modality(shapes, self.reference_modalities)
        ref_tokens = _slice_modality(refined, shapes, ref_name)
        ref_grid = _to_pixel_grid(ref_tokens)  # [C*B, P_H, P_W, D_dec]

        # Per-pixel prediction head.
        mps = self.max_patch_size
        raw = self.pixel_pred_head(ref_grid)  # [C*B, P_H, P_W, mps²]
        preds = rearrange(
            raw,
            "n ph pw (i j) -> n (ph i) (pw j)",
            i=mps,
            j=mps,
        )  # [C*B, P_H*mps, P_W*mps]
        out_h, out_w = preds.shape[1], preds.shape[2]

        # Downsample to target size if needed.
        if target_size is not None and target_size != (out_h, out_w):
            if target_size[0] > out_h or target_size[1] > out_w:
                raise ValueError(
                    f"target_size {target_size} exceeds native resolution "
                    f"({out_h}, {out_w}). Cannot upsample."
                )
            preds = F.interpolate(
                preds.unsqueeze(1).float(),
                size=target_size,
                mode=self.upsample_mode,
                align_corners=False if self.upsample_mode == "bilinear" else None,
            ).squeeze(1)

        return rearrange(preds, "(c b) h w -> c b h w", c=n_classes, b=b)

    def _predict_regression_direct(
        self,
        encoder_tokens: torch.Tensor,
        shapes: dict[str, tuple[int, ...]],
        modality_name: str,
        target_size: tuple[int, int] | None,
    ) -> torch.Tensor:
        """Predict regression values directly from encoder output (mode a).

        Args:
            encoder_tokens: ``[B, N, D]`` flat encoder output.
            shapes: Per-modality token shapes.
            modality_name: Which regression head to use.
            target_size: ``(H_out, W_out)`` for output resolution.

        Returns:
            ``[B, H_out, W_out]`` per-pixel predictions.
        """
        assert self.regression_heads is not None
        head = self.regression_heads[modality_name]

        ref_name = _select_reference_modality(shapes, self.reference_modalities)
        ref_tokens = _slice_modality(encoder_tokens, shapes, ref_name)
        ref_grid = _to_pixel_grid(ref_tokens)  # [B, P_H, P_W, D]

        mps = self.max_patch_size
        raw = head(ref_grid)  # [B, P_H, P_W, mps²]
        preds = rearrange(
            raw,
            "b ph pw (i j) -> b (ph i) (pw j)",
            i=mps,
            j=mps,
        )  # [B, P_H*mps, P_W*mps]
        out_h, out_w = preds.shape[1], preds.shape[2]

        if target_size is not None and target_size != (out_h, out_w):
            if target_size[0] > out_h or target_size[1] > out_w:
                raise ValueError(
                    f"target_size {target_size} exceeds native resolution "
                    f"({out_h}, {out_w}). Cannot upsample."
                )
            preds = F.interpolate(
                preds.unsqueeze(1).float(),
                size=target_size,
                mode=self.upsample_mode,
                align_corners=False if self.upsample_mode == "bilinear" else None,
            ).squeeze(1)

        return preds

    def forward(
        self,
        tokens_and_masks: TokensAndMasks,
        batch: MaskedOlmoEarthSample,
        patch_size: int,
        text_tokens: torch.Tensor,
        text_attn_mask: torch.Tensor | None,
        class_entries: list[ClassEntry],
        per_image_selections: list[list[tuple[int, ClassEntry, bool]]],
        target_size: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run NLP supervision: predict and compute loss.

        Args:
            tokens_and_masks: Encoder output (encode-decode modalities only).
            batch: Full sample including map modality raw data for GT.
            patch_size: Patch size used by the encoder.
            text_tokens: ``[C, L, D_text]`` text embeddings for the class union.
            text_attn_mask: ``[C, L]`` text padding mask.
            class_entries: Union of all sampled class entries (length C).
            per_image_selections: For each image, list of
                ``(class_index_in_union, entry, is_positive)`` tuples.
            target_size: ``(H, W)`` output resolution for predictions.

        Returns:
            Tuple of (total_loss, metrics_dict) where metrics_dict maps
            ``"nlp_supervision/{source}"`` → detached per-source loss.
        """
        # Flatten encoder tokens.
        encoder_tokens, context_mask, shapes = flatten_encoder_tokens(tokens_and_masks)

        # Split entries: which go through the CLIP decoder vs direct regression.
        if self.text_condition_regression:
            clip_entries = class_entries
            direct_regression_entries: list[ClassEntry] = []
        else:
            clip_entries = [e for e in class_entries if not e.is_regression]
            direct_regression_entries = [e for e in class_entries if e.is_regression]

        # Build index maps for the CLIP path.
        clip_entry_to_idx: dict[tuple[str, str], int] = {}
        clip_text_indices: list[int] = []
        for i, entry in enumerate(class_entries):
            if entry in clip_entries:
                clip_entry_to_idx[(entry.source, entry.text)] = len(clip_text_indices)
                clip_text_indices.append(i)

        # Run text-conditioned predictions for CLIP entries.
        clip_preds: torch.Tensor | None = None
        if clip_entries and clip_text_indices:
            clip_text_tokens = text_tokens[clip_text_indices]
            clip_text_mask = (
                text_attn_mask[clip_text_indices]
                if text_attn_mask is not None
                else None
            )
            clip_preds = self._predict_text_conditioned(
                encoder_tokens=encoder_tokens,
                context_mask=context_mask,
                shapes=shapes,
                text_tokens=clip_text_tokens,
                text_attn_mask=clip_text_mask,
                n_classes=len(clip_text_indices),
                target_size=target_size,
            )  # [C_clip, B, H, W]

        # Run direct regression predictions for mode (a).
        direct_preds: dict[str, torch.Tensor] = {}
        if direct_regression_entries and self.regression_heads is not None:
            seen_sources: set[str] = set()
            for entry in direct_regression_entries:
                if (
                    entry.source not in seen_sources
                    and entry.source in self.regression_heads
                ):
                    direct_preds[entry.source] = self._predict_regression_direct(
                        encoder_tokens=encoder_tokens,
                        shapes=shapes,
                        modality_name=entry.source,
                        target_size=target_size,
                    )  # [B, H, W]
                    seen_sources.add(entry.source)

        # Compute loss across all (image, class) assignments.
        total_loss = encoder_tokens.new_zeros([])
        per_source_loss: dict[str, float] = {}
        per_source_count: dict[str, int] = {}
        num_assignments = 0

        # Cache extracted GT maps per (source, text).
        gt_cache: dict[tuple[str, str], torch.Tensor] = {}

        for image_index, assignments in enumerate(per_image_selections):
            for class_idx_in_union, entry, _is_positive in assignments:
                # Extract ground-truth.
                gt_key = (entry.source, entry.text)
                if gt_key not in gt_cache:
                    source_tensor = getattr(batch, entry.source, None)
                    if source_tensor is None:
                        continue
                    gt_cache[gt_key] = entry.extractor(source_tensor)
                gt = gt_cache[gt_key][image_index]  # [H, W]

                # Get prediction.
                entry_key = (entry.source, entry.text)
                if entry.is_regression and not self.text_condition_regression:
                    # Mode (a): direct regression head.
                    if entry.source not in direct_preds:
                        continue
                    pred = direct_preds[entry.source][image_index]  # [H, W]
                else:
                    # CLIP decoder path.
                    if clip_preds is None or entry_key not in clip_entry_to_idx:
                        continue
                    clip_idx = clip_entry_to_idx[entry_key]
                    pred = clip_preds[clip_idx, image_index]  # [H, W]

                # Compute loss based on task type.
                if entry.is_regression:
                    pair_loss = _compute_regression_loss(pred, gt)
                else:
                    pair_loss = _compute_classification_loss(pred, gt)

                total_loss = total_loss + pair_loss
                num_assignments += 1

                # Track per-source metrics.
                src = entry.source
                per_source_loss[src] = (
                    per_source_loss.get(src, 0.0) + pair_loss.detach().item()
                )
                per_source_count[src] = per_source_count.get(src, 0) + 1

        # Normalize by number of assignments.
        if num_assignments > 0:
            total_loss = total_loss / num_assignments

        # Apply global supervision weight.
        total_loss = total_loss * self.supervision_weight

        # Build metrics.
        metrics: dict[str, Any] = {}
        for src in per_source_loss:
            count = per_source_count[src]
            if count > 0:
                metrics[f"nlp_supervision/{src}"] = per_source_loss[src] / count
        metrics["nlp_supervision/num_assignments"] = num_assignments
        metrics["nlp_supervision/total"] = total_loss.detach().item()

        return total_loss, metrics

    def apply_fsdp(
        self,
        mesh: DeviceMesh | None = None,
        mp_policy: MixedPrecisionPolicy | None = None,
        **kwargs: Any,
    ) -> None:
        """Apply FSDP sharding."""
        fsdp_config = dict(mesh=mesh, mp_policy=mp_policy)
        for layer in self.cross_attn_decoder.layers:
            fully_shard(layer, **fsdp_config)
        fully_shard(self.cross_attn_decoder, **fsdp_config)
        if self.regression_heads is not None:
            fully_shard(self.regression_heads, **fsdp_config)
        fully_shard(self, **fsdp_config)

    def apply_compile(self) -> None:
        """Apply torch.compile."""
        self.cross_attn_decoder = torch.compile(self.cross_attn_decoder)  # type: ignore[assignment]
        if self.regression_heads is not None:
            for name in list(self.regression_heads.keys()):
                self.regression_heads[name] = torch.compile(self.regression_heads[name])  # type: ignore[assignment]

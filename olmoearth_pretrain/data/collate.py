"""Collate functions for OlmoEarth Pretrain datasets."""

from __future__ import annotations

import torch

from olmoearth_pretrain.data.transform import Transform
from olmoearth_pretrain.datatypes import (
    MaskedOlmoEarthSample,
    MaskValue,
    OlmoEarthSample,
)
from olmoearth_pretrain.train.masking import MaskingStrategy

TEXT_TARGETS = ("caption_cell", "text_label")


def mask_without_captions(
    masking_strategy: MaskingStrategy, sample: OlmoEarthSample, patch_size: int
) -> MaskedOlmoEarthSample:
    """Mask the sample, carrying the text targets past the masking strategy.

    caption_cell and text_label are text targets, never model inputs: they get no
    tokens, and their masks are MISSING throughout.
    """
    targets = {name: getattr(sample, name) for name in TEXT_TARGETS}
    masked = masking_strategy.apply_mask(
        sample._replace(**dict.fromkeys(TEXT_TARGETS)), patch_size
    )
    for name, target in targets.items():
        if target is not None:
            mask = torch.full((*target.shape[:-1], 1), MaskValue.MISSING.value)
            masked = masked._replace(**{name: target, f"{name}_mask": mask})
    return masked


def collate_olmoearth_pretrain(
    batch: list[tuple[int, OlmoEarthSample]],
) -> tuple[int, OlmoEarthSample]:
    """Collate function that automatically handles any modalities present in the samples."""

    # Stack tensors while handling None values
    def stack_or_none(attr: str) -> torch.Tensor | None:
        """Stack the tensors while handling None values."""
        # For partially missing samples we use MISSING_VALUE so we only check the first sample
        if getattr(batch[0][1], attr) is None:
            return None
        stacked_tensor = torch.stack(
            [torch.from_numpy(getattr(sample, attr)) for _, sample in batch], dim=0
        )
        return stacked_tensor

    patch_size, batch_zero = batch[0]
    # Get all fields including timestamps
    sample_fields = batch_zero.modalities_with_timestamps

    # Create a dictionary of stacked tensors for each field
    collated_dict = {field: stack_or_none(field) for field in sample_fields}
    return patch_size, OlmoEarthSample(**collated_dict)


def collate_single_masked_batched(
    batch: list[tuple[int, OlmoEarthSample]],
    transform: Transform | None,
    masking_strategy: MaskingStrategy,
) -> tuple[int, MaskedOlmoEarthSample]:
    """Collate function that applies transform and masking to the full batch.

    This function first collates raw OlmoEarthSamples into a batched tensor,
    then applies transform and masking to the entire batch at once, enabling
    vectorized operations.

    Args:
        batch: List of (patch_size, OlmoEarthSample) tuples.
        transform: Optional transform to apply to the batch.
        masking_strategy: Masking strategy to apply to the batch.

    Returns:
        A tuple of (patch_size, MaskedOlmoEarthSample).
    """
    # First, collate raw samples into a batched OlmoEarthSample
    patch_size, stacked_sample = collate_olmoearth_pretrain(batch)

    # Apply transform to the batch (if configured)
    if transform is not None:
        stacked_sample = transform.apply(stacked_sample)

    # Apply masking to the batch
    masked_sample = mask_without_captions(masking_strategy, stacked_sample, patch_size)

    return patch_size, masked_sample


def collate_double_masked_batched(
    batch: list[tuple[int, OlmoEarthSample]],
    transform: Transform | None,
    masking_strategy: MaskingStrategy,
    masking_strategy_b: MaskingStrategy | None,
) -> tuple[int, MaskedOlmoEarthSample, MaskedOlmoEarthSample]:
    """Collate function that applies transform and two masking strategies to the full batch.

    This function first collates raw OlmoEarthSamples into a batched tensor,
    then applies transform and two independent masking strategies to the entire
    batch at once, enabling vectorized operations.

    Args:
        batch: List of (patch_size, OlmoEarthSample) tuples.
        transform: Optional transform to apply to the batch.
        masking_strategy: First masking strategy to apply.
        masking_strategy_b: Second masking strategy to apply. If None, uses masking_strategy.

    Returns:
        A tuple of (patch_size, MaskedOlmoEarthSample_a, MaskedOlmoEarthSample_b).
    """
    # First, collate raw samples into a batched OlmoEarthSample
    patch_size, stacked_sample = collate_olmoearth_pretrain(batch)

    # Apply transform to the batch (if configured)
    if transform is not None:
        stacked_sample = transform.apply(stacked_sample)

    # Apply both masking strategies to the batch
    masked_sample_a = mask_without_captions(
        masking_strategy, stacked_sample, patch_size
    )
    strategy_b = (
        masking_strategy_b if masking_strategy_b is not None else masking_strategy
    )
    masked_sample_b = mask_without_captions(strategy_b, stacked_sample, patch_size)

    return patch_size, masked_sample_a, masked_sample_b

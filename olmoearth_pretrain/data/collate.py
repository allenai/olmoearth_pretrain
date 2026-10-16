"""Collate functions for OlmoEarth Pretrain datasets."""

from __future__ import annotations

import torch

from olmoearth_pretrain.data.transform import Transform
from olmoearth_pretrain.datatypes import (
    MaskedOlmoEarthSample,
    OlmoEarthSample,
)
from olmoearth_pretrain.train.masking import MaskingStrategy


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
    masked_sample = masking_strategy.apply_mask(stacked_sample, patch_size)

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
    masked_sample_a = masking_strategy.apply_mask(stacked_sample, patch_size)
    strategy_b = (
        masking_strategy_b if masking_strategy_b is not None else masking_strategy
    )
    masked_sample_b = strategy_b.apply_mask(stacked_sample, patch_size)

    return patch_size, masked_sample_a, masked_sample_b


def collate_two_draw_masked_batched(
    batch: list[tuple[int, tuple[OlmoEarthSample, OlmoEarthSample] | OlmoEarthSample]],
    transform: Transform | None,
    masking_strategy: MaskingStrategy,
    masking_strategy_b: MaskingStrategy | None,
) -> tuple[int, MaskedOlmoEarthSample, MaskedOlmoEarthSample]:
    """Collate (view a, view b) pairs into two independently masked batches (E5 arm C).

    Each item is ``(patch_size, (sample_a, sample_b))`` from a dataset with
    ``allcap_two_draws=True``: two date-draws of the same sample that share the spatial
    crop and the number of timesteps, so the a-batch and the b-batch have identical
    shapes. The a-views and b-views are stacked separately, the transform (if any) is
    applied to each batch, and each is masked with its strategy (``masking_strategy_b``
    or, if None, ``masking_strategy``). Row ``i`` of both outputs is the same sample.

    A plain ``(patch_size, sample)`` item is used as both views (e.g. mock batches).

    Note: a random transform is drawn independently per call, so with e.g.
    ``flip_and_rotate`` the two views would also differ by a flip/rotation. The E5
    recipe uses no dataloader transform.

    Returns:
        A tuple of (patch_size, MaskedOlmoEarthSample_a, MaskedOlmoEarthSample_b), the
        format :class:`ContrastiveLatentMIMTrainModule`-style ``train_batch`` takes.
    """
    batch_a: list[tuple[int, OlmoEarthSample]] = []
    batch_b: list[tuple[int, OlmoEarthSample]] = []
    for patch_size, item in batch:
        if isinstance(item, OlmoEarthSample):
            sample_a = sample_b = item
        else:
            sample_a, sample_b = item
        batch_a.append((patch_size, sample_a))
        batch_b.append((patch_size, sample_b))

    patch_size, stacked_a = collate_olmoearth_pretrain(batch_a)
    _, stacked_b = collate_olmoearth_pretrain(batch_b)
    if transform is not None:
        stacked_a = transform.apply(stacked_a)
        stacked_b = transform.apply(stacked_b)

    masked_sample_a = masking_strategy.apply_mask(stacked_a, patch_size)
    strategy_b = (
        masking_strategy_b if masking_strategy_b is not None else masking_strategy
    )
    masked_sample_b = strategy_b.apply_mask(stacked_b, patch_size)
    return patch_size, masked_sample_a, masked_sample_b

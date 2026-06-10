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

METADATA_DROPOUT_VIEW_MODES = ("shared", "decorrelated")


class MetadataDropout:
    """Batch-level dropout of sample metadata (absolute year, latlon).

    Draws happen once per rank batch in the collator, so all microbatches
    sliced from the batch — and the encoder, decoder, and target-encoder
    forwards — share a single dropout condition. Per-sample draws would split
    the negative pool of the cross-sample contrastive losses into mixed
    metadata conditions, where presence alone distinguishes embeddings.

    Dropping the year sets timestamps[..., 2] = 0; year 0 is the in-band
    "year unknown" sentinel consumed by the temporal encoding (the sin/cos
    fractional-year channels survive, only absolute time is removed).
    Dropping latlon sets the field to None, so the latlon token is absent
    from the token sequence — bit-identical to eval tasks without latlon.

    view_mode controls the two contrastive views:
      - "shared": one draw, applied to the stacked sample before masking, so
        both views see the same condition.
      - "decorrelated": each view draws independently. This removes the
        metadata-matching shortcut in the instance contrastive loss (views of
        one sample share identical latlon/year, unique within a microbatch)
        at the cost of pressure toward metadata-invariant class tokens.
    """

    def __init__(
        self,
        year_dropout_rate: float = 0.0,
        latlon_dropout_rate: float = 0.0,
        view_mode: str = "shared",
    ) -> None:
        """Initialize batch-level metadata dropout.

        Args:
            year_dropout_rate: Probability that a rank batch has the absolute
                year removed from its timestamps (sentinel year 0).
            latlon_dropout_rate: Probability that a rank batch has latlon
                removed entirely (no latlon token).
            view_mode: "shared" or "decorrelated" (see class docstring).
        """
        if not 0.0 <= year_dropout_rate <= 1.0:
            raise ValueError(
                f"year_dropout_rate must be in [0, 1], got {year_dropout_rate}"
            )
        if not 0.0 <= latlon_dropout_rate <= 1.0:
            raise ValueError(
                f"latlon_dropout_rate must be in [0, 1], got {latlon_dropout_rate}"
            )
        if view_mode not in METADATA_DROPOUT_VIEW_MODES:
            raise ValueError(
                f"view_mode must be one of {METADATA_DROPOUT_VIEW_MODES}, got {view_mode}"
            )
        self.year_dropout_rate = year_dropout_rate
        self.latlon_dropout_rate = latlon_dropout_rate
        self.view_mode = view_mode

    def apply(self, sample: OlmoEarthSample) -> OlmoEarthSample:
        """Draw one batch-level condition and apply it to the stacked sample."""
        replacements: dict[str, torch.Tensor | None] = {}
        if (
            self.year_dropout_rate > 0.0
            and sample.timestamps is not None
            and bool(torch.rand(()) < self.year_dropout_rate)
        ):
            timestamps = sample.timestamps.clone()
            timestamps[..., 2] = 0
            replacements["timestamps"] = timestamps
        if (
            self.latlon_dropout_rate > 0.0
            and sample.latlon is not None
            and bool(torch.rand(()) < self.latlon_dropout_rate)
        ):
            replacements["latlon"] = None
        if replacements:
            sample = sample._replace(**replacements)
        return sample


def pin_latlon_to_encoder(
    masked_sample: MaskedOlmoEarthSample,
) -> MaskedOlmoEarthSample:
    """Force the latlon token to be encoder-visible whenever it is present.

    The latlon token is a metadata input, not a prediction target: masking
    strategies randomize non-spatial modality masks across the batch dim,
    which would make latlon a DECODER target (or hide it from the encoder)
    for a random subset of samples. Its visibility must be controlled solely
    by the batch-level MetadataDropout, so any non-MISSING mask value is
    pinned to ONLINE_ENCODER here, after the masking strategy has run.
    """
    if masked_sample.latlon is None or masked_sample.latlon_mask is None:
        return masked_sample
    mask = masked_sample.latlon_mask
    pinned = torch.where(
        mask == MaskValue.MISSING.value,
        mask,
        torch.full_like(mask, MaskValue.ONLINE_ENCODER.value),
    )
    return masked_sample._replace(latlon_mask=pinned)


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
    metadata_dropout: MetadataDropout | None = None,
) -> tuple[int, MaskedOlmoEarthSample]:
    """Collate function that applies transform and masking to the full batch.

    This function first collates raw OlmoEarthSamples into a batched tensor,
    then applies transform and masking to the entire batch at once, enabling
    vectorized operations.

    Args:
        batch: List of (patch_size, OlmoEarthSample) tuples.
        transform: Optional transform to apply to the batch.
        masking_strategy: Masking strategy to apply to the batch.
        metadata_dropout: Optional batch-level year/latlon dropout.

    Returns:
        A tuple of (patch_size, MaskedOlmoEarthSample).
    """
    # First, collate raw samples into a batched OlmoEarthSample
    patch_size, stacked_sample = collate_olmoearth_pretrain(batch)

    # Apply transform to the batch (if configured)
    if transform is not None:
        stacked_sample = transform.apply(stacked_sample)

    if metadata_dropout is not None:
        stacked_sample = metadata_dropout.apply(stacked_sample)

    # Apply masking to the batch
    masked_sample = pin_latlon_to_encoder(
        masking_strategy.apply_mask(stacked_sample, patch_size)
    )

    return patch_size, masked_sample


def collate_double_masked_batched(
    batch: list[tuple[int, OlmoEarthSample]],
    transform: Transform | None,
    masking_strategy: MaskingStrategy,
    masking_strategy_b: MaskingStrategy | None,
    metadata_dropout: MetadataDropout | None = None,
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
        metadata_dropout: Optional batch-level year/latlon dropout; its
            view_mode decides whether the two views share one draw.

    Returns:
        A tuple of (patch_size, MaskedOlmoEarthSample_a, MaskedOlmoEarthSample_b).
    """
    # First, collate raw samples into a batched OlmoEarthSample
    patch_size, stacked_sample = collate_olmoearth_pretrain(batch)

    # Apply transform to the batch (if configured)
    if transform is not None:
        stacked_sample = transform.apply(stacked_sample)

    if metadata_dropout is not None and metadata_dropout.view_mode == "decorrelated":
        stacked_sample_a = metadata_dropout.apply(stacked_sample)
        stacked_sample_b = metadata_dropout.apply(stacked_sample)
    elif metadata_dropout is not None:
        stacked_sample_a = stacked_sample_b = metadata_dropout.apply(stacked_sample)
    else:
        stacked_sample_a = stacked_sample_b = stacked_sample

    # Apply both masking strategies to the batch
    masked_sample_a = pin_latlon_to_encoder(
        masking_strategy.apply_mask(stacked_sample_a, patch_size)
    )
    strategy_b = (
        masking_strategy_b if masking_strategy_b is not None else masking_strategy
    )
    masked_sample_b = pin_latlon_to_encoder(
        strategy_b.apply_mask(stacked_sample_b, patch_size)
    )

    return patch_size, masked_sample_a, masked_sample_b

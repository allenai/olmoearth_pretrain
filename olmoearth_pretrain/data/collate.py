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

METADATA_DROPOUT_VIEW_MODES = ("shared", "decorrelated", "opposite")


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
    Dropping latlon marks the latlon token MISSING for the whole batch (see
    pin_latlon_to_encoder): MISSING tokens are removed before attention, so
    the model sees the same condition as eval tasks without latlon, while
    the latlon embedding module keeps participating in every backward —
    required for stable FSDP gradient reduction.

    view_mode controls the two contrastive views:
      - "shared": one draw, applied to the stacked sample before masking, so
        both views see the same condition.
      - "decorrelated": each view draws independently. This removes the
        metadata-matching shortcut in the instance contrastive loss (views of
        one sample share identical latlon/year, unique within a microbatch)
        at the cost of pressure toward metadata-invariant class tokens.
      - "opposite": the two views get field-wise OPPOSITE drop status — for
        each enabled field (rate > 0), exactly one view has it and the other
        has it dropped (coin flip on which view). The instance-contrastive
        positive pair is then always (present, absent) for that field, so the
        class token cannot match views by copying metadata and is forced to
        be metadata-invariant. The rate acts only as an enable flag here.
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

    def draw(self) -> tuple[bool, bool]:
        """Draw one batch-level (drop_year, drop_latlon) condition."""
        drop_year = self.year_dropout_rate > 0.0 and bool(
            torch.rand(()) < self.year_dropout_rate
        )
        drop_latlon = self.latlon_dropout_rate > 0.0 and bool(
            torch.rand(()) < self.latlon_dropout_rate
        )
        return drop_year, drop_latlon

    def draw_opposite(self) -> tuple[tuple[bool, bool], tuple[bool, bool]]:
        """Two field-wise OPPOSITE (drop_year, drop_latlon) view conditions.

        For each enabled field, exactly one of the two views drops it (coin
        flip on which); disabled fields (rate == 0) are dropped in neither.
        Returns ((year_a, latlon_a), (year_b, latlon_b)).
        """
        year_a = self.year_dropout_rate > 0.0 and bool(torch.rand(()) < 0.5)
        latlon_a = self.latlon_dropout_rate > 0.0 and bool(torch.rand(()) < 0.5)
        year_b = self.year_dropout_rate > 0.0 and not year_a
        latlon_b = self.latlon_dropout_rate > 0.0 and not latlon_a
        return (year_a, latlon_a), (year_b, latlon_b)

    @staticmethod
    def apply_year(sample: OlmoEarthSample, drop_year: bool) -> OlmoEarthSample:
        """Write the year-unknown sentinel (year 0) for the whole batch."""
        if not drop_year or sample.timestamps is None:
            return sample
        timestamps = sample.timestamps.clone()
        timestamps[..., 2] = 0
        return sample._replace(timestamps=timestamps)


def pin_latlon_to_encoder(
    masked_sample: MaskedOlmoEarthSample,
    drop_latlon: bool = False,
) -> MaskedOlmoEarthSample:
    """Control latlon token visibility after the masking strategy has run.

    The latlon token is a metadata input, not a prediction target: masking
    strategies randomize non-spatial modality masks across the batch dim,
    which would make latlon a DECODER target (or hide it from the encoder)
    for a random subset of samples. Any non-MISSING mask value is therefore
    pinned to ONLINE_ENCODER here.

    Batch-level latlon dropout sets the mask to MISSING instead of removing
    the field: MISSING tokens are removed before attention, so the model sees
    exactly the eval-time-absent condition, while the latlon embedding module
    still participates in every forward/backward. Module participation that
    flips per batch/rank destabilizes FSDP gradient reduction (observed as
    corrupted gradient shards in the adjacent decoder parameters; see the
    grad-explosion debugging notes, 2026-06-11).
    """
    if masked_sample.latlon is None or masked_sample.latlon_mask is None:
        return masked_sample
    mask = masked_sample.latlon_mask
    if drop_latlon:
        pinned = torch.full_like(mask, MaskValue.MISSING.value)
    else:
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

    drop_latlon = False
    if metadata_dropout is not None:
        drop_year, drop_latlon = metadata_dropout.draw()
        stacked_sample = metadata_dropout.apply_year(stacked_sample, drop_year)

    # Apply masking to the batch
    masked_sample = pin_latlon_to_encoder(
        masking_strategy.apply_mask(stacked_sample, patch_size),
        drop_latlon=drop_latlon,
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

    drop_a = drop_b = (False, False)
    if metadata_dropout is not None:
        if metadata_dropout.view_mode == "opposite":
            drop_a, drop_b = metadata_dropout.draw_opposite()
        elif metadata_dropout.view_mode == "decorrelated":
            drop_a = metadata_dropout.draw()
            drop_b = metadata_dropout.draw()
        else:  # "shared"
            drop_a = metadata_dropout.draw()
            drop_b = drop_a
    stacked_sample_a = (
        metadata_dropout.apply_year(stacked_sample, drop_a[0])
        if metadata_dropout is not None
        else stacked_sample
    )
    stacked_sample_b = (
        metadata_dropout.apply_year(stacked_sample, drop_b[0])
        if metadata_dropout is not None
        else stacked_sample
    )

    # Apply both masking strategies to the batch
    masked_sample_a = pin_latlon_to_encoder(
        masking_strategy.apply_mask(stacked_sample_a, patch_size),
        drop_latlon=drop_a[1],
    )
    strategy_b = (
        masking_strategy_b if masking_strategy_b is not None else masking_strategy
    )
    masked_sample_b = pin_latlon_to_encoder(
        strategy_b.apply_mask(stacked_sample_b, patch_size),
        drop_latlon=drop_b[1],
    )

    return patch_size, masked_sample_a, masked_sample_b

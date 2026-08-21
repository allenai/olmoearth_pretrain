"""Collate functions for OlmoEarth Pretrain datasets."""

from __future__ import annotations

import torch

from olmoearth_pretrain.data.cloud_mask_cache import NODATA
from olmoearth_pretrain.data.transform import Transform
from olmoearth_pretrain.datatypes import (
    MaskedOlmoEarthSample,
    OlmoEarthSample,
)
from olmoearth_pretrain.train.masking import MaskingStrategy

CLOUD_FIELDS = ("sentinel2_l2a_cloud", "landsat_cloud")


def _stack_cloud_field(
    batch: list[tuple[int, OlmoEarthSample]], name: str
) -> torch.Tensor | None:
    """Stack one ``*_cloud`` field, filling samples that lack it with NODATA.

    A sample is missing a cloud map when its h5 holds no such modality at all -- about
    2% of the OSM-sampling set has no Landsat -- so there is nothing to skip for it
    anyway: its modality mask is already all-MISSING. Filling with NODATA, which
    ``_apply_cloud_skip`` counts as NOT cloud, makes that sample a no-op while every
    other sample in the batch keeps its masking.

    This replaces an all-or-nothing rule that dropped the field for the WHOLE batch if
    any single sample lacked it. At ~98% per-sample Landsat coverage and 64 samples per
    rank that left Landsat masking active in only ~27% of batches, and silently.

    Returns None only when NO sample in the batch has the field (cloud genuinely
    disabled, e.g. an uncached run), which is the one case that should be a no-op.
    """
    arrays = [getattr(sample, name, None) for _, sample in batch]
    present = [a for a in arrays if a is not None]
    if not present:
        return None
    reference = torch.from_numpy(present[0])
    return torch.stack(
        [
            torch.from_numpy(a)
            if a is not None
            else torch.full(reference.shape, NODATA, dtype=reference.dtype)
            for a in arrays
        ],
        dim=0,
    )


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
    collated_dict: dict[str, torch.Tensor | None] = {
        field: stack_or_none(field) for field in batch_zero.modalities_with_timestamps
    }
    # modalities_with_timestamps excludes the `*_cloud` side-payloads (see
    # datatypes._modalities). We add them back here so they get stacked and,
    # crucially, transformed in lockstep with their modality (flip/rotate) -- then
    # they are pulled off by extract_cloud_payload before masking and never reach the
    # model. Stacked per-field with a NODATA fill for samples that lack one, so a
    # single Landsat-less sample no longer disables Landsat masking for the whole
    # batch (see _stack_cloud_field).
    for name in CLOUD_FIELDS:
        stacked = _stack_cloud_field(batch, name)
        if stacked is not None:
            collated_dict[name] = stacked
    return patch_size, OlmoEarthSample(**collated_dict)


def extract_cloud_payload(
    sample: OlmoEarthSample,
) -> dict[str, torch.Tensor] | None:
    """Pull the stacked `*_cloud` fields off a (post-transform) batched sample.

    Returns {modality_cloud: (B,H,W,T,1)} for the masking strategy, or None.
    """
    cloud = {
        name: getattr(sample, name)
        for name in CLOUD_FIELDS
        if getattr(sample, name, None) is not None
    }
    return cloud or None


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

    # Apply transform to the batch (if configured). Cloud rides inside
    # stacked_sample so it is flipped/rotated identically to its modality.
    if transform is not None:
        stacked_sample = transform.apply(stacked_sample)

    # Pull cloud (now transform-aligned) back out; pass it only when present so
    # other strategies / non-cloud runs keep their exact original apply_mask call.
    cloud = extract_cloud_payload(stacked_sample)
    if cloud is not None:
        masked_sample = masking_strategy.apply_mask(
            stacked_sample, patch_size, cloud=cloud
        )
    else:
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

    # Apply transform to the batch (if configured). Cloud rides inside
    # stacked_sample so it is flipped/rotated identically to its modality.
    if transform is not None:
        stacked_sample = transform.apply(stacked_sample)

    # Pull cloud (now transform-aligned) back out. Both masked views share the
    # same transformed data, so the same cloud applies to both.
    cloud = extract_cloud_payload(stacked_sample)
    strategy_b = (
        masking_strategy_b if masking_strategy_b is not None else masking_strategy
    )
    if cloud is not None:
        masked_sample_a = masking_strategy.apply_mask(
            stacked_sample, patch_size, cloud=cloud
        )
        masked_sample_b = strategy_b.apply_mask(stacked_sample, patch_size, cloud=cloud)
    else:
        masked_sample_a = masking_strategy.apply_mask(stacked_sample, patch_size)
        masked_sample_b = strategy_b.apply_mask(stacked_sample, patch_size)

    return patch_size, masked_sample_a, masked_sample_b

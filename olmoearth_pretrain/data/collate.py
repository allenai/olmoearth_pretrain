"""Collate functions for OlmoEarth Pretrain datasets."""

from __future__ import annotations

import torch

from olmoearth_pretrain.data.constants import MISSING_VALUE
from olmoearth_pretrain.data.transform import Transform
from olmoearth_pretrain.datatypes import (
    S2_CLOUD_FIELD,
    TIME_INDEX_FIELDS,
    TIME_INDEX_SUFFIX,
    MaskedOlmoEarthSample,
    MaskValue,
    OlmoEarthSample,
)
from olmoearth_pretrain.train.masking import MaskingStrategy

# An S2 token is cloudy when more than this fraction of its patch's pixels are.
CLOUDY_TOKEN_FRACTION = 0.5


def _with_time_indices(
    masked: MaskedOlmoEarthSample, sample: OlmoEarthSample
) -> MaskedOlmoEarthSample:
    """Carry ``{modality}_time_index`` fields over to the masked sample.

    Masking strategies rebuild the sample from modalities, masks and timestamps.
    """
    updates = {
        name: getattr(sample, name)
        for name in TIME_INDEX_FIELDS
        if getattr(sample, name) is not None
    }
    return masked._replace(**updates) if updates else masked


def _drop_cloudy_s2_targets(
    masked: MaskedOlmoEarthSample, sample: OlmoEarthSample, patch_size: int
) -> MaskedOlmoEarthSample:
    """Cloudy S2 decoder targets become MISSING, so the loss never asks for a cloud.

    Uses the sample's ``S2_CLOUD_FIELD`` flags (absent: no change). Cloudy S2
    tokens the masking made encoder inputs stay inputs; other modalities are
    untouched.
    """
    cloud = getattr(sample, S2_CLOUD_FIELD)
    if cloud is None:
        return masked
    mask = masked.sentinel2_l2a_mask  # [B, H, W, T_s2, bandsets]
    assert mask is not None
    b, h, w, t = cloud.shape
    p = patch_size
    fraction = cloud.float().reshape(b, h // p, p, w // p, p, t).mean(dim=(2, 4))
    cloudy = (fraction > CLOUDY_TOKEN_FRACTION).repeat_interleave(p, dim=1)
    cloudy = cloudy.repeat_interleave(p, dim=2).unsqueeze(-1)
    drop = cloudy & (mask == MaskValue.DECODER.value)
    return masked._replace(
        sentinel2_l2a_mask=mask.masked_fill(drop, MaskValue.MISSING.value)
    )


def _cap_decoder_tokens(
    masked: MaskedOlmoEarthSample, patch_size: int, max_decode_tokens: int | None
) -> MaskedOlmoEarthSample:
    """Keep at most ``max_decode_tokens`` DECODER tokens per sample, chosen at random.

    The rest become TARGET_ENCODER_ONLY (neither encoded nor decoded), which is what
    the masking already does with leftover tokens. The pick is uniform over a
    sample's decoder tokens across all modalities, so the modality mix of the targets
    is kept. Long every-capture samples otherwise carry several times v1.3's decode
    targets, and the decoder's memory and cost scale with them.
    """
    if max_decode_tokens is None:
        return masked
    names, token_masks = [], []
    for name in masked.modalities:
        mask = getattr(masked, MaskedOlmoEarthSample.get_masked_modality_name(name))
        if mask is None or mask.ndim < 4:  # non-spatial modalities are left alone
            continue
        names.append(name)
        token_masks.append(mask[:, ::patch_size, ::patch_size])
    if not names:
        return masked
    b = token_masks[0].shape[0]
    flat = torch.cat([m.reshape(b, -1) for m in token_masks], dim=1)
    is_decoder = flat == MaskValue.DECODER.value
    if int(is_decoder.sum(dim=1).max()) <= max_decode_tokens:
        return masked
    scores = torch.rand(flat.shape, device=flat.device)
    scores = scores.masked_fill(~is_decoder, float("inf"))
    k = min(max_decode_tokens, flat.shape[1])
    keep = torch.zeros_like(is_decoder)
    keep.scatter_(1, scores.topk(k, dim=1, largest=False).indices, True)
    demote = is_decoder & ~keep
    updates = {}
    start = 0
    for name, token_mask in zip(names, token_masks):
        n = token_mask[0].numel()
        drop = demote[:, start : start + n].reshape(token_mask.shape)
        start += n
        drop = drop.repeat_interleave(patch_size, dim=1).repeat_interleave(
            patch_size, dim=2
        )
        field = MaskedOlmoEarthSample.get_masked_modality_name(name)
        full = getattr(masked, field)
        drop = drop[:, : full.shape[1], : full.shape[2]]
        updates[field] = full.masked_fill(drop, MaskValue.TARGET_ENCODER_ONLY.value)
    return masked._replace(**updates)


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
        arrays = [torch.from_numpy(getattr(sample, attr)) for _, sample in batch]
        shape = tuple(max(sizes) for sizes in zip(*(a.shape for a in arrays)))
        if all(tuple(a.shape) == shape for a in arrays):
            return torch.stack(arrays, dim=0)
        # Samples assembled on per-sample timelines (per_modality_timestamps
        # datasets) differ in length: pad modalities with MISSING_VALUE (so the
        # padded steps become MISSING tokens), time indices with -1 (padding
        # slot) and timestamps with copies of the last timestamp, as the dataset
        # does when padding to max_sequence_length.
        if attr.endswith(TIME_INDEX_SUFFIX):
            fill = -1
        elif attr == S2_CLOUD_FIELD:
            fill = 0  # padding slots are MISSING anyway
        else:
            fill = MISSING_VALUE
        stacked = torch.full((len(arrays), *shape), fill, dtype=arrays[0].dtype)
        for i, array in enumerate(arrays):
            stacked[i][tuple(slice(0, n) for n in array.shape)] = array
            if attr == "timestamps" and array.shape[0]:
                stacked[i, array.shape[0] :] = array[-1]
        return stacked

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
    uint8_masks: bool = False,
    max_decode_tokens: int | None = None,
    drop_cloudy_s2_targets: bool = True,
) -> tuple[int, MaskedOlmoEarthSample]:
    """Collate function that applies transform and masking to the full batch.

    This function first collates raw OlmoEarthSamples into a batched tensor,
    then applies transform and masking to the entire batch at once, enabling
    vectorized operations.

    Args:
        batch: List of (patch_size, OlmoEarthSample) tuples.
        transform: Optional transform to apply to the batch.
        masking_strategy: Masking strategy to apply to the batch.
        uint8_masks: Send masks as uint8 (restored to int64 by ``to_device``).
        max_decode_tokens: Optional cap on decoder tokens per sample (see
            ``_cap_decoder_tokens``).
        drop_cloudy_s2_targets: With S2 cloud flags present, make cloudy S2
            decoder targets MISSING (``_drop_cloudy_s2_targets``).

    Returns:
        A tuple of (patch_size, MaskedOlmoEarthSample).
    """
    # First, collate raw samples into a batched OlmoEarthSample
    patch_size, stacked_sample = collate_olmoearth_pretrain(batch)

    # Apply transform to the batch (if configured)
    if transform is not None:
        stacked_sample = transform.apply(stacked_sample)

    # Apply masking to the batch
    masked_sample = _with_time_indices(
        masking_strategy.apply_mask(stacked_sample, patch_size), stacked_sample
    )
    if drop_cloudy_s2_targets:
        masked_sample = _drop_cloudy_s2_targets(
            masked_sample, stacked_sample, patch_size
        )
    masked_sample = _cap_decoder_tokens(masked_sample, patch_size, max_decode_tokens)
    if uint8_masks:
        masked_sample = masked_sample.with_uint8_masks()

    return patch_size, masked_sample


def collate_double_masked_batched(
    batch: list[tuple[int, OlmoEarthSample]],
    transform: Transform | None,
    masking_strategy: MaskingStrategy,
    masking_strategy_b: MaskingStrategy | None,
    uint8_masks: bool = False,
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
        uint8_masks: Send masks as uint8 (restored to int64 by ``to_device``).

    Returns:
        A tuple of (patch_size, MaskedOlmoEarthSample_a, MaskedOlmoEarthSample_b).
    """
    # First, collate raw samples into a batched OlmoEarthSample
    patch_size, stacked_sample = collate_olmoearth_pretrain(batch)

    # Apply transform to the batch (if configured)
    if transform is not None:
        stacked_sample = transform.apply(stacked_sample)

    # Apply both masking strategies to the batch
    masked_sample_a = _with_time_indices(
        masking_strategy.apply_mask(stacked_sample, patch_size), stacked_sample
    )
    strategy_b = (
        masking_strategy_b if masking_strategy_b is not None else masking_strategy
    )
    masked_sample_b = _with_time_indices(
        strategy_b.apply_mask(stacked_sample, patch_size), stacked_sample
    )
    masked_sample_a = _drop_cloudy_s2_targets(
        masked_sample_a, stacked_sample, patch_size
    )
    masked_sample_b = _drop_cloudy_s2_targets(
        masked_sample_b, stacked_sample, patch_size
    )
    if uint8_masks:
        masked_sample_a = masked_sample_a.with_uint8_masks()
        masked_sample_b = masked_sample_b.with_uint8_masks()

    return patch_size, masked_sample_a, masked_sample_b

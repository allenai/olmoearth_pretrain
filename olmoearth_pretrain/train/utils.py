"""Training utilities specific to OlmoEarth Pretrain."""

import logging
import os
from dataclasses import dataclass

import psutil
import torch

from olmoearth_pretrain.data.constants import MISSING_VALUE
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue

logger = logging.getLogger(__name__)


@dataclass
class HistogramConfig:
    """Configuration for histogram computation for a single modality.

    Args:
        num_bins: Number of histogram bins.
        categorical: If True, treat values as categorical class indices.
            If False, discretize continuous values into bins.
        min_val: Minimum value for binning (only used if categorical=False).
        max_val: Maximum value for binning (only used if categorical=False).
    """

    num_bins: int
    categorical: bool = True
    min_val: float = 0.0
    max_val: float = 1.0


def compute_histogram(
    values: torch.Tensor,
    num_bins: int,
    mask: torch.Tensor | None = None,
    categorical: bool = True,
    min_val: float = 0.0,
    max_val: float = 1.0,
) -> torch.Tensor:
    """Compute normalized histogram per sample, excluding missing values.

    Args:
        values: Input tensor of shape [B, ...] where B is batch size.
            For categorical data, values should be integer class indices.
            For continuous data, values will be discretized into bins.
        num_bins: Number of histogram bins.
        mask: Optional mask tensor of same shape as values.
            Values where mask == MaskValue.MISSING.value are excluded.
            If None, only checks for MISSING_VALUE sentinel in values.
        categorical: If True, treat values as categorical class indices.
            If False, discretize continuous values into bins.
        min_val: Minimum value for binning (only used if categorical=False).
        max_val: Maximum value for binning (only used if categorical=False).

    Returns:
        Normalized histogram tensor of shape [B, num_bins].
        If a sample has no valid values, returns uniform distribution.
    """
    batch_size = values.shape[0]
    device = values.device

    # Flatten spatial/temporal dimensions: [B, ...]  -> [B, N]
    flat_values = values.reshape(batch_size, -1)

    # Build validity mask combining both sentinel values and mask tensor
    valid_mask = flat_values != MISSING_VALUE
    if mask is not None:
        flat_mask = mask.reshape(batch_size, -1)
        # Mask shape may differ from values shape (mask is per-bandset, values per-channel)
        # Only use mask if shapes are compatible, otherwise rely on MISSING_VALUE check
        if flat_mask.shape == flat_values.shape:
            valid_mask = valid_mask & (flat_mask != MaskValue.MISSING.value)
        elif flat_mask.shape[1] < flat_values.shape[1]:
            # Mask has fewer elements (bandsets < channels)
            # Expand mask to match values by repeating along the channel dimension
            # This assumes channels are grouped into bandsets
            repeat_factor = flat_values.shape[1] // flat_mask.shape[1]
            if flat_mask.shape[1] * repeat_factor == flat_values.shape[1]:
                expanded_mask = flat_mask.repeat_interleave(repeat_factor, dim=1)
                valid_mask = valid_mask & (expanded_mask != MaskValue.MISSING.value)

    # Initialize output histogram
    histograms = torch.zeros(batch_size, num_bins, device=device, dtype=torch.float32)

    if categorical:
        # For categorical data, values are class indices
        # Clamp to valid range and convert to long
        bin_indices = flat_values.long().clamp(0, num_bins - 1)
    else:
        # For continuous data, discretize into bins
        # Normalize to [0, 1] range then scale to bin indices
        normalized = (flat_values.float() - min_val) / (max_val - min_val + 1e-8)
        bin_indices = (normalized * num_bins).long().clamp(0, num_bins - 1)

    # Count values per bin for each sample, respecting the mask
    for b in range(batch_size):
        sample_valid = valid_mask[b]
        if sample_valid.any():
            valid_indices = bin_indices[b][sample_valid]
            histograms[b] = torch.bincount(valid_indices, minlength=num_bins).float()[
                :num_bins
            ]

    # Normalize histograms (sum to 1)
    hist_sums = histograms.sum(dim=1, keepdim=True)
    # Avoid division by zero - use uniform distribution for empty samples
    empty_samples = hist_sums.squeeze(-1) == 0
    hist_sums = hist_sums.clamp(min=1.0)
    histograms = histograms / hist_sums

    # Set empty samples to uniform distribution
    if empty_samples.any():
        histograms[empty_samples] = 1.0 / num_bins

    return histograms


def compute_histograms_for_batch(
    batch: MaskedOlmoEarthSample,
    histogram_configs: dict[str, HistogramConfig],
) -> dict[str, torch.Tensor]:
    """Compute histograms for multiple modalities in a batch.

    Args:
        batch: The masked sample batch.
        histogram_configs: Dictionary mapping modality names to their histogram configs.

    Returns:
        Dictionary mapping modality names to histogram tensors [B, num_bins].
    """
    histograms = {}
    for modality_name, config in histogram_configs.items():
        # Get the raw values for this modality
        values = getattr(batch, modality_name, None)
        if values is None:
            continue

        # Get the corresponding mask
        mask_name = f"{modality_name}_mask"
        mask = getattr(batch, mask_name, None)

        histograms[modality_name] = compute_histogram(
            values=values,
            num_bins=config.num_bins,
            mask=mask,
            categorical=config.categorical,
            min_val=config.min_val,
            max_val=config.max_val,
        )

    return histograms


def split_masked_batch(
    batch: MaskedOlmoEarthSample, microbatch_size: int
) -> list[MaskedOlmoEarthSample]:
    """Split a 'batch' MaskedOlmoEarthSample into a list of micro-batches.

    Each micro-batch has a batch dimension up to microbatch_size.

    Args:
        batch (MaskedOlmoEarthSample): A MaskedOlmoEarthSample object whose first
            dimension (B) is the batch size.
        microbatch_size (int): The maximum batch size for each micro-batch.

    Returns:
        list[MaskedOlmoEarthSample]: List of MaskedOlmoEarthSample objects.
    """
    batch_size = batch.timestamps.shape[0]

    if batch_size <= microbatch_size:
        return [batch]

    num_microbatches = (batch_size + microbatch_size - 1) // microbatch_size

    # Compute split sizes (last chunk may be smaller)
    split_sizes = [microbatch_size] * (num_microbatches - 1)
    split_sizes.append(batch_size - microbatch_size * (num_microbatches - 1))

    splits: dict[str, tuple] = {}
    for field in batch._fields:
        data = getattr(batch, field)
        if data is not None:
            splits[field] = data.split(split_sizes, dim=0)

    # Build microbatches
    return [
        MaskedOlmoEarthSample(**{f: chunks[i] for f, chunks in splits.items()})
        for i in range(num_microbatches)
    ]


def log_memory_usage_for_process(process: psutil.Process) -> tuple[int, int, int, int]:
    """Log memory usage for a given process and return memory stats."""
    try:
        memory_info = process.memory_info()
        rss = memory_info.rss
        pss = 0
        uss = 0
        shared = 0

        # Iterate over memory maps
        for mmap in process.memory_maps():
            pss += mmap.pss
            uss += mmap.private_clean + mmap.private_dirty
            shared += mmap.shared_clean + mmap.shared_dirty

        return rss, pss, uss, shared

    except psutil.NoSuchProcess:
        # The process may have terminated between the time we got the list and now
        return 0, 0, 0, 0


def log_total_memory_usage() -> float:
    """Log total memory usage for the main process and its children."""
    # Get the current process (main process)
    main_process = psutil.Process(os.getpid())

    # Initialize total memory usage counters
    total_rss = 0
    total_pss = 0
    total_uss = 0
    total_shared = 0

    # Log memory usage for the main process
    logger.info("Logging memory usage for main process")
    rss, pss, uss, shared = log_memory_usage_for_process(main_process)
    total_rss += rss
    total_pss += pss
    total_uss += uss
    total_shared += shared

    # Iterate over child processes and log their memory usage
    logger.info("Logging memory usage for child processes")
    for child in main_process.children(recursive=True):
        rss, pss, uss, shared = log_memory_usage_for_process(child)
        total_rss += rss
        total_pss += pss
        total_uss += uss
        total_shared += shared

    return total_pss / (1024 * 1024 * 1024)

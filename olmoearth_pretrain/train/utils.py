"""Training utilities specific to OlmoEarth Pretrain."""

import logging
import os

import psutil
import torch

from olmoearth_pretrain.data.dataset import OlmoEarthSample
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue

logger = logging.getLogger(__name__)


def check_input(masked_batch: MaskedOlmoEarthSample, patch_size: int) -> None:
    """Check if any samples in a batch don't have encoder or decoder tokens.

    We raise an error if this is the case.
    """
    encoded: torch.Tensor | None = None
    decoded: torch.Tensor | None = None
    shape: torch.Size | None = None
    for key, value in masked_batch.as_dict(return_none=False).items():
        if key.endswith("mask"):
            assert isinstance(value, torch.Tensor)
            flat_mask = torch.flatten(value, start_dim=1)
            encoded_for_modality = (flat_mask == MaskValue.ONLINE_ENCODER.value).sum(
                dim=-1
            )
            decoded_for_modality = (flat_mask == MaskValue.DECODER.value).sum(dim=-1)
            if encoded is None:
                encoded = encoded_for_modality
            else:
                encoded += encoded_for_modality

            if decoded is None:
                decoded = decoded_for_modality
            else:
                decoded += decoded_for_modality

            if shape is None:
                shape = value.shape
            elif len(value.shape) > len(shape):
                shape = value.shape

    if (encoded_for_modality == 0).any():
        raise RuntimeError(
            f"Got 0 encoded tokens for batch with {masked_batch.modalities}, "
            f"with shape {shape} and patch size {patch_size}"
        )
    if (decoded_for_modality == 0).any():
        raise RuntimeError(
            f"Got 0 decoded tokens for batch with {masked_batch.modalities}, "
            f"with shape {shape} and patch size {patch_size}"
        )


def split_batch(batch: OlmoEarthSample, microbatch_size: int) -> list[OlmoEarthSample]:
    """Split a 'batch' OlmoEarthSample into a list of micro-batches.

    Each micro-batch has a batch dimension up to microbatch_size.

    Args:
        batch (OlmoEarthSample): A OlmoEarthSample object whose first dimension (B) is the batch size.
        microbatch_size (int): The maximum batch size for each micro-batch.

    Returns:
        list[OlmoEarthSample]: List of OlmoEarthSample objects.
    """
    batch_size = batch.batch_size

    # If the batch is already small enough, no need to split.
    if batch_size <= microbatch_size:
        return [batch]

    # Calculate how many micro-batches we need.
    num_microbatches = (batch_size + microbatch_size - 1) // microbatch_size
    microbatches = []

    # Convert the OlmoEarthSample to a dictionary so we can slice each field if present.
    batch_dict = batch.as_dict(ignore_nones=True)

    for mb_idx in range(num_microbatches):
        start = mb_idx * microbatch_size
        end = min(start + microbatch_size, batch_size)

        # Create a new dict for the sliced data
        microbatch_dict = {}
        for field_name, data in batch_dict.items():
            assert data is not None
            # Otherwise, assume the first dimension is batch dimension and slice it
            microbatch_dict[field_name] = data[start:end]

        # Create a new OlmoEarthSample from the sliced fields
        microbatches.append(OlmoEarthSample(**microbatch_dict))

    return microbatches


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

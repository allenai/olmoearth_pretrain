"""Training utilities specific to OlmoEarth Pretrain."""

import logging
import os

import psutil

from olmoearth_pretrain.data.dataset import OlmoEarthSample
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample

logger = logging.getLogger(__name__)


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

    # Split all non-None fields at once using torch.split
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

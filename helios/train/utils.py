"""Training utilities specific to Helios."""

from helios.data.dataset import HeliosSample


def split_batch(batch: HeliosSample, microbatch_size: int) -> list[HeliosSample]:
    """Split a 'batch' HeliosSample into a list of micro-batches.

    Each micro-batch has a batch dimension up to microbatch_size.

    Args:
        batch (HeliosSample): A HeliosSample object whose first dimension (B) is the batch size.
        microbatch_size (int): The maximum batch size for each micro-batch.

    Returns:
        list[HeliosSample]: List of HeliosSample objects.
    """
    batch_size = batch.batch_size

    # If the batch is already small enough, no need to split.
    if batch_size <= microbatch_size:
        return [batch]

    # Calculate how many micro-batches we need.
    num_microbatches = (batch_size + microbatch_size - 1) // microbatch_size
    microbatches = []

    # Convert the HeliosSample to a dictionary so we can slice each field if present.
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

        # Create a new HeliosSample from the sliced fields
        microbatches.append(HeliosSample(**microbatch_dict))

    return microbatches

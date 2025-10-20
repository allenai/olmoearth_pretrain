"""Test the OlmoEarthDataset class compute_norm_stats method."""

import logging

import numpy as np

from olmoearth_pretrain.data.utils import update_streaming_stats

logger = logging.getLogger(__name__)


def test_streaming_stats_multiple_batches() -> None:
    """Test the streaming stats update method."""
    # Generate random data for multiple batches
    batch1 = np.random.rand(10, 10, 3)
    batch2 = np.random.rand(10, 10, 3)
    batch3 = np.random.rand(10, 10, 3)
    batches = [batch1, batch2, batch3]

    # Compute true mean and std using all data
    all_data = np.concatenate(batches, axis=0)
    true_mean = all_data.mean()
    true_std = all_data.std()

    # Initialize streaming stats
    current_count = 0
    current_mean = 0.0
    current_var = 0.0

    # Update stats using streaming method
    for batch in batches:
        current_count, current_mean, current_var = update_streaming_stats(
            current_count, current_mean, current_var, batch
        )

    # Compute std from variance
    streaming_std = (current_var / current_count) ** 0.5

    # Assertions
    assert np.isclose(current_mean, true_mean, atol=1e-5)
    assert np.isclose(streaming_std, true_std, atol=1e-5)

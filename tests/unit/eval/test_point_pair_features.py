"""Tests for the point-pair embedding features."""

import torch

from olmoearth_pretrain.evals.embeddings import point_pair_features
from olmoearth_pretrain.nn.pooling import PoolingType


def test_gathers_tokens_under_points_and_pools() -> None:
    """Points map to the grid cell containing them; the grid mean is appended."""
    grid = torch.arange(2 * 4 * 4 * 3, dtype=torch.float32).reshape(2, 4, 4, 3)
    # Window fractions: (0.1, 0.9) -> cell (0, 3); (0.6, 0.3) -> cell (2, 1).
    labels = torch.tensor(
        [[1, 0.1, 0.9, 0.6, 0.3], [0, 0.99, 0.01, 0.5, 0.5]], dtype=torch.float32
    )
    features, classes = point_pair_features(grid, labels, PoolingType.MEAN)

    assert features.shape == (2, 9)
    torch.testing.assert_close(features[0, :3], grid[0, 0, 3])
    torch.testing.assert_close(features[0, 3:6], grid[0, 2, 1])
    torch.testing.assert_close(features[1, :3], grid[1, 3, 0])
    torch.testing.assert_close(features[1, 3:6], grid[1, 2, 2])
    torch.testing.assert_close(features[:, 6:], grid.mean(dim=(1, 2)))
    assert classes.tolist() == [1, 0]
    assert classes.dtype == torch.long

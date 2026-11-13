"""Tests for the point-pair embedding features."""

import torch

from olmoearth_pretrain.evals.datasets.configs import TaskType
from olmoearth_pretrain.evals.embeddings import get_embeddings, point_pair_features
from olmoearth_pretrain.evals.eval_wrapper import OlmoEarthEvalWrapper
from olmoearth_pretrain.nn.pooling import PoolingType
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample


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


def test_get_embeddings_handles_point_pair_classification() -> None:
    """Classification pools the grid, while point-pair classification gathers from it."""
    batch_size, height, width, dim = 2, 4, 4, 3
    grid = torch.arange(
        batch_size * height * width * dim, dtype=torch.float32
    ).reshape(batch_size, height, width, dim)
    sample = MaskedOlmoEarthSample(
        timestamps=torch.zeros(batch_size, 1, 3, dtype=torch.long)
    )
    labels = torch.tensor(
        [[1, 0.1, 0.9, 0.6, 0.3], [0, 0.99, 0.01, 0.5, 0.5]],
        dtype=torch.float32,
    )

    class LocationTokenModel(torch.nn.Module):
        use_attention_pooler = True

        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(
            self,
            sample: MaskedOlmoEarthSample,
            patch_size: int,
            fast_pass: bool,
        ) -> dict[str, torch.Tensor]:
            return {"location_tokens": grid}

    def wrapper(point_pair: bool = False) -> OlmoEarthEvalWrapper:
        return OlmoEarthEvalWrapper(
            model=LocationTokenModel(),
            task_type=TaskType.CLASSIFICATION,
            patch_size=1,
            pooling_type=PoolingType.MEAN,
            point_pair=point_pair,
        )

    classes = labels[:, 0].long()
    pooled, pooled_classes = get_embeddings([(sample, classes)], wrapper())
    assert pooled.shape == (batch_size, dim)
    torch.testing.assert_close(pooled, grid.mean(dim=(1, 2)))
    assert pooled_classes.shape == (batch_size,)
    assert pooled_classes.dtype == torch.long

    pair_features, pair_classes = get_embeddings(
        [(sample, labels)], wrapper(point_pair=True)
    )
    rows = (labels[:, 1::2] * height).long().clamp(0, height - 1)
    cols = (labels[:, 2::2] * width).long().clamp(0, width - 1)
    batch = torch.arange(batch_size)
    expected = torch.cat(
        (
            grid[batch, rows[:, 0], cols[:, 0]],
            grid[batch, rows[:, 1], cols[:, 1]],
            grid.mean(dim=(1, 2)),
        ),
        dim=-1,
    )
    assert pair_features.shape == (batch_size, 3 * dim)
    torch.testing.assert_close(pair_features, expected)
    assert pair_classes.shape == (batch_size,)
    assert pair_classes.dtype == torch.long
    torch.testing.assert_close(pair_classes, classes)

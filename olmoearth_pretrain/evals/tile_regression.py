"""Tile-level regression readouts over tiled (``tile_samples``) embeddings.

Some registry datasets carry labels that describe a whole stored sample rather than
a pixel -- e.g. the US socioeconomic eval, where each 128 x 128 window has a median
income, home value and population density. Under the per-pixel embedding-product
convention those samples are tiled into ``window_size`` crops (``tile_samples``) and
each crop is embedded at patch size 1. This module

* mean-pools each sample's crop embeddings back to one vector per sample
  (:func:`pool_tiles`), and
* fits one closed-form ridge per label property, reading every property straight
  from the windows' vector label layer (:func:`window_label_matrix`,
  :func:`ridge_readouts`),

so several labels share a single embedding pass instead of one pass per task.

Tiled loaders are never shuffled for embedding extraction, so a sample's crops
arrive as consecutive rows, in the order of the rslearn dataset's examples.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import torch

from olmoearth_pretrain.evals.metrics import EvalMetric, EvalResult

if TYPE_CHECKING:
    from rslearn.dataset import Window

logger = logging.getLogger(__name__)

# Ridge penalties searched on val, relative to the (standardized) feature count.
RIDGE_ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
TASK_SUFFIX = "tile_ridge"


def tile_ridge_task_name(host_task: str, prop: str) -> str:
    """Synthetic task name a property's ridge readout is reported under."""
    return f"{host_task}_{TASK_SUFFIX}_{prop}"


def pool_tiles(
    embeddings: torch.Tensor, labels: torch.Tensor, tiles_per_sample: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-pool consecutive groups of tile embeddings into one row per sample.

    Labels are per-sample, so every tile of a sample carries the same label and the
    first tile's is kept.
    """
    n = embeddings.shape[0]
    if tiles_per_sample < 1 or n % tiles_per_sample:
        raise ValueError(
            f"{n} tile embeddings do not split into groups of {tiles_per_sample}"
        )
    pooled = embeddings.float().reshape(n // tiles_per_sample, tiles_per_sample, -1)
    grouped_labels = labels.reshape(n // tiles_per_sample, tiles_per_sample, -1)
    return pooled.mean(dim=1), grouped_labels[:, 0].reshape(
        n // tiles_per_sample, *labels.shape[1:]
    )


def window_label_matrix(
    windows: list[Window], layer: str, properties: list[str]
) -> torch.Tensor:
    """(num_windows, num_properties) label matrix; NaN where a window lacks one."""
    from rslearn.utils.vector_format import GeojsonVectorFormat

    fmt = GeojsonVectorFormat()
    out = torch.full((len(windows), len(properties)), float("nan"))
    for i, window in enumerate(windows):
        features = fmt.decode_vector(
            window.get_layer_dir(layer), window.projection, window.bounds
        )
        props = features[0].properties if features else {}
        for j, prop in enumerate(properties):
            value = props.get(prop)
            if value is not None:
                out[i, j] = float(value)
    return out


def _fit_ridge(x: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    """Ridge weights (with bias as the last row) for standardized features x."""
    xb = torch.cat([x, torch.ones(x.shape[0], 1, dtype=x.dtype)], dim=1)
    reg = alpha * x.shape[1] * torch.eye(xb.shape[1], dtype=x.dtype)
    reg[-1, -1] = 0.0  # don't penalize the bias
    return torch.linalg.solve(xb.T @ xb + reg, xb.T @ y)


def _predict(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, torch.ones(x.shape[0], 1, dtype=x.dtype)], dim=1) @ w


def _regression_metrics(pred: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    err = pred - y
    total = ((y - y.mean()) ** 2).sum()
    return {
        EvalMetric.MAE.value: err.abs().mean().item(),
        EvalMetric.RMSE.value: math.sqrt((err**2).mean().item()),
        EvalMetric.R2.value: (1.0 - (err**2).sum() / (total + 1e-12)).item(),
    }


def ridge_readouts(
    embeddings_by_split: dict[str, torch.Tensor | None],
    labels_by_split: dict[str, torch.Tensor | None],
    properties: list[str],
    alphas: tuple[float, ...] = RIDGE_ALPHAS,
) -> dict[str, tuple[EvalResult, EvalResult | None]]:
    """Per property: ridge fit on train, alpha picked by val R^2, scored on val/test.

    Rows whose label is NaN are dropped per property. Features are standardized with
    train statistics. Returns {property: (val_result, test_result or None)}, primary
    metric R^2.
    """
    train_x = embeddings_by_split["train"].double()  # type: ignore[union-attr]
    mean, std = train_x.mean(0), train_x.std(0).clamp_min(1e-8)

    def standardize(x: torch.Tensor) -> torch.Tensor:
        return (x.double() - mean) / std

    xs: dict[str, Any] = {
        split: standardize(x) if x is not None else None
        for split, x in embeddings_by_split.items()
    }
    results: dict[str, tuple[EvalResult, EvalResult | None]] = {}
    for j, prop in enumerate(properties):

        def rows(split: str) -> tuple[torch.Tensor, torch.Tensor] | None:
            x, y = xs.get(split), labels_by_split.get(split)
            if x is None or y is None:
                return None
            yj = y[:, j].double()
            keep = ~torch.isnan(yj)
            return x[keep], yj[keep]

        train, val, test = rows("train"), rows("val"), rows("test")
        assert train is not None and val is not None
        best: tuple[float, float, torch.Tensor] | None = None
        for alpha in alphas:
            w = _fit_ridge(train[0], train[1], alpha)
            r2 = _regression_metrics(_predict(val[0], w), val[1])[EvalMetric.R2.value]
            if best is None or r2 > best[0]:
                best = (r2, alpha, w)
        assert best is not None
        _, alpha, w = best

        def result(split_rows: tuple[torch.Tensor, torch.Tensor]) -> EvalResult:
            metrics = _regression_metrics(_predict(split_rows[0], w), split_rows[1])
            metrics["alpha"] = alpha
            metrics["n"] = float(split_rows[1].shape[0])
            return EvalResult(
                primary=metrics[EvalMetric.R2.value],
                primary_metric=EvalMetric.R2,
                primary_metric_key=EvalMetric.R2.value,
                metrics=metrics,
            )

        val_result = result(val)
        results[prop] = (val_result, result(test) if test is not None else None)
        logger.info(
            f"Tile ridge [{prop}]: val R2 {val_result.primary:.4f} "
            f"(alpha {alpha}, n_train {train[1].shape[0]}, n_val {val[1].shape[0]})"
        )
    return results

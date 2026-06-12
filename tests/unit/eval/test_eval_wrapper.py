"""Unit tests for eval wrapper call contracts."""

import pytest
import torch
from torch import nn

from olmoearth_pretrain.evals.datasets.configs import TaskType
from olmoearth_pretrain.evals.eval_wrapper import (
    ClayEvalWrapper,
    CromaEvalWrapper,
    DINOv3EvalWrapper,
    EvalWrapper,
    GalileoEvalWrapper,
    PanopticonEvalWrapper,
    PrestoEvalWrapper,
    PrithviV2EvalWrapper,
    SatlasEvalWrapper,
    TerramindEvalWrapper,
    TesseraEvalWrapper,
)
from olmoearth_pretrain.nn.pooling import PoolingType


class _RecordingAdapter(nn.Module):
    """Minimal adapter that records eval-wrapper pooling arguments."""

    def __init__(self) -> None:
        super().__init__()
        self.embeddings = torch.ones(2, 3)
        self.calls: list[tuple[object, PoolingType, bool]] = []

    def forward(
        self,
        sample: object,
        *,
        pooling: PoolingType,
        spatial_pool: bool,
    ) -> torch.Tensor:
        self.calls.append((sample, pooling, spatial_pool))
        return self.embeddings


class _ForwardFeaturesAdapter(nn.Module):
    """Minimal adapter that records which embedding method was called."""

    def __init__(self) -> None:
        super().__init__()
        self.embeddings = torch.ones(2, 3)
        self.calls: list[tuple[str, object, PoolingType]] = []

    def forward(self, sample: object, *, pooling: PoolingType) -> torch.Tensor:
        self.calls.append(("forward", sample, pooling))
        return self.embeddings

    def forward_features(
        self,
        sample: object,
        *,
        pooling: PoolingType,
    ) -> torch.Tensor:
        self.calls.append(("forward_features", sample, pooling))
        return self.embeddings


@pytest.mark.parametrize(
    "wrapper_class",
    [
        TerramindEvalWrapper,
        GalileoEvalWrapper,
        PrithviV2EvalWrapper,
        ClayEvalWrapper,
        CromaEvalWrapper,
        PrestoEvalWrapper,
        SatlasEvalWrapper,
        TesseraEvalWrapper,
    ],
)
def test_simple_adapter_wrappers_forward_pooling_arguments(
    wrapper_class: type[EvalWrapper],
) -> None:
    """Simple adapter wrappers should share the same embedding-call contract."""
    model = _RecordingAdapter()
    wrapper = wrapper_class(
        model=model,
        task_type=TaskType.CLASSIFICATION,
        patch_size=8,
        pooling_type=PoolingType.MEAN,
    )
    sample = object()
    labels = torch.arange(2)

    embeddings, returned_labels = wrapper(sample, labels)

    assert embeddings is model.embeddings
    assert returned_labels is labels
    assert model.calls == [(sample, PoolingType.MEAN, False)]


@pytest.mark.parametrize(
    ("task_type", "expected_spatial_pool"),
    [
        (TaskType.CLASSIFICATION, False),
        (TaskType.SEGMENTATION, True),
        (TaskType.REGRESSION, True),
    ],
)
def test_adapter_wrapper_spatial_pool_follows_task_type(
    task_type: TaskType,
    expected_spatial_pool: bool,
) -> None:
    """Spatial pooling is a base wrapper decision, not per-adapter behavior."""
    model = _RecordingAdapter()
    wrapper = TerramindEvalWrapper(
        model=model,
        task_type=task_type,
        patch_size=8,
        pooling_type=PoolingType.MEAN,
    )

    wrapper(object(), torch.arange(2))

    assert model.calls[0][2] is expected_spatial_pool


@pytest.mark.parametrize("wrapper_class", [PanopticonEvalWrapper, DINOv3EvalWrapper])
@pytest.mark.parametrize(
    ("task_type", "expected_call"),
    [
        (TaskType.CLASSIFICATION, "forward"),
        (TaskType.SEGMENTATION, "forward_features"),
    ],
)
def test_forward_features_wrappers_use_dense_features_for_spatial_tasks(
    wrapper_class: type[EvalWrapper],
    task_type: TaskType,
    expected_call: str,
) -> None:
    """Some adapters use a separate dense-feature path for spatial tasks."""
    model = _ForwardFeaturesAdapter()
    wrapper = wrapper_class(
        model=model,
        task_type=task_type,
        patch_size=8,
        pooling_type=PoolingType.MEAN,
    )
    sample = object()
    labels = torch.arange(2)

    embeddings, returned_labels = wrapper(sample, labels)

    assert embeddings is model.embeddings
    assert returned_labels is labels
    assert model.calls == [(expected_call, sample, PoolingType.MEAN)]

"""Unit tests for the precomputed embedding baseline (e.g. AlphaEarth/GSE)."""

import pytest
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.evals.datasets.configs import TaskType
from olmoearth_pretrain.evals.eval_wrapper import (
    PrecomputedEmbeddingEvalWrapper,
    get_eval_wrapper,
)
from olmoearth_pretrain.evals.models import PrecomputedEmbedding
from olmoearth_pretrain.nn.pooling import PoolingType
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

B, H, W, C = 2, 4, 4, 64


def _sample_with_gse(gse: torch.Tensor | None) -> MaskedOlmoEarthSample:
    return MaskedOlmoEarthSample(
        timestamps=torch.zeros(B, 1, 3, dtype=torch.long),
        gse=gse,
    )


class TestPrecomputedEmbedding:
    """Tests for the PrecomputedEmbedding baseline module."""

    def test_rejects_unknown_modality(self) -> None:
        """Unknown modality names raise at construction."""
        with pytest.raises(ValueError, match="Unknown modality"):
            PrecomputedEmbedding(modality="not_a_modality")

    def test_missing_modality_on_sample_raises(self) -> None:
        """A sample without the embedding modality raises a clear error."""
        model = PrecomputedEmbedding(modality=Modality.GSE.name)
        with pytest.raises(ValueError, match="missing precomputed embedding"):
            model(_sample_with_gse(None))

    def test_pooled_output(self) -> None:
        """Without spatial_pool the embeddings are pooled to (B, C)."""
        model = PrecomputedEmbedding(modality=Modality.GSE.name)
        gse = torch.randn(B, H, W, 1, C)
        out = model(_sample_with_gse(gse), pooling=PoolingType.MEAN)
        assert out.shape == (B, C)
        torch.testing.assert_close(out, gse.squeeze(3).mean(dim=(1, 2)))

    def test_spatial_output(self) -> None:
        """With spatial_pool the (B, H, W, C) grid is preserved."""
        model = PrecomputedEmbedding(modality=Modality.GSE.name)
        gse = torch.randn(B, H, W, 1, C)
        out = model(_sample_with_gse(gse), spatial_pool=True)
        assert out.shape == (B, H, W, C)
        torch.testing.assert_close(out, gse.squeeze(3))

    def test_multi_timestep_is_averaged(self) -> None:
        """Multiple timesteps are averaged defensively."""
        model = PrecomputedEmbedding(modality=Modality.GSE.name)
        gse = torch.randn(B, H, W, 3, C)
        out = model(_sample_with_gse(gse), spatial_pool=True)
        torch.testing.assert_close(out, gse.mean(dim=3))

    def test_has_baseline_interface(self) -> None:
        """The module exposes the attributes the eval harness expects."""
        model = PrecomputedEmbedding(modality=Modality.GSE.name)
        assert model.patch_size == 1
        assert model.supported_modalities == [Modality.GSE.name]
        assert model.required_modalities == [Modality.GSE.name]
        assert not model.requires_timeseries
        # The eval harness resolves device via model parameters.
        assert next(model.parameters()).device == model.device


class TestPrecomputedEmbeddingEvalWrapper:
    """Tests for PrecomputedEmbeddingEvalWrapper via get_eval_wrapper."""

    def _wrapper(
        self, task_type: TaskType, use_center_token: bool = False
    ) -> PrecomputedEmbeddingEvalWrapper:
        wrapper = get_eval_wrapper(
            PrecomputedEmbedding(modality=Modality.GSE.name),
            task_type=task_type,
            patch_size=1,
            pooling_type=PoolingType.MEAN,
            use_center_token=use_center_token,
        )
        assert isinstance(wrapper, PrecomputedEmbeddingEvalWrapper)
        return wrapper

    def test_classification_pools_space(self) -> None:
        """Classification tasks pool space to a single embedding."""
        wrapper = self._wrapper(TaskType.CLASSIFICATION)
        gse = torch.randn(B, H, W, 1, C)
        labels = torch.zeros(B, dtype=torch.long)
        embeddings, out_labels = wrapper(_sample_with_gse(gse), labels)
        assert embeddings.shape == (B, C)
        assert torch.equal(out_labels, labels)

    def test_classification_center_token(self) -> None:
        """use_center_token returns the center pixel embedding."""
        wrapper = self._wrapper(TaskType.CLASSIFICATION, use_center_token=True)
        gse = torch.randn(B, H, W, 1, C)
        labels = torch.zeros(B, dtype=torch.long)
        embeddings, _ = wrapper(_sample_with_gse(gse), labels)
        assert embeddings.shape == (B, C)
        torch.testing.assert_close(embeddings, gse[:, H // 2, W // 2, 0, :])

    def test_segmentation_keeps_spatial_grid(self) -> None:
        """Segmentation tasks keep the spatial grid for per-pixel probes."""
        wrapper = self._wrapper(TaskType.SEGMENTATION)
        gse = torch.randn(B, H, W, 1, C)
        labels = torch.zeros(B, H, W, dtype=torch.long)
        embeddings, _ = wrapper(_sample_with_gse(gse), labels)
        assert embeddings.shape == (B, H, W, C)

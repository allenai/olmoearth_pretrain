"""Unit tests for segmentation metrics."""

import pytest
import torch

from olmoearth_pretrain.evals.metrics import (
    EvalResult,
    _build_confusion_matrix,
    segmentation_metrics,
)


class TestEvalResult:
    """Tests for EvalResult dataclass."""

    def test_from_classification(self) -> None:
        """Test creating EvalResult from classification accuracy."""
        result = EvalResult.from_classification(0.95)
        assert result.primary == 0.95
        assert result.metrics == {"accuracy": 0.95}

    def test_from_classification_with_f1(self) -> None:
        """Test creating EvalResult from multilabel classification with accuracy and F1."""
        result = EvalResult.from_classification(0.80, f1=0.85)
        assert result.primary == 0.80
        assert result.metrics == {"accuracy": 0.80, "f1": 0.85}
    def test_from_segmentation(self) -> None:
        """Test creating EvalResult from segmentation metrics."""
        result = EvalResult.from_segmentation(
            miou=0.8, overall_acc=0.9, macro_acc=0.85, macro_f1=0.82
        )
        assert result.primary == 0.8
        assert result.metrics == {
            "miou": 0.8,
            "overall_acc": 0.9,
            "macro_acc": 0.85,
            "macro_f1": 0.82,
        }

    def test_metrics_contains_primary(self) -> None:
        """Primary metric should be in the metrics dict."""
        result = EvalResult.from_classification(0.95)
        assert "accuracy" in result.metrics
        assert result.metrics["accuracy"] == result.primary


class TestBuildConfusionMatrix:
    """Tests for _build_confusion_matrix."""

    def test_perfect_prediction(self) -> None:
        """Perfect predictions should have diagonal confusion matrix."""
        preds = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
        labels = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
        confusion = _build_confusion_matrix(preds, labels, num_classes=2)

        expected = torch.tensor([[2, 0], [0, 2]])
        assert torch.equal(confusion, expected)

    def test_all_wrong(self) -> None:
        """All wrong predictions should have off-diagonal entries."""
        preds = torch.tensor([[1, 0], [1, 0]], dtype=torch.long)
        labels = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
        confusion = _build_confusion_matrix(preds, labels, num_classes=2)

        expected = torch.tensor([[0, 2], [2, 0]])
        assert torch.equal(confusion, expected)

    def test_ignore_label(self) -> None:
        """Pixels with ignore_label should be excluded."""
        preds = torch.tensor([[0, 1, 0], [0, 1, 0]], dtype=torch.long)
        labels = torch.tensor([[0, 1, -1], [0, -1, -1]], dtype=torch.long)
        confusion = _build_confusion_matrix(
            preds, labels, num_classes=2, ignore_label=-1
        )

        # Only 3 valid pixels: (0,0), (0,1), (1,0)
        expected = torch.tensor([[2, 0], [0, 1]])
        assert torch.equal(confusion, expected)

    def test_multiclass(self) -> None:
        """Test with 3 classes."""
        preds = torch.tensor([[0, 1, 2]], dtype=torch.long)
        labels = torch.tensor([[0, 1, 2]], dtype=torch.long)
        confusion = _build_confusion_matrix(preds, labels, num_classes=3)

        expected = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert torch.equal(confusion, expected)

    def test_batch_dimension(self) -> None:
        """Test with batched input (B, H, W)."""
        preds = torch.tensor(
            [
                [[0, 1], [0, 1]],
                [[0, 1], [0, 1]],
            ],
            dtype=torch.long,
        )
        labels = torch.tensor(
            [
                [[0, 1], [0, 1]],
                [[0, 1], [0, 1]],
            ],
            dtype=torch.long,
        )
        confusion = _build_confusion_matrix(preds, labels, num_classes=2)

        # 8 pixels total, 4 of each class
        expected = torch.tensor([[4, 0], [0, 4]])
        assert torch.equal(confusion, expected)

    def test_float_dtype_raises_error(self) -> None:
        """Float predictions should raise TypeError."""
        preds = torch.tensor([[0.0, 1.0], [0.0, 1.0]])  # float32
        labels = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
        with pytest.raises(TypeError, match="predictions must be integer"):
            _build_confusion_matrix(preds, labels, num_classes=2)

    def test_float_labels_raises_error(self) -> None:
        """Float labels should raise TypeError."""
        preds = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
        labels = torch.tensor([[0.0, 1.0], [0.0, 1.0]])  # float32
        with pytest.raises(TypeError, match="labels must be integer"):
            _build_confusion_matrix(preds, labels, num_classes=2)


class TestSegmentationMetrics:
    """Tests for segmentation_metrics function."""

    def test_perfect_prediction(self) -> None:
        """Perfect predictions should give all metrics = 1.0."""
        preds = torch.tensor([[[0, 1], [0, 1]]], dtype=torch.long)
        labels = torch.tensor([[[0, 1], [0, 1]]], dtype=torch.long)
        result = segmentation_metrics(preds, labels, num_classes=2)

        assert isinstance(result, EvalResult)
        assert result.primary == pytest.approx(1.0)  # miou
        assert result.metrics["miou"] == pytest.approx(1.0)
        assert result.metrics["overall_acc"] == pytest.approx(1.0)
        assert result.metrics["macro_acc"] == pytest.approx(1.0)
        assert result.metrics["macro_f1"] == pytest.approx(1.0)

    def test_all_wrong(self) -> None:
        """All wrong predictions should give miou = 0."""
        preds = torch.tensor([[[1, 0], [1, 0]]], dtype=torch.long)
        labels = torch.tensor([[[0, 1], [0, 1]]], dtype=torch.long)
        result = segmentation_metrics(preds, labels, num_classes=2)

        assert result.metrics["miou"] == pytest.approx(0.0, abs=1e-6)
        assert result.metrics["overall_acc"] == pytest.approx(0.0, abs=1e-6)
        assert result.metrics["macro_acc"] == pytest.approx(0.0, abs=1e-6)

    def test_half_correct(self) -> None:
        """50% accuracy case."""
        # 2 correct, 2 wrong
        preds = torch.tensor([[[0, 0], [1, 1]]], dtype=torch.long)
        labels = torch.tensor([[[0, 1], [0, 1]]], dtype=torch.long)
        result = segmentation_metrics(preds, labels, num_classes=2)

        assert result.metrics["overall_acc"] == pytest.approx(0.5)
        # Each class has 50% recall
        assert result.metrics["macro_acc"] == pytest.approx(0.5)

    def test_ignore_label(self) -> None:
        """Ignored pixels should not affect metrics."""
        preds = torch.tensor([[[0, 1, 0], [0, 1, 0]]], dtype=torch.long)
        labels = torch.tensor([[[0, 1, -1], [0, -1, -1]]], dtype=torch.long)
        result = segmentation_metrics(preds, labels, num_classes=2, ignore_label=-1)

        # 3 valid pixels, all correct
        assert result.metrics["overall_acc"] == pytest.approx(1.0)
        assert result.metrics["miou"] == pytest.approx(1.0)

    def test_empty_class(self) -> None:
        """Classes with no samples should not affect mean metrics."""
        # Only class 0 present in ground truth
        preds = torch.tensor([[[0, 0], [0, 0]]], dtype=torch.long)
        labels = torch.tensor([[[0, 0], [0, 0]]], dtype=torch.long)
        result = segmentation_metrics(preds, labels, num_classes=3)

        # Class 0 has IoU=1, classes 1,2 have no samples (excluded from mean)
        assert result.metrics["miou"] == pytest.approx(1.0)
        assert result.metrics["overall_acc"] == pytest.approx(1.0)
        assert result.metrics["macro_acc"] == pytest.approx(1.0)

    def test_batch_dimension(self) -> None:
        """Test with multiple samples in batch."""
        preds = torch.tensor(
            [
                [[0, 1], [0, 1]],
                [[0, 1], [0, 1]],
            ],
            dtype=torch.long,
        )
        labels = torch.tensor(
            [
                [[0, 1], [0, 1]],
                [[0, 1], [0, 1]],
            ],
            dtype=torch.long,
        )
        result = segmentation_metrics(preds, labels, num_classes=2)

        assert result.metrics["miou"] == pytest.approx(1.0)
        assert result.metrics["overall_acc"] == pytest.approx(1.0)

    def test_all_ignored(self) -> None:
        """All pixels ignored should return zeros (via epsilon protection)."""
        preds = torch.tensor([[[0, 1]]], dtype=torch.long)
        labels = torch.tensor([[[-1, -1]]], dtype=torch.long)
        result = segmentation_metrics(preds, labels, num_classes=2, ignore_label=-1)

        # No valid pixels - metrics should be ~0 due to empty confusion matrix
        assert result.metrics["overall_acc"] == pytest.approx(0.0, abs=1e-6)

    def test_returns_eval_result(self) -> None:
        """Verify return type is EvalResult with expected keys."""
        preds = torch.tensor([[[0, 1]]], dtype=torch.long)
        labels = torch.tensor([[[0, 1]]], dtype=torch.long)
        result = segmentation_metrics(preds, labels, num_classes=2)

        assert isinstance(result, EvalResult)
        expected_keys = {"miou", "overall_acc", "macro_acc", "macro_f1"}
        assert set(result.metrics.keys()) == expected_keys

        for key in expected_keys:
            assert isinstance(result.metrics[key], float)
            assert 0.0 <= result.metrics[key] <= 1.0

"""Unit tests for segmentation metrics."""

import pytest
import torch

from olmoearth_pretrain.evals.metrics import (
    _build_confusion_matrix,
    mean_iou,
    segmentation_metrics,
)


class TestBuildConfusionMatrix:
    """Tests for _build_confusion_matrix."""

    def test_perfect_prediction(self) -> None:
        """Perfect predictions should have diagonal confusion matrix."""
        preds = torch.tensor([[0, 1], [0, 1]])
        labels = torch.tensor([[0, 1], [0, 1]])
        confusion = _build_confusion_matrix(preds, labels, num_classes=2)

        expected = torch.tensor([[2, 0], [0, 2]])
        assert torch.equal(confusion, expected)

    def test_all_wrong(self) -> None:
        """All wrong predictions should have off-diagonal entries."""
        preds = torch.tensor([[1, 0], [1, 0]])
        labels = torch.tensor([[0, 1], [0, 1]])
        confusion = _build_confusion_matrix(preds, labels, num_classes=2)

        expected = torch.tensor([[0, 2], [2, 0]])
        assert torch.equal(confusion, expected)

    def test_ignore_label(self) -> None:
        """Pixels with ignore_label should be excluded."""
        preds = torch.tensor([[0, 1, 0], [0, 1, 0]])
        labels = torch.tensor([[0, 1, -1], [0, -1, -1]])  # -1 is ignored
        confusion = _build_confusion_matrix(
            preds, labels, num_classes=2, ignore_label=-1
        )

        # Only 3 valid pixels: (0,0), (0,1), (1,0)
        expected = torch.tensor([[2, 0], [0, 1]])
        assert torch.equal(confusion, expected)

    def test_multiclass(self) -> None:
        """Test with 3 classes."""
        preds = torch.tensor([[0, 1, 2]])
        labels = torch.tensor([[0, 1, 2]])
        confusion = _build_confusion_matrix(preds, labels, num_classes=3)

        expected = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert torch.equal(confusion, expected)


class TestSegmentationMetrics:
    """Tests for segmentation_metrics function."""

    def test_perfect_prediction(self) -> None:
        """Perfect predictions should give all metrics = 1.0."""
        preds = torch.tensor([[[0, 1], [0, 1]]])
        labels = torch.tensor([[[0, 1], [0, 1]]])
        result = segmentation_metrics(preds, labels, num_classes=2)

        assert result["miou"] == pytest.approx(1.0)
        assert result["overall_acc"] == pytest.approx(1.0)
        assert result["macro_acc"] == pytest.approx(1.0)
        assert result["micro_f1"] == pytest.approx(1.0)
        assert result["macro_f1"] == pytest.approx(1.0)

    def test_all_wrong(self) -> None:
        """All wrong predictions should give miou = 0."""
        preds = torch.tensor([[[1, 0], [1, 0]]])
        labels = torch.tensor([[[0, 1], [0, 1]]])
        result = segmentation_metrics(preds, labels, num_classes=2)

        assert result["miou"] == pytest.approx(0.0, abs=1e-6)
        assert result["overall_acc"] == pytest.approx(0.0, abs=1e-6)
        assert result["macro_acc"] == pytest.approx(0.0, abs=1e-6)

    def test_half_correct(self) -> None:
        """50% accuracy case."""
        # 2 correct, 2 wrong
        preds = torch.tensor([[[0, 0], [1, 1]]])
        labels = torch.tensor([[[0, 1], [0, 1]]])
        result = segmentation_metrics(preds, labels, num_classes=2)

        assert result["overall_acc"] == pytest.approx(0.5)
        # Each class has 50% recall
        assert result["macro_acc"] == pytest.approx(0.5)

    def test_ignore_label(self) -> None:
        """Ignored pixels should not affect metrics."""
        preds = torch.tensor([[[0, 1, 0], [0, 1, 0]]])
        labels = torch.tensor([[[0, 1, -1], [0, -1, -1]]])
        result = segmentation_metrics(preds, labels, num_classes=2, ignore_label=-1)

        # 3 valid pixels, all correct
        assert result["overall_acc"] == pytest.approx(1.0)
        assert result["miou"] == pytest.approx(1.0)

    def test_empty_class(self) -> None:
        """Classes with no samples should not affect mean metrics."""
        # Only class 0 present in ground truth
        preds = torch.tensor([[[0, 0], [0, 0]]])
        labels = torch.tensor([[[0, 0], [0, 0]]])
        result = segmentation_metrics(preds, labels, num_classes=3)

        # Class 0 has IoU=1, classes 1,2 have no samples (excluded from mean)
        assert result["miou"] == pytest.approx(1.0)
        assert result["overall_acc"] == pytest.approx(1.0)
        assert result["macro_acc"] == pytest.approx(1.0)

    def test_batch_dimension(self) -> None:
        """Test with multiple samples in batch."""
        preds = torch.tensor(
            [
                [[0, 1], [0, 1]],
                [[0, 1], [0, 1]],
            ]
        )
        labels = torch.tensor(
            [
                [[0, 1], [0, 1]],
                [[0, 1], [0, 1]],
            ]
        )
        result = segmentation_metrics(preds, labels, num_classes=2)

        assert result["miou"] == pytest.approx(1.0)
        assert result["overall_acc"] == pytest.approx(1.0)

    def test_all_ignored(self) -> None:
        """All pixels ignored should return zeros (via epsilon protection)."""
        preds = torch.tensor([[[0, 1]]])
        labels = torch.tensor([[[-1, -1]]])
        result = segmentation_metrics(preds, labels, num_classes=2, ignore_label=-1)

        # No valid pixels - metrics should be ~0 due to empty confusion matrix
        assert result["overall_acc"] == pytest.approx(0.0, abs=1e-6)

    def test_returns_dict_with_expected_keys(self) -> None:
        """Verify return type and keys."""
        preds = torch.tensor([[[0, 1]]])
        labels = torch.tensor([[[0, 1]]])
        result = segmentation_metrics(preds, labels, num_classes=2)

        assert isinstance(result, dict)
        expected_keys = {"miou", "overall_acc", "macro_acc", "micro_f1", "macro_f1"}
        assert set(result.keys()) == expected_keys

        for key in expected_keys:
            assert isinstance(result[key], float)
            assert 0.0 <= result[key] <= 1.0


class TestMeanIou:
    """Tests for mean_iou function (legacy interface)."""

    def test_perfect_prediction(self) -> None:
        """Perfect predictions should give mIoU = 1.0."""
        preds = torch.tensor([[[0, 1], [0, 1]]])
        labels = torch.tensor([[[0, 1], [0, 1]]])
        result = mean_iou(preds, labels, num_classes=2)

        assert result == pytest.approx(1.0)

    def test_all_wrong(self) -> None:
        """All wrong predictions should give mIoU = 0."""
        preds = torch.tensor([[[1, 0], [1, 0]]])
        labels = torch.tensor([[[0, 1], [0, 1]]])
        result = mean_iou(preds, labels, num_classes=2)

        assert result == pytest.approx(0.0, abs=1e-6)

    def test_ignore_label(self) -> None:
        """Ignored pixels should not affect mIoU."""
        preds = torch.tensor([[[0, 1, 0]]])
        labels = torch.tensor([[[0, 1, -1]]])
        result = mean_iou(preds, labels, num_classes=2, ignore_label=-1)

        assert result == pytest.approx(1.0)

    def test_returns_float(self) -> None:
        """Verify return type is float."""
        preds = torch.tensor([[[0, 1]]])
        labels = torch.tensor([[[0, 1]]])
        result = mean_iou(preds, labels, num_classes=2)

        assert isinstance(result, float)


class TestMetricsConsistency:
    """Tests verifying consistency between different metric functions."""

    def test_miou_matches_segmentation_metrics(self) -> None:
        """mean_iou should match segmentation_metrics miou."""
        preds = torch.randint(0, 3, (4, 16, 16))
        labels = torch.randint(0, 3, (4, 16, 16))

        miou_standalone = mean_iou(preds, labels, num_classes=3)
        metrics = segmentation_metrics(preds, labels, num_classes=3)

        assert miou_standalone == pytest.approx(metrics["miou"], abs=1e-6)

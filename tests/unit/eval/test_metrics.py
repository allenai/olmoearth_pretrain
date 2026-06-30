"""Unit tests for segmentation metrics."""

import math

import numpy as np
import pytest
import torch

from olmoearth_pretrain.evals.metrics import (
    EvalMetric,
    EvalResult,
    _build_confusion_matrix,
    _macro_ovr_auroc,
    classification_metrics,
    regression_metrics,
    segmentation_metrics,
)


def _seg_onehot_scores(preds: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Build one-hot per-pixel scores (N, C, H, W) from predictions (N, H, W)."""
    return torch.nn.functional.one_hot(preds, num_classes).permute(0, 3, 1, 2).float()


class TestEvalResult:
    """Tests for EvalResult dataclass."""

    def test_from_classification(self) -> None:
        """Test creating EvalResult from classification accuracy."""
        result = EvalResult.from_classification(0.95)
        assert result.primary == 0.95
        assert result.primary_metric == EvalMetric.ACCURACY
        assert result.metrics == {"accuracy": 0.95}

    def test_from_classification_with_f1(self) -> None:
        """Test creating EvalResult from multilabel classification with accuracy and F1."""
        result = EvalResult.from_classification(0.80, f1=0.85, is_multilabel=True)
        assert result.primary == 0.85
        assert result.primary_metric == EvalMetric.F1
        assert result.metrics == {"accuracy": 0.80, "f1": 0.85}

    def test_from_classification_multilabel_requires_f1(self) -> None:
        """Multilabel classification requires f1 to set primary metric."""
        with pytest.raises(ValueError, match="not found in computed metrics"):
            EvalResult.from_classification(0.80, is_multilabel=True)

    def test_from_segmentation(self) -> None:
        """Test creating EvalResult from segmentation metrics."""
        result = EvalResult.from_segmentation(
            miou=0.8, overall_acc=0.9, macro_acc=0.85, macro_f1=0.82, micro_f1=0.87
        )
        assert result.primary == 0.8
        assert result.primary_metric == EvalMetric.MIOU
        assert result.metrics == {
            "miou": 0.8,
            "overall_acc": 0.9,
            "macro_acc": 0.85,
            "macro_f1": 0.82,
            "micro_f1": 0.87,
        }

    def test_from_regression(self) -> None:
        """Test creating EvalResult from regression metrics."""
        result = EvalResult.from_regression(mae=1.0, rmse=2.0, r2=0.5)
        assert result.primary == -2.0
        assert result.primary_metric == EvalMetric.NEG_RMSE
        assert result.metrics == {"mae": 1.0, "rmse": 2.0, "neg_rmse": -2.0, "r2": 0.5}

    def test_metrics_contains_primary(self) -> None:
        """Primary metric should be in the metrics dict."""
        result = EvalResult.from_classification(0.95)
        assert "accuracy" in result.metrics
        assert result.metrics["accuracy"] == result.primary

    def test_with_primary_metric(self) -> None:
        """Test overriding the primary metric."""
        result = EvalResult.from_classification(0.80, f1=0.85)
        overridden = result.with_primary_metric(EvalMetric.F1)
        assert overridden.primary == 0.85
        assert overridden.primary_metric == EvalMetric.F1
        assert overridden.metrics == result.metrics

    def test_with_primary_metric_invalid(self) -> None:
        """Test overriding with a metric not in the dict."""
        result = EvalResult.from_classification(0.80)
        with pytest.raises(ValueError, match="not found in metrics"):
            result.with_primary_metric(EvalMetric.MIOU)

    def test_from_classification_with_per_class_f1(self) -> None:
        """Test creating EvalResult with per-class F1 scores."""
        result = EvalResult.from_classification(0.90, per_class_f1=[0.85, 0.92, 0.88])
        assert result.metrics["f1_class_0"] == 0.85
        assert result.metrics["f1_class_1"] == 0.92
        assert result.metrics["f1_class_2"] == 0.88

    def test_with_primary_metric_class_f1(self) -> None:
        """Test overriding primary to a specific class F1."""
        result = EvalResult.from_classification(0.90, per_class_f1=[0.85, 0.92, 0.88])
        overridden = result.with_primary_metric(EvalMetric.CLASS_F1, class_idx=1)
        assert overridden.primary == 0.92
        assert overridden.primary_metric == EvalMetric.CLASS_F1

    def test_with_primary_metric_class_f1_requires_class_idx(self) -> None:
        """CLASS_F1 without class_idx should raise."""
        result = EvalResult.from_classification(0.90, per_class_f1=[0.85, 0.92])
        with pytest.raises(ValueError, match="class_idx is required"):
            result.with_primary_metric(EvalMetric.CLASS_F1)


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
        scores = _seg_onehot_scores(preds, 2)
        result = segmentation_metrics(preds, labels, num_classes=2, scores=scores)

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
        scores = _seg_onehot_scores(preds, 2)
        result = segmentation_metrics(preds, labels, num_classes=2, scores=scores)

        assert result.metrics["miou"] == pytest.approx(0.0, abs=1e-6)
        assert result.metrics["overall_acc"] == pytest.approx(0.0, abs=1e-6)
        assert result.metrics["macro_acc"] == pytest.approx(0.0, abs=1e-6)

    def test_half_correct(self) -> None:
        """50% accuracy case."""
        # 2 correct, 2 wrong
        preds = torch.tensor([[[0, 0], [1, 1]]], dtype=torch.long)
        labels = torch.tensor([[[0, 1], [0, 1]]], dtype=torch.long)
        scores = _seg_onehot_scores(preds, 2)
        result = segmentation_metrics(preds, labels, num_classes=2, scores=scores)

        assert result.metrics["overall_acc"] == pytest.approx(0.5)
        # Each class has 50% recall
        assert result.metrics["macro_acc"] == pytest.approx(0.5)

    def test_ignore_label(self) -> None:
        """Ignored pixels should not affect metrics."""
        preds = torch.tensor([[[0, 1, 0], [0, 1, 0]]], dtype=torch.long)
        labels = torch.tensor([[[0, 1, -1], [0, -1, -1]]], dtype=torch.long)
        scores = _seg_onehot_scores(preds, 2)
        result = segmentation_metrics(
            preds, labels, num_classes=2, scores=scores, ignore_label=-1
        )

        # 3 valid pixels, all correct
        assert result.metrics["overall_acc"] == pytest.approx(1.0)
        assert result.metrics["miou"] == pytest.approx(1.0)

    def test_empty_class(self) -> None:
        """Classes with no samples should not affect mean metrics."""
        # Only class 0 present in ground truth
        preds = torch.tensor([[[0, 0], [0, 0]]], dtype=torch.long)
        labels = torch.tensor([[[0, 0], [0, 0]]], dtype=torch.long)
        scores = _seg_onehot_scores(preds, 3)
        result = segmentation_metrics(preds, labels, num_classes=3, scores=scores)

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
        scores = _seg_onehot_scores(preds, 2)
        result = segmentation_metrics(preds, labels, num_classes=2, scores=scores)

        assert result.metrics["miou"] == pytest.approx(1.0)
        assert result.metrics["overall_acc"] == pytest.approx(1.0)

    def test_returns_eval_result(self) -> None:
        """Verify return type is EvalResult with expected keys."""
        preds = torch.tensor([[[0, 1]]], dtype=torch.long)
        labels = torch.tensor([[[0, 1]]], dtype=torch.long)
        scores = _seg_onehot_scores(preds, 2)
        result = segmentation_metrics(preds, labels, num_classes=2, scores=scores)

        assert isinstance(result, EvalResult)
        num_classes = 2
        expected_keys = {
            "miou",
            "overall_acc",
            "macro_acc",
            "macro_f1",
            "micro_f1",
            "auroc",
        } | {f"f1_class_{i}" for i in range(num_classes)}
        assert set(result.metrics.keys()) == expected_keys

        for key in expected_keys:
            assert isinstance(result.metrics[key], float)
            assert 0.0 <= result.metrics[key] <= 1.0


class TestClassificationMetrics:
    """Tests for classification_metrics function."""

    def test_single_label_classification(self) -> None:
        """Single-label metrics should include accuracy and macro/per-class F1."""
        preds = torch.tensor([0, 1, 1, 0], dtype=torch.long)
        labels = torch.tensor([0, 1, 0, 0], dtype=torch.long)
        scores = torch.nn.functional.one_hot(preds, 2).float()
        result = classification_metrics(preds, labels, scores, is_multilabel=False)

        assert result.primary_metric == EvalMetric.ACCURACY
        assert "accuracy" in result.metrics
        assert "macro_f1" in result.metrics
        assert "f1_class_0" in result.metrics
        assert "f1_class_1" in result.metrics
        assert "auroc" in result.metrics

    def test_multilabel_classification(self) -> None:
        """Multilabel metrics should include exact-match accuracy and micro F1."""
        preds = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.int)
        labels = torch.tensor([[1, 0, 1], [1, 0, 0]], dtype=torch.int)
        scores = preds.float()
        result = classification_metrics(preds, labels, scores, is_multilabel=True)

        assert result.primary_metric == EvalMetric.F1
        assert "accuracy" in result.metrics
        assert "f1" in result.metrics
        assert "macro_f1" in result.metrics
        assert "f1_class_0" in result.metrics
        assert "auroc" in result.metrics


class TestRegressionMetrics:
    """Tests for regression metrics."""

    def test_perfect_prediction(self) -> None:
        """Perfect predictions have zero error and R2 of 1."""
        labels = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = regression_metrics(labels, labels)

        assert result.primary == pytest.approx(0.0)
        assert result.metrics["mae"] == pytest.approx(0.0)
        assert result.metrics["rmse"] == pytest.approx(0.0)
        assert result.metrics["neg_rmse"] == pytest.approx(0.0)
        assert result.metrics["r2"] == pytest.approx(1.0)

    def test_nonzero_error(self) -> None:
        """Regression metrics reflect prediction errors."""
        preds = torch.tensor([1.0, 3.0])
        labels = torch.tensor([1.0, 1.0])
        result = regression_metrics(preds, labels)

        assert result.metrics["mae"] == pytest.approx(1.0)
        assert result.metrics["rmse"] == pytest.approx(2**0.5)


class TestMacroOvrAuroc:
    """Tests for the _macro_ovr_auroc helper."""

    def test_perfect_separation(self) -> None:
        """Scores perfectly ranking each class give AUROC = 1.0."""
        labels = np.array([0, 0, 1, 1, 2, 2])
        scores = np.array(
            [
                [0.9, 0.05, 0.05],
                [0.8, 0.1, 0.1],
                [0.1, 0.85, 0.05],
                [0.1, 0.8, 0.1],
                [0.05, 0.1, 0.85],
                [0.1, 0.1, 0.8],
            ]
        )
        assert _macro_ovr_auroc(scores, labels, num_classes=3) == pytest.approx(1.0)

    def test_random_scores_near_half(self) -> None:
        """Uninformative (constant) scores give AUROC ~0.5."""
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 3, size=2000)
        scores = np.full((2000, 3), 1.0 / 3)
        # Constant scores -> every threshold ties -> AUROC exactly 0.5.
        assert _macro_ovr_auroc(scores, labels, num_classes=3) == pytest.approx(0.5)

    def test_skips_degenerate_class(self) -> None:
        """A class absent from labels is skipped, not errored on."""
        # Class 2 never appears; sklearn would raise if not skipped.
        labels = np.array([0, 0, 1, 1])
        scores = np.array(
            [
                [0.9, 0.05, 0.05],
                [0.8, 0.1, 0.1],
                [0.1, 0.85, 0.05],
                [0.1, 0.8, 0.1],
            ]
        )
        assert _macro_ovr_auroc(scores, labels, num_classes=3) == pytest.approx(1.0)

    def test_all_degenerate_returns_nan(self) -> None:
        """If every class is all-positive/all-negative, return NaN."""
        labels = np.array([0, 0, 0])
        scores = np.array([[1.0], [0.5], [0.2]])
        assert math.isnan(_macro_ovr_auroc(scores, labels, num_classes=1))


class TestAurocMetrics:
    """Tests for AUROC wiring through classification and segmentation metrics."""

    def test_single_label_classification_auroc(self) -> None:
        """Single-label classification includes AUROC when scores are passed."""
        preds = torch.tensor([0, 1, 2, 0])
        labels = torch.tensor([0, 1, 2, 0])
        scores = torch.tensor(
            [
                [0.9, 0.05, 0.05],
                [0.05, 0.9, 0.05],
                [0.05, 0.05, 0.9],
                [0.8, 0.1, 0.1],
            ]
        )
        result = classification_metrics(
            preds, labels, scores=scores, primary_metric=EvalMetric.AUROC
        )
        assert result.primary_metric == EvalMetric.AUROC
        assert result.metrics["auroc"] == pytest.approx(1.0)

    def test_multilabel_classification_auroc(self) -> None:
        """Multilabel classification includes AUROC when scores are passed."""
        preds = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.int)
        labels = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.int)
        scores = torch.tensor([[0.9, 0.1, 0.8], [0.2, 0.85, 0.1]])
        result = classification_metrics(
            preds, labels, is_multilabel=True, scores=scores
        )
        assert "auroc" in result.metrics
        assert result.metrics["auroc"] == pytest.approx(1.0)

    def test_segmentation_auroc(self) -> None:
        """Segmentation includes AUROC computed over valid pixels."""
        # (N=1, H=2, W=2) labels, scores (N, C=2, H, W).
        labels = torch.tensor([[[0, 1], [0, 1]]], dtype=torch.long)
        preds = torch.tensor([[[0, 1], [0, 1]]], dtype=torch.long)
        scores = torch.tensor(
            [
                [
                    [[0.9, 0.1], [0.8, 0.2]],
                    [[0.1, 0.9], [0.2, 0.8]],
                ]
            ]
        )
        result = segmentation_metrics(
            preds, labels, num_classes=2, scores=scores, primary_metric=EvalMetric.AUROC
        )
        assert result.primary_metric == EvalMetric.AUROC
        assert result.metrics["auroc"] == pytest.approx(1.0)

    def test_segmentation_auroc_respects_ignore_label(self) -> None:
        """Ignored pixels are excluded from the AUROC computation."""
        # The ignored pixel has a misleading score; it must not affect AUROC.
        labels = torch.tensor([[[0, 1], [0, -1]]], dtype=torch.long)
        preds = torch.tensor([[[0, 1], [0, 0]]], dtype=torch.long)
        scores = torch.tensor(
            [
                [
                    [[0.9, 0.1], [0.8, 0.99]],
                    [[0.1, 0.9], [0.2, 0.01]],
                ]
            ]
        )
        result = segmentation_metrics(
            preds,
            labels,
            num_classes=2,
            ignore_label=-1,
            scores=scores,
            primary_metric=EvalMetric.AUROC,
        )
        assert result.metrics["auroc"] == pytest.approx(1.0)

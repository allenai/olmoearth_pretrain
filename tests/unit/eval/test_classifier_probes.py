"""Tests for the off-the-shelf classifier probes."""

from typing import Any

import numpy as np
import pytest
import torch

from olmoearth_pretrain.evals.classifier_probes import (
    CLASSIFIERS,
    ClassifierProbeConfig,
    classifier_scores,
    classifier_task_name,
    fit_classifier,
    predict_scores,
    run_classifier_probes,
)
from olmoearth_pretrain.evals.datasets.configs import EvalDatasetConfig
from olmoearth_pretrain.evals.metrics import SEGMENTATION_IGNORE_LABEL
from olmoearth_pretrain.evals.task_types import TaskType

NUM_CLASSES = 4
DIM = 8


def _has_xgboost() -> bool:
    try:
        import xgboost  # noqa: F401
    except Exception:  # missing wheel or missing OpenMP runtime
        return False
    return True


# Every predictor, with xgboost skipped where its runtime is unavailable.
PREDICTORS = [
    pytest.param(
        name,
        marks=pytest.mark.skipif(
            name == "xgb" and not _has_xgboost(), reason="xgboost unavailable"
        ),
    )
    for name in CLASSIFIERS
]


def _config(task_type: TaskType, num_classes: int = NUM_CLASSES) -> EvalDatasetConfig:
    return EvalDatasetConfig(
        task_type=task_type,
        imputes=[],
        num_classes=num_classes,
        is_multilabel=False,
        supported_modalities=[],
        height_width=4 if task_type == TaskType.SEGMENTATION else None,
    )


def _separable(labels: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Class-conditional Gaussians around well-separated means."""
    generator = torch.Generator().manual_seed(seed)
    means = torch.randn(NUM_CLASSES, DIM, generator=generator) * 4.0
    noise = torch.randn(*labels.shape, DIM, generator=generator)
    return means[labels] + noise


def _fast_config(names: tuple[str, ...], **kwargs: Any) -> ClassifierProbeConfig:
    return ClassifierProbeConfig(
        names=names,
        n_jobs=1,
        rf_n_estimators=20,
        xgb_n_estimators=20,
        mlp_hidden=16,
        mlp_max_iter=100,
        **kwargs,
    )


class TestScores:
    """Fitting and scoring on flat rows."""

    @pytest.mark.parametrize("name", PREDICTORS)
    def test_separable_classes_are_recovered(self, name: str) -> None:
        """Every predictor recovers well-separated classes on held-out rows."""
        labels = torch.arange(200) % NUM_CLASSES
        features = _separable(labels).numpy()
        scores = classifier_scores(
            name,
            _fast_config((name,)),
            features[:120],
            labels[:120].numpy(),
            features[120:],
            NUM_CLASSES,
        )
        assert scores.shape == (80, NUM_CLASSES)
        assert scores.dtype == torch.float32
        accuracy = (scores.argmax(dim=1) == labels[120:]).float().mean()
        assert accuracy > 0.9

    @pytest.mark.parametrize("name", PREDICTORS)
    def test_absent_classes_get_zero_columns(self, name: str) -> None:
        """A draw missing a class still scores against every task class."""
        labels = torch.tensor([0, 0, 0, 2, 2, 2, 3, 3, 3] * 5)  # class 1 absent
        features = _separable(labels).numpy()
        fitted = fit_classifier(name, _fast_config((name,)), features, labels.numpy())
        assert fitted.classes.tolist() == [0, 2, 3]
        scores = predict_scores(fitted, features, NUM_CLASSES, chunk_rows=7)
        assert scores.shape == (labels.numel(), NUM_CLASSES)
        assert np.all(scores[:, 1] == 0.0)
        np.testing.assert_allclose(scores.sum(axis=1), 1.0, atol=1e-4)
        assert (scores.argmax(axis=1) == labels.numpy()).mean() > 0.9

    def test_single_class_draw_is_one_hot(self) -> None:
        """A one-class training set scores that class with probability one."""
        labels = np.full(10, 2)
        features = np.random.default_rng(0).normal(size=(10, DIM)).astype(np.float32)
        fitted = fit_classifier("rf", _fast_config(("rf",)), features, labels)
        scores = predict_scores(fitted, features, NUM_CLASSES, chunk_rows=100)
        assert np.all(scores[:, 2] == 1.0)
        assert scores.sum() == 10.0

    def test_unknown_name_is_rejected(self) -> None:
        """An unknown predictor name fails at validation, not mid-eval."""
        with pytest.raises(ValueError, match="Unknown classifier"):
            ClassifierProbeConfig(names=("svm",)).validate()


class TestRunClassifierProbes:
    """The train -> val / test path the evaluator callback calls."""

    def test_classification_reports_val_and_test(self) -> None:
        """Flat tasks report a val and a test EvalResult per predictor."""
        labels = torch.arange(240) % NUM_CLASSES
        features = _separable(labels)
        results = run_classifier_probes(
            config=_config(TaskType.CLASSIFICATION),
            train_embeddings=features[:120],
            train_labels=labels[:120],
            val_embeddings=features[120:180],
            val_labels=labels[120:180],
            test_embeddings=features[180:],
            test_labels=labels[180:],
            probe_config=_fast_config(("rf", "logreg")),
        )
        assert set(results) == {"rf", "logreg"}
        for val_result, test_result in results.values():
            assert test_result is not None
            assert val_result.primary > 0.9
            assert test_result.primary > 0.9
            assert "macro_f1" in val_result.metrics

    def test_segmentation_scores_pixels_and_honours_ignore_label(self) -> None:
        """Dense tasks are scored per pixel and skip the ignore label."""
        n, h, w = 12, 4, 4
        generator = torch.Generator().manual_seed(1)
        labels = torch.randint(0, NUM_CLASSES, (n, h, w), generator=generator)
        features = _separable(labels)
        # Ignored pixels carry garbage features; they must not be trained on
        # or scored.
        labels[:, 0, 0] = SEGMENTATION_IGNORE_LABEL
        features[:, 0, 0] = 100.0
        results = run_classifier_probes(
            config=_config(TaskType.SEGMENTATION),
            train_embeddings=features[:8],
            train_labels=labels[:8],
            val_embeddings=features[8:10],
            val_labels=labels[8:10],
            test_embeddings=features[10:],
            test_labels=labels[10:],
            probe_config=_fast_config(("rf",), max_train_samples=50),
        )
        val_result, test_result = results["rf"]
        assert val_result.primary_metric_key == "miou"
        assert val_result.primary > 0.7
        assert test_result is not None and test_result.primary > 0.7

    def test_segmentation_requires_one_embedding_per_label_pixel(self) -> None:
        """A token-to-block mismatch is refused rather than resampled."""
        with pytest.raises(ValueError, match="one embedding per label pixel"):
            run_classifier_probes(
                config=_config(TaskType.SEGMENTATION),
                train_embeddings=torch.zeros(2, 2, 2, DIM),
                train_labels=torch.zeros(2, 4, 4, dtype=torch.long),
                val_embeddings=torch.zeros(2, 2, 2, DIM),
                val_labels=torch.zeros(2, 4, 4, dtype=torch.long),
                test_embeddings=None,
                test_labels=None,
                probe_config=_fast_config(("rf",)),
            )

    def test_no_names_runs_nothing(self) -> None:
        """An empty names tuple is the off switch."""
        results = run_classifier_probes(
            config=_config(TaskType.CLASSIFICATION),
            train_embeddings=torch.zeros(4, DIM),
            train_labels=torch.zeros(4, dtype=torch.long),
            val_embeddings=torch.zeros(4, DIM),
            val_labels=torch.zeros(4, dtype=torch.long),
            test_embeddings=None,
            test_labels=None,
            probe_config=ClassifierProbeConfig(),
        )
        assert results == {}

    def test_task_name_carries_the_predictor(self) -> None:
        """The synthetic task name appends _clf_{predictor}."""
        assert (
            classifier_task_name("pastis_ws16_ps1_s2", "rf")
            == "pastis_ws16_ps1_s2_clf_rf"
        )

"""Off-the-shelf classifiers fit on already-materialized embeddings.

Our probe protocol scores an embedding with two fixed readouts: a kNN and a
linear layer trained by SGD (plus, on the AEF trials, a closed-form ridge). Both
are linear in the embedding, so a gap between two arms could equally be "the
information is not there" or "the information is there but not linearly
readable". Fitting a few standard non-linear classifiers on the same embeddings
separates those readings without adding a hyperparameter search: everything
here runs at library defaults (random forest, xgboost, logistic regression, and
a one-hidden-layer MLP).

Two entry points share one predictor registry:

- :func:`classifier_scores` fits one predictor on a small training draw and
  returns per-class scores for an eval set. The AEF balanced trials call it once
  per fold (``BalancedTrialConfig.classifiers``).
- :func:`run_classifier_probes` runs every requested predictor under OUR
  train -> val / test protocol on the embeddings a KNN / linear-probe task has
  already materialized. Dense (segmentation) tasks are flattened to one row per
  labeled pixel with a seeded cap on the training rows, since a random forest
  on tens of millions of pixels is neither tractable nor informative. Results
  are reported under synthetic task names (``{host}_clf_{predictor}``) so a
  forest's mIoU can never be read as the linear probe's.

The predictors are plain scikit-learn / xgboost estimators on CPU numpy (xgboost
may use the GPU when one is present); the imports are lazy so the training
image does not need them unless a task asks.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from olmoearth_pretrain.evals.datasets.configs import EvalDatasetConfig
from olmoearth_pretrain.evals.metrics import (
    SEGMENTATION_IGNORE_LABEL,
    EvalMetric,
    EvalResult,
    classification_metrics,
    segmentation_metrics,
)
from olmoearth_pretrain.evals.task_types import TaskType

logger = logging.getLogger(__name__)

# Marks the synthetic task names the probes report under, e.g.
# "pastis_..._landsat_clf_rf". Like the balanced trials' "aeftrial", the
# predictor lives in the TASK name so its number cannot be read as the host
# task's own linear-probe number.
TASK_SUFFIX = "clf"

RANDOM_FOREST = "rf"
XGBOOST = "xgb"
LOGISTIC_REGRESSION = "logreg"
MLP = "mlp"
CLASSIFIERS: tuple[str, ...] = (RANDOM_FOREST, XGBOOST, LOGISTIC_REGRESSION, MLP)

# Cap on the worker threads handed to the estimators. ``os.cpu_count()`` inside
# a container reports the host's cores, not the job's share, so -1 would
# oversubscribe; the CPU affinity mask is the honest figure where available.
MAX_DEFAULT_JOBS = 16


def classifier_task_name(host_task: str, predictor: str) -> str:
    """Synthetic task name a classifier probe's result is reported under."""
    return f"{host_task}_{TASK_SUFFIX}_{predictor}"


def default_n_jobs() -> int:
    """Worker threads for the estimators: the job's CPU share, capped."""
    try:
        cpus = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:  # macOS
        cpus = os.cpu_count() or 1
    return max(1, min(MAX_DEFAULT_JOBS, cpus))


@dataclass
class ClassifierProbeConfig:
    """Which off-the-shelf classifiers to fit, and their (default) settings.

    ``names`` empty means disabled, so the config can sit on every task as a
    non-optional field and be switched on from the sweep CLI with one override
    (``classifier_probes.names=[rf,xgb]``). The estimator settings are the
    libraries' own defaults on purpose: the point is a fixed, hyperparameter-free
    readout, the same argument AEF makes for lambda = 0.
    """

    names: tuple[str, ...] = ()
    seed: int = 0
    # None -> default_n_jobs().
    n_jobs: int | None = None
    # Dense tasks only: cap on the (valid-label) training pixels the estimators
    # see, drawn at random with ``seed``. None = every labeled pixel.
    max_train_samples: int | None = 1_000_000
    # Eval rows are scored in chunks of this size to bound the estimators'
    # transient memory (a forest's predict_proba materializes one (rows, C)
    # array per tree in flight).
    predict_chunk_rows: int = 1_000_000
    # scikit-learn RandomForestClassifier defaults.
    rf_n_estimators: int = 100
    rf_max_depth: int | None = None
    # 1 grows every tree until its leaves are pure; on millions of noisy pixel
    # rows that is ~one leaf per row and tens of GB of trees, so dense tasks
    # should raise it (5 caps the node count at 2N/5).
    rf_min_samples_leaf: int = 1
    # xgboost library defaults (eta 0.3, depth 6, 100 rounds, hist).
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.3
    # "auto" -> cuda when torch sees a GPU, else cpu.
    xgb_device: str = "auto"
    # scikit-learn LogisticRegression defaults (L2, C=1, lbfgs, multinomial),
    # with max_iter raised from 100 so the fit actually converges.
    logreg_c: float = 1.0
    logreg_max_iter: int = 1000
    # A two-layer MLP (one hidden ReLU layer, softmax output): scikit-learn's
    # MLPClassifier at its defaults (adam, lr 1e-3, batch 200, L2 1e-4, stops
    # when the loss plateaus for 10 epochs) behind a StandardScaler -- the MLP
    # is the one predictor here that is scale-sensitive, and our linear probe
    # puts a BatchNorm in front for the same reason. Hidden width 256 rather
    # than sklearn's 100 so a 128-d embedding is not squeezed on the way in.
    mlp_hidden: int = 256
    mlp_max_iter: int = 300

    def resolved_n_jobs(self) -> int:
        """The worker-thread count the estimators are built with."""
        return self.n_jobs if self.n_jobs is not None else default_n_jobs()

    def validate(self) -> None:
        """Raise on an unknown predictor name."""
        unknown = [name for name in self.names if name not in CLASSIFIERS]
        if unknown:
            raise ValueError(
                f"Unknown classifier probe(s) {unknown}; choose from {list(CLASSIFIERS)}"
            )


@dataclass
class _Fitted:
    """A fitted estimator plus the class ids its probability columns map to."""

    model: Any
    # Sorted original class ids present in the training rows. The estimator was
    # fit on their 0..k-1 encoding (xgboost requires contiguous labels), so its
    # predict_proba columns line up with this array.
    classes: np.ndarray


def _build_estimator(name: str, config: ClassifierProbeConfig) -> Any:
    n_jobs = config.resolved_n_jobs()
    if name == RANDOM_FOREST:
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=config.rf_n_estimators,
            max_depth=config.rf_max_depth,
            min_samples_leaf=config.rf_min_samples_leaf,
            n_jobs=n_jobs,
            random_state=config.seed,
        )
    if name == XGBOOST:
        from xgboost import XGBClassifier

        device = config.xgb_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return XGBClassifier(
            n_estimators=config.xgb_n_estimators,
            max_depth=config.xgb_max_depth,
            learning_rate=config.xgb_learning_rate,
            tree_method="hist",
            device=device,
            n_jobs=n_jobs,
            random_state=config.seed,
            verbosity=0,
        )
    if name == LOGISTIC_REGRESSION:
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(C=config.logreg_c, max_iter=config.logreg_max_iter)
    if name == MLP:
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(config.mlp_hidden,),
                max_iter=config.mlp_max_iter,
                random_state=config.seed,
            ),
        )
    raise ValueError(f"Unknown classifier probe '{name}'; choose from {CLASSIFIERS}")


def fit_classifier(
    name: str, config: ClassifierProbeConfig, features: np.ndarray, labels: np.ndarray
) -> _Fitted:
    """Fit one predictor on (N, D) features and integer (N,) labels."""
    classes, encoded = np.unique(labels, return_inverse=True)
    if classes.size < 2:
        # Nothing to discriminate; predict_scores handles the degenerate case.
        return _Fitted(model=None, classes=classes)
    model = _build_estimator(name, config)
    model.fit(features, encoded)
    return _Fitted(model=model, classes=classes)


def predict_scores(
    fitted: _Fitted, features: np.ndarray, num_classes: int, chunk_rows: int
) -> np.ndarray:
    """Per-class scores of shape (N, num_classes), float32.

    Classes absent from the training rows get a column of zeros, so the output
    lines up with the task's class ids however many the draw happened to hold.
    """
    n_rows = features.shape[0]
    scores = np.zeros((n_rows, num_classes), dtype=np.float32)
    if fitted.model is None:
        if fitted.classes.size == 1:
            scores[:, int(fitted.classes[0])] = 1.0
        return scores
    for start in range(0, n_rows, max(1, chunk_rows)):
        stop = min(start + chunk_rows, n_rows)
        proba = fitted.model.predict_proba(features[start:stop])
        scores[start:stop, fitted.classes] = np.asarray(proba, dtype=np.float32)
    return scores


def classifier_scores(
    name: str,
    config: ClassifierProbeConfig,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    eval_features: np.ndarray,
    num_classes: int,
) -> torch.Tensor:
    """Fit ``name`` on the training rows and score the eval rows: (N, C) float32."""
    fitted = fit_classifier(name, config, train_features, train_labels)
    scores = predict_scores(
        fitted, eval_features, num_classes, config.predict_chunk_rows
    )
    return torch.from_numpy(scores)


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """A float32 numpy view (or copy) of a tensor, whatever device it is on."""
    return tensor.detach().to("cpu", torch.float32).contiguous().numpy()


def _flatten_dense(
    embeddings: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """(N, H, W, D) embeddings + (N, H, W) labels -> one row per pixel."""
    if embeddings.ndim != 4 or labels.ndim != 3:
        raise ValueError(
            f"Dense classifier probes expect (N, H, W, D) embeddings and (N, H, W) "
            f"labels, got {tuple(embeddings.shape)} and {tuple(labels.shape)}"
        )
    if tuple(embeddings.shape[:3]) != tuple(labels.shape):
        # The linear probe rearranges one token into a ps x ps block of
        # logits; a per-row classifier has no such output shape, so it needs
        # one embedding per label pixel (patch_size=1 at the label resolution).
        raise ValueError(
            f"Dense classifier probes need one embedding per label pixel, got "
            f"embeddings {tuple(embeddings.shape)} vs labels {tuple(labels.shape)}"
        )
    return embeddings.reshape(-1, embeddings.shape[-1]), labels.reshape(-1)


def _training_rows(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    config: ClassifierProbeConfig,
    dense: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Valid-label training rows as numpy, capped for dense tasks."""
    valid = (labels >= 0) & (labels < num_classes)
    if dense:
        valid &= labels != SEGMENTATION_IGNORE_LABEL
    indices = valid.nonzero(as_tuple=True)[0]
    if dense and config.max_train_samples is not None:
        if indices.numel() > config.max_train_samples:
            rng = np.random.default_rng(config.seed)
            chosen = rng.choice(
                indices.numel(), size=config.max_train_samples, replace=False
            )
            indices = indices[torch.from_numpy(np.sort(chosen))]
    return _to_numpy(features[indices]), labels[indices].long().numpy()


def run_classifier_probes(
    config: EvalDatasetConfig,
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    val_embeddings: torch.Tensor,
    val_labels: torch.Tensor,
    test_embeddings: torch.Tensor | None,
    test_labels: torch.Tensor | None,
    probe_config: ClassifierProbeConfig,
    primary_metric: EvalMetric | None = None,
    primary_metric_class: int | None = None,
) -> dict[str, tuple[EvalResult, EvalResult | None]]:
    """Fit each requested classifier on train and score val (and test).

    Args:
        config: Dataset config (task type and class count).
        train_embeddings: (N, D) for classification, (N, H, W, D) for
            segmentation -- the tensors the host probe consumed.
        train_labels: (N,) or (N, H, W) integer labels.
        val_embeddings: Same layout as train.
        val_labels: Same layout as train labels.
        test_embeddings: Optional test split, same layout.
        test_labels: Optional test labels.
        probe_config: Which predictors, and their settings.
        primary_metric: The host task's primary metric, so the synthetic tasks
            rank by the same number.
        primary_metric_class: Class index for a CLASS_F1 primary metric.

    Returns:
        predictor name -> (val EvalResult, test EvalResult or None).
    """
    probe_config.validate()
    if not probe_config.names:
        return {}
    if config.task_type not in (TaskType.CLASSIFICATION, TaskType.SEGMENTATION):
        raise ValueError(
            f"Classifier probes support classification and segmentation tasks, "
            f"got '{config.task_type.value}'"
        )
    if config.is_multilabel:
        raise ValueError("Classifier probes do not support multilabel tasks")
    dense = config.task_type == TaskType.SEGMENTATION
    num_classes = config.num_classes

    if dense:
        train_rows, train_row_labels = _flatten_dense(train_embeddings, train_labels)
    else:
        train_rows, train_row_labels = train_embeddings, train_labels
    train_x, train_y = _training_rows(
        train_rows, train_row_labels, num_classes, probe_config, dense
    )
    logger.info(
        f"Classifier probes {list(probe_config.names)}: fitting on {train_x.shape[0]} "
        f"rows x {train_x.shape[1]} dims ({np.unique(train_y).size}/{num_classes} "
        f"classes present)"
    )

    splits: list[tuple[str, torch.Tensor, torch.Tensor]] = [
        ("val", val_embeddings, val_labels)
    ]
    if test_embeddings is not None and test_labels is not None:
        splits.append(("test", test_embeddings, test_labels))

    results: dict[str, tuple[EvalResult, EvalResult | None]] = {}
    for name in probe_config.names:
        fit_start = time.time()
        fitted = fit_classifier(name, probe_config, train_x, train_y)
        logger.info(f"Classifier probe [{name}]: fit in {time.time() - fit_start:.1f}s")
        per_split: dict[str, EvalResult] = {}
        for split, embeddings, labels in splits:
            score_start = time.time()
            per_split[split] = _score_split(
                fitted,
                embeddings,
                labels,
                config,
                probe_config,
                primary_metric,
                primary_metric_class,
            )
            logger.info(
                f"Classifier probe [{name}] {split}: {per_split[split].primary:.4f} "
                f"({per_split[split].primary_metric_key}) in "
                f"{time.time() - score_start:.1f}s"
            )
        results[name] = (per_split["val"], per_split.get("test"))
    return results


def _score_split(
    fitted: _Fitted,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    config: EvalDatasetConfig,
    probe_config: ClassifierProbeConfig,
    primary_metric: EvalMetric | None,
    primary_metric_class: int | None,
) -> EvalResult:
    """Score one split with the shared metric code, at the host's layout."""
    num_classes = config.num_classes
    if config.task_type == TaskType.SEGMENTATION:
        rows, _ = _flatten_dense(embeddings, labels)
        scores = predict_scores(
            fitted, _to_numpy(rows), num_classes, probe_config.predict_chunk_rows
        )
        n, h, w = labels.shape
        # (N*H*W, C) -> (N, C, H, W), the layout segmentation_metrics expects.
        scores_tensor = torch.from_numpy(scores).reshape(n, h, w, num_classes)
        scores_tensor = scores_tensor.permute(0, 3, 1, 2).contiguous()
        predictions = scores_tensor.argmax(dim=1)
        return segmentation_metrics(
            predictions,
            labels.long(),
            num_classes=num_classes,
            ignore_label=SEGMENTATION_IGNORE_LABEL,
            scores=scores_tensor,
            primary_metric=primary_metric,
            primary_metric_class=primary_metric_class,
        )
    scores = predict_scores(
        fitted, _to_numpy(embeddings), num_classes, probe_config.predict_chunk_rows
    )
    scores_tensor = torch.from_numpy(scores)
    return classification_metrics(
        predictions=scores_tensor.argmax(dim=1),
        labels=labels.long(),
        scores=scores_tensor,
        primary_metric=primary_metric,
        primary_metric_class=primary_metric_class,
    )

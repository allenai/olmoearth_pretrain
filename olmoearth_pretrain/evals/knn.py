"""KNN evals of OlmoEarth Pretrain models."""

import logging

import numpy as np
import torch
import torch.nn as nn
from olmo_core.data.utils import get_rng
from sklearn.metrics import accuracy_score, f1_score

from olmoearth_pretrain.evals.datasets.configs import EvalDatasetConfig

logger = logging.getLogger(__name__)


def run_knn(
    config: EvalDatasetConfig,
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    val_embeddings: torch.Tensor,
    val_labels: torch.Tensor,
    test_embeddings: torch.Tensor | None,
    test_labels: torch.Tensor | None,
    device: torch.device,
    k: int = 20,
    skip_idx: bool = False,
    n_bootstrap: int = 0,
    bootstrap_seed: int = 42,
) -> dict[str, float | dict]:
    """Run KNN on the OlmoEarth Pretrain model.

    Args:
        config: Dataset configuration
        train_embeddings: Training embeddings
        train_labels: Training labels
        val_embeddings: Validation embeddings
        val_labels: Validation labels
        test_embeddings: Test embeddings (optional)
        test_labels: Test labels (optional)
        device: Device to run on
        k: Number of nearest neighbors
        skip_idx: Whether to skip the first neighbor (for train set evaluation)
        n_bootstrap: Number of bootstrap samples for uncertainty estimation (0 = no bootstrap)
        bootstrap_seed: Random seed for bootstrap sampling

    Returns:
        Dictionary with keys:
            - val_score: Validation score
            - test_score: Test score (0.0 if no test set)
            - bootstrap_stats: Bootstrap statistics dict (empty dict if n_bootstrap == 0)
    """
    if not config.is_multilabel:
        val_predictions = _run_knn_for_k(
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            test_embeddings=val_embeddings,
            num_classes=config.num_classes,
            k=k,
            device=device,
            skip_idx=skip_idx,
        )
        val_score = accuracy_score(y_true=val_labels, y_pred=val_predictions)

        if test_embeddings is not None:
            if test_labels is None:
                raise ValueError("Can't have test embeddings without test labels")
            test_predictions = _run_knn_for_k(
                train_embeddings=train_embeddings,
                train_labels=train_labels,
                test_embeddings=test_embeddings,
                num_classes=config.num_classes,
                k=k,
                device=device,
                skip_idx=skip_idx,
            )
            test_score = accuracy_score(y_true=test_labels, y_pred=test_predictions)

            # Perform bootstrap sampling if requested
            if n_bootstrap > 0:
                bootstrap_stats = _bootstrap_knn_test(
                    config=config,
                    train_embeddings=train_embeddings,
                    train_labels=train_labels,
                    test_embeddings=test_embeddings,
                    test_labels=test_labels,
                    device=device,
                    k=k,
                    skip_idx=skip_idx,
                    n_bootstrap=n_bootstrap,
                    seed=bootstrap_seed,
                )
            else:
                bootstrap_stats = {}
        else:
            test_score = 0.0
            bootstrap_stats = {}

        return {
            "val_score": val_score,
            "test_score": test_score,
            "bootstrap_stats": bootstrap_stats,
        }
    else:
        # multilabel dataset, e.g., BigEarthNet
        # we will run KNN or K-Means once per class to compute predictions
        # labels are shape (num_samples, num_classes)
        assert config.num_classes == train_labels.shape[-1]
        assert config.num_classes == val_labels.shape[-1]
        if test_labels is not None:
            assert config.num_classes == test_labels.shape[-1]
        val_predictions = []
        test_predictions = []
        for class_idx in range(config.num_classes):
            train_single_labels = train_labels[:, class_idx]  # (num_samples)
            single_val_predictions = _run_knn_for_k(
                train_embeddings=train_embeddings,
                train_labels=train_single_labels,
                test_embeddings=val_embeddings,
                num_classes=2,  # binary prediction for each class
                k=k,
                device=device,
                skip_idx=skip_idx,
            )  # (num_samples)
            val_predictions.append(single_val_predictions)

            if test_embeddings is not None:
                if test_labels is None:
                    raise ValueError("Can't have test embeddings without test labels")
                single_test_predictions = _run_knn_for_k(
                    train_embeddings=train_embeddings,
                    train_labels=train_single_labels,
                    test_embeddings=test_embeddings,
                    num_classes=2,  # binary prediction for each class
                    k=k,
                    device=device,
                    skip_idx=skip_idx,
                )  # (num_samples)
                test_predictions.append(single_test_predictions)

        val_predictions = torch.stack(
            val_predictions, dim=1
        )  # (num_samples, num_classes)
        val_score = f1_score(y_true=val_labels, y_pred=val_predictions, average="micro")
        if len(test_predictions) > 0:
            test_predictions = torch.stack(
                test_predictions, dim=1
            )  # (num_samples, num_classes)
            test_score = f1_score(
                y_true=test_labels, y_pred=test_predictions, average="micro"
            )

            # Perform bootstrap sampling if requested
            if n_bootstrap > 0:
                bootstrap_stats = _bootstrap_knn_test(
                    config=config,
                    train_embeddings=train_embeddings,
                    train_labels=train_labels,
                    test_embeddings=test_embeddings,
                    test_labels=test_labels,
                    device=device,
                    k=k,
                    skip_idx=skip_idx,
                    n_bootstrap=n_bootstrap,
                    seed=bootstrap_seed,
                )
            else:
                bootstrap_stats = {}
        else:
            test_score = 0.0
            bootstrap_stats = {}

        return {
            "val_score": val_score,
            "test_score": test_score,
            "bootstrap_stats": bootstrap_stats,
        }


def _bootstrap_knn_test(
    config: EvalDatasetConfig,
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    test_embeddings: torch.Tensor,
    test_labels: torch.Tensor,
    device: torch.device,
    k: int,
    skip_idx: bool,
    n_bootstrap: int,
    seed: int,
) -> dict:
    """Perform bootstrap sampling on test set to estimate uncertainty.

    Args:
        config: Dataset configuration
        train_embeddings: Training embeddings
        train_labels: Training labels
        test_embeddings: Test embeddings
        test_labels: Test labels
        device: Device to run on
        k: Number of nearest neighbors
        skip_idx: Whether to skip the first neighbor
        n_bootstrap: Number of bootstrap iterations
        seed: Random seed for reproducibility

    Returns:
        Dictionary with bootstrap statistics including:
            - bootstrap_scores: list of all bootstrap scores
            - mean: mean of bootstrap distribution
            - std: standard deviation of bootstrap distribution
            - ci_lower: lower bound of 95% confidence interval
            - ci_upper: upper bound of 95% confidence interval
    """
    # Optimized bootstrap: compute predictions once, then resample
    logger.info(
        f"Computing predictions once for {test_embeddings.shape[0]} test samples..."
    )

    # Compute predictions ONCE for all test samples
    if not config.is_multilabel:
        all_predictions = _run_knn_for_k(
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            test_embeddings=test_embeddings,
            num_classes=config.num_classes,
            k=k,
            device=device,
            skip_idx=skip_idx,
        )
    else:
        # Multilabel case: compute predictions once per class
        all_predictions = []
        for class_idx in range(config.num_classes):
            train_single_labels = train_labels[:, class_idx]
            single_predictions = _run_knn_for_k(
                train_embeddings=train_embeddings,
                train_labels=train_single_labels,
                test_embeddings=test_embeddings,
                num_classes=2,
                k=k,
                device=device,
                skip_idx=skip_idx,
            )
            all_predictions.append(single_predictions)
        all_predictions = torch.stack(all_predictions, dim=1)

    # Bootstrap resample the predictions (very fast!)
    rng = get_rng(seed)
    n_test_samples = test_embeddings.shape[0]
    bootstrap_scores = []

    logger.info(
        f"Running {n_bootstrap} bootstrap iterations on precomputed predictions..."
    )

    for i in range(n_bootstrap):
        # Resample indices only - no cosine similarity computation!
        bootstrap_indices = rng.choice(
            n_test_samples, size=n_test_samples, replace=True
        )

        bootstrap_preds = all_predictions[bootstrap_indices]
        bootstrap_labels = test_labels[bootstrap_indices]

        # Compute metric on resampled predictions
        if not config.is_multilabel:
            score = accuracy_score(y_true=bootstrap_labels, y_pred=bootstrap_preds)
        else:
            score = f1_score(
                y_true=bootstrap_labels,
                y_pred=bootstrap_preds,
                average="micro",
            )

        bootstrap_scores.append(score)

        if (i + 1) % 10 == 0:
            logger.info(f"Completed {i + 1}/{n_bootstrap} bootstrap iterations")

    bootstrap_scores_array = np.array(bootstrap_scores)

    # Calculate statistics
    mean = np.mean(bootstrap_scores_array)
    std = np.std(bootstrap_scores_array)
    ci_lower = np.percentile(bootstrap_scores_array, 2.5)
    ci_upper = np.percentile(bootstrap_scores_array, 97.5)

    logger.info(
        f"Bootstrap results: mean={mean:.4f}, std={std:.4f}, 95% CI=[{ci_lower:.4f}, {ci_upper:.4f}]"
    )

    return {
        "bootstrap_scores": bootstrap_scores_array.tolist(),
        "mean": float(mean),
        "std": float(std),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "n_bootstrap": n_bootstrap,
    }


def _run_knn_for_k(
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    test_embeddings: torch.Tensor,
    num_classes: int,
    k: int,
    device: torch.device,
    skip_idx: bool,
) -> torch.Tensor:
    """Run KNN classification with chunked batch processing for efficiency.

    Args:
        train_embeddings: Training embeddings of shape (n_train, embedding_dim)
        train_labels: Training labels of shape (n_train,)
        test_embeddings: Test embeddings of shape (n_test, embedding_dim)
        num_classes: Number of classes
        k: Number of nearest neighbors
        device: Device to run on
        skip_idx: Whether to skip the first neighbor

    Returns:
        Predictions of shape (n_test,)
    """
    # Move to device
    train_embeddings = train_embeddings.to(device)
    test_embeddings = test_embeddings.to(device)
    train_labels = train_labels.to(device)

    # Normalize embeddings once for efficient cosine similarity computation
    # Use eps=1e-8 to match torch.nn.functional.cosine_similarity default
    # Formula: similarity = (x1 · x2) / (max(||x1||_2, eps) · max(||x2||_2, eps))
    eps = 1e-8
    train_embeddings_norm = torch.nn.functional.normalize(
        train_embeddings, p=2, dim=1, eps=eps
    )

    # Process test embeddings in chunks to avoid memory explosion
    chunk_size = 2000
    n_test = test_embeddings.shape[0]
    all_predictions = []

    # Determine effective k (accounting for skip_idx)
    effective_k = k if not skip_idx else k + 1

    for start_idx in range(0, n_test, chunk_size):
        end_idx = min(start_idx + chunk_size, n_test)
        test_chunk = test_embeddings[start_idx:end_idx]

        # Normalize test chunk with same eps to match cosine_similarity
        test_chunk_norm = torch.nn.functional.normalize(test_chunk, p=2, dim=1, eps=eps)

        # Compute cosine similarity matrix: (chunk_size, n_train)
        # After normalization: x_norm = x / max(||x||_2, eps)
        # Dot product of normalized vectors gives: (x1 · x2) / (max(||x1||_2, eps) · max(||x2||_2, eps))
        # This matches torch.nn.functional.cosine_similarity formula exactly
        similarities = torch.mm(test_chunk_norm, train_embeddings_norm.T)

        # Get top-k neighbors for each test sample
        # If skip_idx is True, we'll get k+1 and then skip the first
        top_k_similarities, top_k_indices = torch.topk(
            similarities, k=effective_k, dim=1
        )

        # Handle skip_idx by removing first neighbor if needed
        if skip_idx:
            top_k_similarities = top_k_similarities[:, 1:]
            top_k_indices = top_k_indices[:, 1:]

        # Get labels for top-k neighbors: (chunk_size, k)
        top_k_labels = train_labels[top_k_indices]

        # Convert to one-hot encoding: (chunk_size, k, num_classes)
        top_k_onehots = nn.functional.one_hot(top_k_labels, num_classes=num_classes)

        # Compute weights: exp(similarity / 0.07)
        # top_k_similarities: (chunk_size, k)
        weights = torch.exp(top_k_similarities / 0.07)

        # Weighted sum of one-hot encodings: (chunk_size, num_classes)
        weighted_sum = (weights.unsqueeze(-1) * top_k_onehots).sum(dim=1)

        # Get predictions: (chunk_size,)
        chunk_predictions = torch.argmax(weighted_sum, dim=1)
        all_predictions.append(chunk_predictions)

    # Concatenate all predictions and return on CPU
    return torch.cat(all_predictions, dim=0).cpu()

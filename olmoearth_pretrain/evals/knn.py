"""KNN evals of OlmoEarth Pretrain models."""

import logging

import numpy as np
import torch
import torch.nn as nn
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
) -> tuple[float, float] | tuple[float, float, dict]:
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
        If n_bootstrap == 0: (val_score, test_score)
        If n_bootstrap > 0: (val_score, test_score, bootstrap_stats)
            where bootstrap_stats contains mean, std, and confidence intervals
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
                return val_score, test_score, bootstrap_stats
        else:
            test_score = 0.0

        if n_bootstrap > 0:
            # No test set, return empty bootstrap stats
            return val_score, test_score, {}
        return val_score, test_score
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
                return val_score, test_score, bootstrap_stats
        else:
            test_score = 0.0

        if n_bootstrap > 0:
            # No test set, return empty bootstrap stats
            return val_score, test_score, {}
        return val_score, test_score


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
    rng = np.random.RandomState(seed)
    n_test_samples = test_embeddings.shape[0]
    bootstrap_scores = []

    logger.info(
        f"Running {n_bootstrap} bootstrap iterations on {n_test_samples} test samples..."
    )

    for i in range(n_bootstrap):
        # Sample with replacement
        bootstrap_indices = rng.choice(
            n_test_samples, size=n_test_samples, replace=True
        )
        bootstrap_test_embeddings = test_embeddings[bootstrap_indices]
        bootstrap_test_labels = test_labels[bootstrap_indices]

        # Run KNN on bootstrap sample
        if not config.is_multilabel:
            bootstrap_predictions = _run_knn_for_k(
                train_embeddings=train_embeddings,
                train_labels=train_labels,
                test_embeddings=bootstrap_test_embeddings,
                num_classes=config.num_classes,
                k=k,
                device=device,
                skip_idx=skip_idx,
            )
            score = accuracy_score(
                y_true=bootstrap_test_labels, y_pred=bootstrap_predictions
            )
        else:
            # Multilabel case
            bootstrap_predictions = []
            for class_idx in range(config.num_classes):
                train_single_labels = train_labels[:, class_idx]
                single_bootstrap_predictions = _run_knn_for_k(
                    train_embeddings=train_embeddings,
                    train_labels=train_single_labels,
                    test_embeddings=bootstrap_test_embeddings,
                    num_classes=2,
                    k=k,
                    device=device,
                    skip_idx=skip_idx,
                )
                bootstrap_predictions.append(single_bootstrap_predictions)

            bootstrap_predictions = torch.stack(bootstrap_predictions, dim=1)
            score = f1_score(
                y_true=bootstrap_test_labels,
                y_pred=bootstrap_predictions,
                average="micro",
            )

        bootstrap_scores.append(score)

        if (i + 1) % 10 == 0:
            logger.info(f"Completed {i + 1}/{n_bootstrap} bootstrap iterations")

    bootstrap_scores = np.array(bootstrap_scores)

    # Calculate statistics
    mean = np.mean(bootstrap_scores)
    std = np.std(bootstrap_scores)
    ci_lower = np.percentile(bootstrap_scores, 2.5)
    ci_upper = np.percentile(bootstrap_scores, 97.5)

    logger.info(
        f"Bootstrap results: mean={mean:.4f}, std={std:.4f}, 95% CI=[{ci_lower:.4f}, {ci_upper:.4f}]"
    )

    return {
        "bootstrap_scores": bootstrap_scores.tolist(),
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
    train_embeddings = train_embeddings.to(device)
    test_embeddings = test_embeddings.to(device)
    train_labels = train_labels.to(device)
    cos = nn.CosineSimilarity(dim=-1)
    all_preds = []
    for idx in range(test_embeddings.shape[0]):
        test_embedding = test_embeddings[idx].unsqueeze(dim=0)
        test_embedding = (
            test_embeddings[idx].unsqueeze(dim=0).repeat(train_embeddings.shape[0], 1)
        )
        sims = cos(test_embedding, train_embeddings)
        top_k = torch.topk(sims, k=k)
        if skip_idx:
            top_k_values = top_k.values[1:]
            top_k_indices = top_k.indices[1:]
        else:
            top_k_values = top_k.values
            top_k_indices = top_k.indices

        fetched_labels = train_labels[top_k_indices]
        fetched_onehots = nn.functional.one_hot(fetched_labels, num_classes=num_classes)
        distances = top_k_values.clone().div_(0.07).exp_()
        weighted_sum_onehots = (distances.unsqueeze(dim=1) * fetched_onehots).sum(dim=0)
        prediction = torch.argmax(weighted_sum_onehots)
        all_preds.append(prediction)

    return torch.LongTensor(all_preds).cpu()

"""KNN evals of OlmoEarth Pretrain models."""

import logging

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
) -> tuple[float, float]:
    """Run KNN on the OlmoEarth Pretrain model."""
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
        else:
            test_score = 0.0
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
        else:
            test_score = 0.0
        return val_score, test_score


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

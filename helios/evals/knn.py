"""KNN evals of Helios models."""

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score


def run_knn(
    eval_type: str,
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    test_embeddings: torch.Tensor,
    test_labels: torch.Tensor,
    num_classes: int,
    is_multilabel: bool,
    device: torch.device,
    skip_idx: bool = False,
) -> float:
    """Run KNN on the Helios model."""
    if not eval_type.startswith("KNN-"):
        raise ValueError(f"Unexpected eval type {eval_type}")
    k = int(eval_type.split("-")[-1])
    if not is_multilabel:
        predictions = _run_knn_for_k(
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            test_embeddings=test_embeddings,
            num_classes=num_classes,
            k=k,
            device=device,
            skip_idx=skip_idx,
        )
        return accuracy_score(y_true=test_labels, y_pred=predictions)
    else:
        # multilabel dataset, e.g., BigEarthNet
        # we will run KNN or K-Means once per class to compute predictions
        # labels are shape (num_samples, num_classes)
        assert num_classes == train_labels.shape[-1]
        assert num_classes == test_labels.shape[-1]
        predictions = []
        for class_idx in range(num_classes):
            train_single_labels = train_labels[:, class_idx]  # (num_samples)
            single_predictions = _run_knn_for_k(
                train_embeddings=train_embeddings,
                train_labels=train_single_labels,
                test_embeddings=test_embeddings,
                num_classes=2,  # binary prediction for each class
                k=k,
                device=device,
                skip_idx=skip_idx,
            )  # (num_samples)
            predictions.append(single_predictions)

        predictions = torch.stack(predictions, dim=1)  # (num_samples, num_classes)
        return f1_score(y_true=test_labels, y_pred=predictions, average="micro")


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

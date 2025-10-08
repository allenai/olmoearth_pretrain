"""Test Linear Probe."""

import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.evals.datasets.configs import EvalDatasetConfig, TaskType
from olmoearth_pretrain.evals.linear_probe import train_and_eval_probe


def test_probe_cls() -> None:
    """Test linear probe for classification."""
    batch_size, embedding_dim = 64, 16
    train_embeddings = torch.rand(64, embedding_dim)
    val_embeddings = torch.rand(64, embedding_dim)
    test_embeddings = torch.rand(64, embedding_dim)
    train_labels = torch.ones(64).long()
    train_labels[:32] = 0
    val_labels = torch.ones(64).long()
    val_labels[:32] = 0
    test_labels = torch.ones(64).long()
    test_labels[:32] = 0

    config = EvalDatasetConfig(
        task_type=TaskType.CLASSIFICATION,
        imputes=[],
        num_classes=2,
        is_multilabel=False,
        supported_modalities=[Modality.SENTINEL2_L2A.name],
    )

    # just testing it runs - since the data is random,
    # performance should be about random (accuracy = 0.5)
    _ = train_and_eval_probe(
        config=config,
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        val_embeddings=val_embeddings,
        val_labels=val_labels,
        test_embeddings=test_embeddings,
        test_labels=test_labels,
        device=train_embeddings.device,
        batch_size=batch_size,
        lr=0.1,
    )


def test_probe_seg() -> None:
    """Test linear probe for segmentation."""
    (
        batch_size,
        h,
        w,
        embedding_dim,
        patch_size,
    ) = (
        64,
        8,
        8,
        16,
        4,
    )
    train_embeddings = torch.rand(64, h // patch_size, w // patch_size, embedding_dim)
    val_embeddings = torch.rand(64, h // patch_size, w // patch_size, embedding_dim)
    test_embeddings = torch.rand(64, h // patch_size, w // patch_size, embedding_dim)
    train_labels = torch.ones(64, h, w).long()
    train_labels[:32] = 0
    val_labels = torch.ones(64, h, w).long()
    val_labels[:32] = 0
    test_labels = torch.ones(64, h, w).long()
    test_labels[:32] = 0

    config = EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[],
        num_classes=2,
        is_multilabel=False,
        height_width=h,
        supported_modalities=[Modality.SENTINEL2_L2A.name],
    )

    # just testing it runs - since the data is random,
    # performance should be about random (accuracy = 0.5)
    _ = train_and_eval_probe(
        config=config,
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        val_embeddings=val_embeddings,
        val_labels=val_labels,
        test_embeddings=test_embeddings,
        test_labels=test_labels,
        device=train_embeddings.device,
        batch_size=batch_size,
        lr=0.1,
    )

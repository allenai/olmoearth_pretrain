"""Finetune the Helios model on a downstream task."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from olmo_core.config import Config
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from helios.evals.datasets import get_eval_dataset
from helios.evals.datasets.configs import (
    EvalDatasetConfig,
    TaskType,
    dataset_to_config,
)
from helios.evals.datasets.utils import eval_collate_fn
from helios.evals.metrics import mean_iou
from helios.evals.utils import adjust_learning_rate
from helios.nn.flexihelios import Encoder, PoolingType
from helios.train.masking import MaskedHeliosSample

# Fine-tuning learning rates
FT_LRs = [1e-5, 3e-5, 6e-5, 1e-4, 3e-4, 6e-4, 1e-3, 3e-3, 6e-3]
# Linear probing learning rates
LP_LRs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]


def finetune_and_eval_cls(
    task_name: str,
    checkpoint_path: Path,
    freeze_encoder: bool,
    patch_size: int,
    pooling_type: PoolingType,
    lr: float,
    epochs: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> float:
    """Finetune the Helios model on a classification task and evaluate it.

    Args:
        task_name: The name of the task to finetune on.
        checkpoint_path: The path to the checkpoint to use for training.
        freeze_encoder: Whether to freeze the encoder, if True, it will be linear probing.
        patch_size: The patch size to use for training.
        pooling_type: The pooling type to use for training.
        lr: The learning rate to use for training.
        epochs: The number of epochs to train for.
        batch_size: The batch size to use for training.
        num_workers: The number of workers to use for training.
        device: The device to use for training.

    Returns:
        The validation accuracy of the finetuned model.
    """
    task_config = dataset_to_config(task_name)
    if task_config.task_type != TaskType.CLASSIFICATION:
        raise ValueError(f"Task {task_name} is not a classification task")

    # By default, we use the norm stats from the pretrained model
    norm_stats_from_pretrained = True
    if task_name == "mados":
        # MADOS has very different norm stats than our pretraining dataset
        norm_stats_from_pretrained = False

    train_loader = DataLoader(
        get_eval_dataset(
            eval_dataset=task_name,
            split="train",
            partition="default",
            norm_stats_from_pretrained=norm_stats_from_pretrained,
        ),
        collate_fn=eval_collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        get_eval_dataset(
            eval_dataset=task_name,
            split="valid",
            partition="default",
            norm_stats_from_pretrained=norm_stats_from_pretrained,
        ),
        collate_fn=eval_collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    finetuned_model = finetune_cls(
        task_config=task_config,
        data_loader=train_loader,
        checkpoint_path=checkpoint_path,
        freeze_encoder=freeze_encoder,
        lr=lr,
        epochs=epochs,
        patch_size=patch_size,
        pooling_type=pooling_type,
        device=device,
    )
    val_acc = evaluate_cls(
        task_config=task_config,
        data_loader=val_loader,
        finetuned_model=finetuned_model,
        device=device,
    )
    return val_acc


def finetune_and_eval_seg(
    task_name: str,
    checkpoint_path: Path,
    freeze_encoder: bool,
    patch_size: int,
    pooling_type: PoolingType,
    lr: float,
    epochs: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> float:
    """Finetune the Helios model on a segmentation task and evaluate it.

    Args:
        task_name: The name of the task to finetune on.
        checkpoint_path: The path to the checkpoint to use for training.
        freeze_encoder: Whether to freeze the encoder, if True, it will be linear probing.
        patch_size: The patch size to use for training.
        pooling_type: The pooling type to use for training.
        lr: The learning rate to use for training.
        epochs: The number of epochs to train for.
        batch_size: The batch size to use for training.
        num_workers: The number of workers to use for training.
        device: The device to use for training.

    Returns:
        The validation mean IoU of the finetuned model.
    """
    task_config = dataset_to_config(task_name)
    if task_config.task_type != TaskType.SEGMENTATION:
        raise ValueError(f"Task {task_name} is not a segmentation task")

    # By default, we use the norm stats from the pretrained model
    norm_stats_from_pretrained = True
    if task_name == "mados":
        # MADOS has very different norm stats than our pretraining dataset
        norm_stats_from_pretrained = False

    train_loader = DataLoader(
        get_eval_dataset(
            eval_dataset=task_name,
            split="train",
            partition="default",
            norm_stats_from_pretrained=norm_stats_from_pretrained,
        ),
        collate_fn=eval_collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        get_eval_dataset(
            eval_dataset=task_name,
            split="valid",
            partition="default",
            norm_stats_from_pretrained=norm_stats_from_pretrained,
        ),
        collate_fn=eval_collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    finetuned_model = finetune_seg(
        task_config=task_config,
        data_loader=train_loader,
        checkpoint_path=checkpoint_path,
        freeze_encoder=freeze_encoder,
        lr=lr,
        epochs=epochs,
        patch_size=patch_size,
        pooling_type=pooling_type,
        device=device,
    )
    val_miou = evaluate_seg(
        task_config=task_config,
        data_loader=val_loader,
        finetuned_model=finetuned_model,
        patch_size=patch_size,
        device=device,
    )
    return val_miou


class EncoderWithHead(nn.Module):
    """Encoder with a prediction head for a downstream task."""

    def __init__(
        self,
        encoder: Encoder,
        freeze_encoder: bool,
        task_type: TaskType,
        num_classes: int,
        patch_size: int,
        pooling_type: PoolingType,
    ) -> None:
        """Initialize the encoder with a prediction head.

        Args:
            encoder: The encoder to use.
            freeze_encoder: Whether to freeze the encoder, if True, it will be linear probing.
            task_type: The type of task to perform (classification or segmentation).
            num_classes: The number of classes to predict.
            patch_size: The patch size to use.
            pooling_type: The pooling type to use.
        """
        super().__init__()

        self.encoder = deepcopy(encoder)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.task_type = task_type
        self.patch_size = patch_size
        self.pooling_type = pooling_type

        if task_type == TaskType.CLASSIFICATION:
            self.head = nn.Linear(encoder.embedding_size, num_classes)
            self.spatial_pool = False
        elif task_type == TaskType.SEGMENTATION:
            logits_per_patch = int(num_classes * patch_size * patch_size)
            self.head = nn.Linear(encoder.embedding_size, logits_per_patch)
            self.spatial_pool = True
        else:
            raise ValueError(f"Invalid task type: {task_type}")

    def forward(self, batch: MaskedHeliosSample) -> torch.Tensor:
        """Forward pass."""
        batch_features = self.encoder(batch, patch_size=self.patch_size)
        batch_embeddings = batch_features.pool_unmasked_tokens(
            self.pooling_type, spatial_pooling=self.spatial_pool
        )
        output = self.head(batch_embeddings)
        return output


def load_config(checkpoint_path: Path) -> Config:
    """Load the config file from the checkpoint input directory."""
    assert (
        checkpoint_path / "config.json"
    ).exists(), f"Config file not found at {checkpoint_path}"

    with open(checkpoint_path / "config.json") as f:
        config_dict = json.load(f)
        model_config = Config.from_dict(config_dict["model"])

    return model_config


def finetune_cls(
    task_config: EvalDatasetConfig,
    data_loader: DataLoader,
    checkpoint_path: Path,
    freeze_encoder: bool,
    lr: float,
    epochs: int,
    patch_size: int,
    pooling_type: PoolingType,
    device: torch.device,
) -> nn.Module:
    """Finetune the Helios model on classification task.

    Args:
        task_config: The config of the task to finetune on.
        data_loader: The data loader to use for training.
        checkpoint_path: The path to the checkpoint to use for training.
        freeze_encoder: Whether to freeze the encoder, if True, it will be linear probing.
        lr: The learning rate to use for training.
        epochs: The number of epochs to train for.
        patch_size: The patch size to use for training.
        pooling_type: The pooling type to use for training.
        device: The device to use for training.

    Returns:
        The finetuned model.
    """
    # Build the model and load only the encoder
    model_config = load_config(checkpoint_path)
    model = model_config.build()

    load_model_and_optim_state(checkpoint_path / "model_and_optim", model)
    encoder = model.encoder

    finetuned_model = EncoderWithHead(
        encoder=encoder,
        freeze_encoder=freeze_encoder,
        task_type=task_config.task_type,
        num_classes=task_config.num_classes,
        patch_size=patch_size,
        pooling_type=pooling_type,
    ).to(device)

    finetuned_model = finetuned_model.train()
    optimizer = torch.optim.AdamW(finetuned_model.parameters(), lr=lr)

    if task_config.is_multilabel:
        loss_function: nn.Module = nn.MultiLabelSoftMarginLoss()
    else:
        loss_function = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print(f"Epoch {epoch} of {epochs}")
        for i, batch in enumerate(data_loader):
            masked_helios_sample, label = batch
            label = label.to(device=device)

            masked_helios_sample_dict = masked_helios_sample.as_dict(return_none=False)
            for key, val in masked_helios_sample_dict.items():
                if key == "timestamps":
                    masked_helios_sample_dict[key] = val.to(device=device)
                else:
                    masked_helios_sample_dict[key] = val.to(
                        device=device, dtype=torch.bfloat16
                    )

            masked_helios_sample = MaskedHeliosSample.from_dict(
                masked_helios_sample_dict
            )

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = finetuned_model(masked_helios_sample)
                loss = loss_function(logits, label)
                print(f"Loss: {loss.item()}")

            loss.backward()
            adjust_learning_rate(
                optimizer=optimizer,
                epoch=epoch + (i / len(data_loader)),
                total_epochs=epochs,
                warmup_epochs=int(epochs * 0.1),
                max_lr=lr,
                min_lr=1.0e-5,
            )
            torch.nn.utils.clip_grad_norm_(finetuned_model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

    return finetuned_model


def evaluate_cls(
    task_config: EvalDatasetConfig,
    data_loader: DataLoader,
    finetuned_model: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate the finetuned model on a classification task.

    Args:
        task_config: The config of the task to evaluate on.
        data_loader: The data loader to use for evaluation.
        finetuned_model: The finetuned model to evaluate.
        device: The device to use for evaluation.

    Returns:
        The accuracy of the finetuned model.
    """
    finetuned_model = finetuned_model.eval()

    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            masked_helios_sample, label = batch
            label = label.to(device=device)

            masked_helios_sample_dict = masked_helios_sample.as_dict(return_none=False)
            for key, val in masked_helios_sample_dict.items():
                if key == "timestamps":
                    masked_helios_sample_dict[key] = val.to(device=device)
                else:
                    masked_helios_sample_dict[key] = val.to(
                        device=device, dtype=torch.bfloat16
                    )

            masked_helios_sample = MaskedHeliosSample.from_dict(
                masked_helios_sample_dict
            )
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = finetuned_model(masked_helios_sample)

            all_logits.append(logits.float().cpu())
            all_labels.append(label.float().cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    if task_config.is_multilabel:
        all_preds = torch.sigmoid(all_logits) > 0.5
        return f1_score(all_labels, all_preds, average="micro")
    else:
        all_preds = torch.argmax(all_logits, dim=-1)
        return accuracy_score(all_labels, all_preds)


def finetune_seg(
    task_config: EvalDatasetConfig,
    data_loader: DataLoader,
    checkpoint_path: Path,
    freeze_encoder: bool,
    lr: float,
    epochs: int,
    patch_size: int,
    pooling_type: PoolingType = PoolingType.MEAN,
    device: torch.device = torch.device("cuda"),
) -> nn.Module:
    """Finetune the Helios model on a segmentation task.

    Args:
        task_config: The config of the task to finetune on.
        data_loader: The data loader to use for training.
        checkpoint_path: The path to the checkpoint to use for training.
        freeze_encoder: Whether to freeze the encoder, if True, it will be linear probing.
        lr: The learning rate to use for training.
        epochs: The number of epochs to train for.
        patch_size: The patch size to use for training.
        pooling_type: The pooling type to use for training.
        device: The device to use for training.

    Returns:
        The finetuned model.
    """
    # Build the model and load only the encoder
    model_config = load_config(checkpoint_path)
    model = model_config.build()

    load_model_and_optim_state(checkpoint_path / "model_and_optim", model)
    encoder = model.encoder

    finetuned_model = EncoderWithHead(
        encoder=encoder,
        freeze_encoder=freeze_encoder,
        task_type=task_config.task_type,
        num_classes=task_config.num_classes,
        patch_size=patch_size,
        pooling_type=pooling_type,
    ).to(device)

    finetuned_model = finetuned_model.train()
    optimizer = torch.optim.AdamW(finetuned_model.parameters(), lr=lr)

    loss_function = nn.CrossEntropyLoss(ignore_index=-1)

    for epoch in range(epochs):
        print(f"Epoch {epoch} of {epochs}")
        for i, batch in enumerate(data_loader):
            masked_helios_sample, label = batch
            label = label.to(device=device)

            masked_helios_sample_dict = masked_helios_sample.as_dict(return_none=False)
            for key, val in masked_helios_sample_dict.items():
                if key == "timestamps":
                    masked_helios_sample_dict[key] = val.to(device=device)
                else:
                    masked_helios_sample_dict[key] = val.to(
                        device=device, dtype=torch.bfloat16
                    )

            masked_helios_sample = MaskedHeliosSample.from_dict(
                masked_helios_sample_dict
            )

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = finetuned_model(
                    masked_helios_sample
                )  # (bs, H, W, logits_per_patch)
                height, width = logits.shape[1], logits.shape[2]
                logits = rearrange(
                    logits,
                    "b h w (c i j) -> b c (h i) (w j)",
                    h=height,
                    w=width,
                    c=task_config.num_classes,
                    i=patch_size,
                    j=patch_size,
                )
                logits = F.interpolate(
                    logits.float(),
                    size=(label.shape[-2], label.shape[-1]),
                    mode="bilinear",
                    align_corners=True,
                )  # (bs, num_classes, H, W)
                loss = loss_function(logits, label)
                print(f"Loss: {loss.item()}")

            loss.backward()
            adjust_learning_rate(
                optimizer=optimizer,
                epoch=epoch + (i / len(data_loader)),
                total_epochs=epochs,
                warmup_epochs=int(epochs * 0.1),
                max_lr=lr,
                min_lr=1.0e-5,
            )
            torch.nn.utils.clip_grad_norm_(finetuned_model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

    return finetuned_model


def evaluate_seg(
    task_config: EvalDatasetConfig,
    data_loader: DataLoader,
    finetuned_model: nn.Module,
    patch_size: int,
    device: torch.device,
) -> float:
    """Evaluate the finetuned model on a segmentation task.

    Args:
        task_config: The config of the task to evaluate on.
        data_loader: The data loader to use for evaluation.
        finetuned_model: The finetuned model to evaluate.
        patch_size: The patch size to use for evaluation.
        device: The device to use for evaluation.

    Returns:
        The mean IoU of the finetuned model.
    """
    finetuned_model = finetuned_model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            masked_helios_sample, label = batch
            label = label.to(device=device)

            masked_helios_sample_dict = masked_helios_sample.as_dict(return_none=False)
            for key, val in masked_helios_sample_dict.items():
                if key == "timestamps":
                    masked_helios_sample_dict[key] = val.to(device=device)
                else:
                    masked_helios_sample_dict[key] = val.to(
                        device=device, dtype=torch.bfloat16
                    )

            masked_helios_sample = MaskedHeliosSample.from_dict(
                masked_helios_sample_dict
            )

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = finetuned_model(
                    masked_helios_sample
                )  # (bs, H, W, logits_per_patch)
                height, width = logits.shape[1], logits.shape[2]
                logits = rearrange(
                    logits,
                    "b h w (c i j) -> b c (h i) (w j)",
                    h=height,
                    w=width,
                    c=task_config.num_classes,
                    i=patch_size,
                    j=patch_size,
                )
                logits = F.interpolate(
                    logits.float(),
                    size=(label.shape[-2], label.shape[-1]),
                    mode="bilinear",
                    align_corners=True,
                )  # (bs, num_classes, H, W)
                preds = torch.argmax(logits, dim=1).cpu()
                all_preds.append(preds)
                all_labels.append(label)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    miou = mean_iou(
        all_preds, all_labels, num_classes=task_config.num_classes, ignore_label=-1
    )

    return miou


def sweep_lr(
    task_name: str,
    checkpoint_path: Path,
    freeze_encoder: bool,
    patch_size: int,
    pooling_type: PoolingType,
    batch_size: int,
    num_workers: int,
    epochs: int,
    num_runs: int,
    device: torch.device,
) -> None:
    """Sweep the learning rate for a given task.

    Args:
        task_name: The name of the task to finetune on.
        checkpoint_path: The path to the checkpoint to use for training.
        freeze_encoder: Whether to freeze the encoder, if True, it will be linear probing.
        patch_size: The patch size to use for training.
        pooling_type: The pooling type to use for training.
        batch_size: The batch size to use for training.
        num_workers: The number of workers to use for training.
        epochs: The number of epochs to train for.
        num_runs: The number of runs to sweep the learning rate over.
        device: The device to use for training.
    """
    task_config = dataset_to_config(task_name)
    task_type = task_config.task_type

    final_scores = {}
    # Set learning rates based on whether we are freezing the encoder
    lrs = LP_LRs if freeze_encoder else FT_LRs
    for _ in range(num_runs):
        for lr in lrs:
            if task_type == TaskType.CLASSIFICATION:
                print(f"Finetuning on {task_name} with lr {lr}")
                val_score = finetune_and_eval_cls(
                    task_name=task_name,
                    checkpoint_path=Path(checkpoint_path),
                    freeze_encoder=freeze_encoder,
                    patch_size=patch_size,
                    pooling_type=pooling_type,
                    lr=lr,
                    epochs=epochs,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    device=device,
                )
            elif task_type == TaskType.SEGMENTATION:
                print(f"Finetuning on {task_name} with lr {lr}")
                val_score = finetune_and_eval_seg(
                    task_name=task_name,
                    checkpoint_path=Path(checkpoint_path),
                    freeze_encoder=freeze_encoder,
                    patch_size=patch_size,
                    pooling_type=pooling_type,
                    lr=lr,
                    epochs=epochs,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    device=device,
                )
            print(f"Val score: {val_score}")
            final_scores[lr] = val_score

    print(
        f"Task: {task_name}, Freeze encoder: {freeze_encoder}, Final scores: {final_scores}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Finetune and evaluate Helios checkpoint on task"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="The name of the task to finetune on",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="The path to the checkpoint to use for training",
    )
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Whether to freeze the encoder, if True, it will be linear probing",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        required=False,
        default=4,
        help="The patch size to use for training",
    )
    parser.add_argument(
        "--pooling_type",
        type=PoolingType,
        required=False,
        default=PoolingType.MEAN,
        help="The pooling type to use for training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=128,
        help="The batch size to use for training",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        required=False,
        default=8,
        help="The number of workers to use for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=False,
        default=50,
        help="The number of epochs to train for",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        required=False,
        default=1,
        help="The number of runs to sweep the learning rate over",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cuda",
        help="The device to use for training",
    )
    args = parser.parse_args()

    sweep_lr(
        task_name=args.task_name,
        checkpoint_path=args.checkpoint_path,
        freeze_encoder=args.freeze_encoder,
        patch_size=args.patch_size,
        pooling_type=args.pooling_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        num_runs=args.num_runs,
        device=torch.device(args.device),
    )

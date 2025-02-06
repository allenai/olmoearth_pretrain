"""Embeddings from models."""

import torch
from torch import nn
from torch.utils.data import DataLoader


def get_embeddings(
    data_loader: DataLoader, model: nn.Module
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get embeddings from model for the data in data_loader."""
    embeddings = []
    labels = []

    model = model.eval()
    with torch.no_grad():
        for helios_sample, label in data_loader:
            with torch.amp.autocast(dtype=torch.bfloat16):
                batch_embeddings = model(helios_sample)  # (bsz, dim)

            embeddings.append(batch_embeddings.to(torch.bfloat16).cpu())
            labels.append(label)

    embeddings = torch.cat(embeddings, dim=0)  # (N, dim)
    labels = torch.cat(labels, dim=0)  # (N)

    return embeddings, labels

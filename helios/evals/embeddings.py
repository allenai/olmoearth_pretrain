import torch
from torch import nn
from torch.utils.data import DataLoader


def get_embeddings(
    data_loader: DataLoader, model: nn.Module, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings = []
    labels = []

    model = model.eval()
    with torch.no_grad():
        for batch in data_loader:
            with torch.amp.autocast(dtype=torch.bfloat16):
                batch_embeddings = model(**batch)  # (bsz, dim)

            embeddings.append(batch_embeddings.to(torch.bfloat16).cpu())
            labels.append(batch_labels)

    embeddings = torch.cat(embeddings, dim=0)  # (N, dim)
    labels = torch.cat(labels, dim=0)  # (N)

    return embeddings, labels

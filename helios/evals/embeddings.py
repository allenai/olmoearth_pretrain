"""Embeddings from models."""

import logging

import torch
from torch.utils.data import DataLoader

from helios.evals.datasets.configs import TaskType
from helios.nn.flexihelios import Encoder, PoolingType, TokensAndMasks
from helios.train.masking import MaskedHeliosSample

logger = logging.getLogger(__name__)


def get_embeddings(
    data_loader: DataLoader,
    task_type: TaskType,
    model: Encoder,
    patch_size: int,
    pooling_type: PoolingType = PoolingType.MAX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get embeddings from model for the data in data_loader."""
    embeddings = []
    labels = []

    model = model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for masked_helios_sample, label in data_loader:
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
                # TODO: Model expects masked helios sample we need to pass empty masks
                # Likely we want to have a flag that checks for eval mode and passes empty masks
                batch_embeddings: TokensAndMasks = model(
                    masked_helios_sample, patch_size=patch_size
                )  # (bsz, dim)
            spatial_pool = True if task_type == TaskType.SEGMENTATION else False
            averaged_embeddings = batch_embeddings.pool_unmasked_tokens(
                pooling_type, spatial_pooling=spatial_pool
            )
            embeddings.append(averaged_embeddings.cpu())
            labels.append(label)

    embeddings = torch.cat(embeddings, dim=0)  # (N, dim)
    labels = torch.cat(labels, dim=0)  # (N)

    return embeddings, labels

"""Embeddings from models."""

import logging

import torch
from torch.utils.data import DataLoader

from helios.nn.flexihelios import Encoder
from helios.train.masking import MaskedHeliosSample

logger = logging.getLogger(__name__)


def get_embeddings(
    data_loader: DataLoader, model: Encoder
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get embeddings from model for the data in data_loader."""
    embeddings = []
    labels = []

    model = model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for helios_sample, label in data_loader:
            masked_helios_sample = MaskedHeliosSample.from_heliossample(
                helios_sample, model.modalities_to_channel_groups_dict
            )
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                # TODO: Hacky patch size
                logger.info(f"Helios sample: {helios_sample}")
                logger.info(f"Patch size: {model.base_patch_size}")
                # TODO: Model expects masked helios sample we need to pass empty masks
                # Likely we want to have a flag that checks for eval mode and passes empty masks
                batch_embeddings = model(
                    masked_helios_sample, patch_size=model.base_patch_size
                )  # (bsz, dim)

            embeddings.append(batch_embeddings.to(torch.bfloat16).cpu())
            labels.append(label)

    embeddings = torch.cat(embeddings, dim=0)  # (N, dim)
    labels = torch.cat(labels, dim=0)  # (N)

    return embeddings, labels

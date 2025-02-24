"""Embeddings from models."""

import logging

import torch
from torch.utils.data import DataLoader

from helios.nn.flexihelios import Encoder
from helios.train.masking import MaskedHeliosSample

logger = logging.getLogger(__name__)


def get_embeddings(
    data_loader: DataLoader, model: Encoder, patch_size: int
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
                    dims_to_reduce = list(range(val.ndim - 1))
                    logger.info(
                        f"{key}: mean: {val.mean(dims_to_reduce)}, std: {val.std(dims_to_reduce)}"
                    )
            masked_helios_sample = MaskedHeliosSample.from_dict(
                masked_helios_sample_dict
            )
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                # TODO: Model expects masked helios sample we need to pass empty masks
                # Likely we want to have a flag that checks for eval mode and passes empty masks
                batch_embeddings = model(
                    masked_helios_sample, patch_size=patch_size
                )  # (bsz, dim)
            embeddings.append(batch_embeddings.average_unmasked_tokens().cpu())
            labels.append(label)

    embeddings = torch.cat(embeddings, dim=0)  # (N, dim)
    labels = torch.cat(labels, dim=0)  # (N)

    return embeddings, labels

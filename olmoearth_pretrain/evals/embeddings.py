"""Embeddings from models."""

import logging

import torch
from torch.utils.data import DataLoader

from olmoearth_pretrain.evals.eval_wrapper import EvalWrapper
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

logger = logging.getLogger(__name__)


def get_embeddings(
    data_loader: DataLoader, model: EvalWrapper
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get embeddings from model for the data in data_loader."""
    embeddings = []
    labels = []
    model.eval()
    device = model.device
    total_samples = len(data_loader)
    with torch.no_grad():
        for i, (masked_olmoearth_sample, label) in enumerate(data_loader):
            masked_olmoearth_sample_dict = masked_olmoearth_sample.as_dict(
                return_none=False
            )
            for key, val in masked_olmoearth_sample_dict.items():
                if key == "timestamps":
                    masked_olmoearth_sample_dict[key] = val.to(device=device)
                else:
                    masked_olmoearth_sample_dict[key] = val.to(
                        device=device,
                    )

            masked_olmoearth_sample = MaskedOlmoEarthSample.from_dict(
                masked_olmoearth_sample_dict
            )
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                batch_embeddings, label = model(
                    masked_olmoearth_sample=masked_olmoearth_sample, labels=label
                )

            embeddings.append(batch_embeddings.cpu())
            labels.append(label)
            logger.info(f"Processed {i} / {total_samples}")

    embeddings = torch.cat(embeddings, dim=0)  # (N, dim)
    labels = torch.cat(labels, dim=0)  # (N)

    return embeddings, labels

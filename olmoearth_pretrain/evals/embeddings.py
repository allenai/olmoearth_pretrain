"""Embeddings from models."""

import logging

import torch
from torch.utils.data import DataLoader

from olmoearth_pretrain.evals.embedding_transforms import quantize_embeddings
from olmoearth_pretrain.evals.eval_wrapper import EvalWrapper
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

logger = logging.getLogger(__name__)


def get_embeddings(
    data_loader: DataLoader,
    model: EvalWrapper,
    is_train: bool = True,
    quantize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get embeddings from model for the data in data_loader.

    Args:
        data_loader: DataLoader for the evaluation dataset.
        model: EvalWrapper-wrapped model to get embeddings from.
        is_train: Whether this is training data (affects some model behaviors).
        quantize: If True, quantize embeddings to int8 for storage efficiency testing.

    Returns:
        Tuple of (embeddings, labels). If quantize=True, embeddings are int8.
    """
    embeddings_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
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
                    masked_olmoearth_sample=masked_olmoearth_sample,
                    labels=label,
                    is_train=is_train,
                )

            embeddings_list.append(batch_embeddings.cpu())
            labels_list.append(label)
            logger.info(f"Processed {i} / {total_samples}")

    embeddings = torch.cat(embeddings_list, dim=0)  # (N, dim)
    labels = torch.cat(labels_list, dim=0)  # (N)

    # Apply quantization if requested
    if quantize:
        logger.info(f"Quantizing embeddings from {embeddings.dtype} to int8")
        embeddings = quantize_embeddings(embeddings)

    return embeddings, labels

"""Embeddings from models."""

import logging

import torch
from torch.utils.data import DataLoader

from olmoearth_pretrain.evals.eval_wrapper import EvalWrapper
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

logger = logging.getLogger(__name__)


def get_embeddings(
    data_loader: DataLoader,
    model: EvalWrapper,
    is_train: bool = True,
    quantize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, float | None]:
    """Get embeddings from model for the data in data_loader.

    Args:
        data_loader: DataLoader for the dataset
        model: Model wrapper for evaluation
        is_train: Whether this is training data
        quantize: If True, quantize embeddings to int8

    Returns:
        Tuple of (embeddings, labels, scale) where:
        - embeddings: Embeddings tensor (int8 if quantize=True, float32 otherwise)
        - labels: Labels tensor
        - scale: Quantization scale factor (None if quantize=False)
    """
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
                    masked_olmoearth_sample=masked_olmoearth_sample,
                    labels=label,
                    is_train=is_train,
                )

            embeddings.append(batch_embeddings.cpu())
            labels.append(label)
            logger.info(f"Processed {i} / {total_samples}")

    embeddings_tensor: torch.Tensor = torch.cat(embeddings, dim=0)  # (N, dim)
    labels = torch.cat(labels, dim=0)  # (N)

    scale: float | None = None
    if quantize:
        # Power-based quantization scheme (similar to AlphaEarth)
        # Constants
        POWER = 2.0
        SCALE = 127.5
        MIN_VALUE = -127.0
        MAX_VALUE = 127.0

        # Quantize: apply square root (pow(1/POWER)) while preserving sign
        # Then scale and clamp to int8 range
        # This matches the AlphaEarth quantization scheme exactly:
        # sat = img.abs().pow(1/POWER).multiply(img.signum())
        # return sat.multiply(SCALE).clamp(MIN_VALUE, MAX_VALUE).int8()
        original_min = embeddings_tensor.min().item()
        original_max = embeddings_tensor.max().item()
        sat = embeddings_tensor.abs().pow(1 / POWER) * embeddings_tensor.sign()
        embeddings_tensor = (
            (sat * SCALE).clamp(MIN_VALUE, MAX_VALUE).round().to(torch.int8)
        )
        scale = SCALE  # Store scale for dequantization
        quantized_min = embeddings_tensor.min().item()
        quantized_max = embeddings_tensor.max().item()
        logger.info(
            f"Quantized embeddings to int8 using power-based scheme (POWER={POWER}, SCALE={SCALE})"
        )
        logger.info(
            f"Original embedding range: [{original_min:.6f}, {original_max:.6f}], "
            f"Quantized range: [{quantized_min}, {quantized_max}]"
        )

    return embeddings_tensor, labels, scale

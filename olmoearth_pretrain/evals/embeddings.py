"""Embeddings from models."""

import logging

import torch
from torch.utils.data import DataLoader

from olmoearth_pretrain.evals.eval_wrapper import EvalWrapper
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

logger = logging.getLogger(__name__)

# Quantization constants (matching AlphaEarth)
QUANTIZE_POWER = 2.0
QUANTIZE_SCALE = 127.5


def quantize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """Quantize float embeddings to int8 using power-based scheme.

    This applies a sqrt transform before scaling to preserve information
    for non-uniform embedding distributions.

    Args:
        embeddings: Float tensor of shape (N, dim) or (N, H, W, dim)

    Returns:
        Int8 tensor of same shape
    """
    # Apply sqrt, preserve sign: sat = |x|^(1/power) * sign(x)
    sat = embeddings.abs().pow(1.0 / QUANTIZE_POWER) * embeddings.sign()
    # Scale to int8 range and quantize
    quantized = (sat * QUANTIZE_SCALE).clamp(-127, 127).round().to(torch.int8)
    return quantized


def dequantize_embeddings(quantized: torch.Tensor) -> torch.Tensor:
    """Dequantize int8 embeddings back to float32.

    This reverses the power-based quantization scheme.

    Args:
        quantized: Int8 tensor of shape (N, dim) or (N, H, W, dim)

    Returns:
        Float32 tensor of same shape
    """
    # Rescale from int8 range
    rescaled = quantized.float() / QUANTIZE_SCALE
    # Apply square, preserve sign: x = |rescaled|^power * sign(rescaled)
    dequantized = rescaled.abs().pow(QUANTIZE_POWER) * rescaled.sign()
    return dequantized


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

    embeddings = torch.cat(embeddings, dim=0)  # (N, dim)
    labels = torch.cat(labels, dim=0)  # (N)

    # Apply quantization if requested
    if quantize:
        logger.info(f"Quantizing embeddings from {embeddings.dtype} to int8")
        embeddings = quantize_embeddings(embeddings)

    return embeddings, labels

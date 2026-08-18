"""Embeddings from models."""

import logging
from collections.abc import Callable, Iterable

import torch

from olmoearth_pretrain.evals.embedding_transforms import (
    quantize_embeddings,
    quantize_embeddings_percentile,
)
from olmoearth_pretrain.evals.eval_wrapper import EvalWrapper
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

logger = logging.getLogger(__name__)


def get_embeddings(
    data_loader: Iterable[tuple[MaskedOlmoEarthSample, torch.Tensor]],
    model: EvalWrapper,
    is_train: bool = True,
    quantize: bool = False,
    quantize_bits: int | None = None,
    quantile_config: dict | None = None,
    sample_transform: Callable[[MaskedOlmoEarthSample], MaskedOlmoEarthSample]
    | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get embeddings from model for the data in data_loader.

    Args:
        data_loader: Anything yielding (sample, label) batches for the evaluation
            dataset -- normally a DataLoader, but a materialized list of batches
            when the same split is embedded repeatedly (see band_sensitivity).
        model: EvalWrapper-wrapped model to get embeddings from.
        is_train: Whether this is training data (affects some model behaviors).
        quantize: If True, quantize embeddings to int8 for storage efficiency testing.
            Uses the legacy power-based scheme if quantize_bits is None.
        quantize_bits: If set (1, 2, 4, or 8), use percentile-based quantization
            with the specified number of bits. Requires quantile_config.
        quantile_config: Dictionary containing precomputed quantile boundaries
            for percentile-based quantization. Required if quantize_bits is set.
        sample_transform: Optional perturbation applied to each batch after it is
            moved to the device and before the model sees it, for ablation
            diagnostics such as per-band occlusion. Applied on-device so a sweep
            over perturbations reuses one pass of data loading and normalization.

    Returns:
        Tuple of (embeddings, labels). If quantize=True, embeddings are int8.
    """
    embeddings_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    model.eval()
    device = model.device
    with torch.no_grad():
        for i, (masked_olmoearth_sample, label) in enumerate(data_loader):
            masked_olmoearth_sample_dict = masked_olmoearth_sample.as_dict()
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
            if sample_transform is not None:
                masked_olmoearth_sample = sample_transform(masked_olmoearth_sample)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                batch_embeddings, label = model(
                    masked_olmoearth_sample=masked_olmoearth_sample,
                    labels=label,
                    is_train=is_train,
                )

            embeddings_list.append(batch_embeddings.cpu())
            labels_list.append(label)
            logger.info("Processed batch %d", i)

    embeddings = torch.cat(embeddings_list, dim=0)  # (N, dim)
    labels = torch.cat(labels_list, dim=0)  # (N)

    # Apply quantization if requested
    if quantize:
        if quantize_bits is not None and quantile_config is not None:
            # Percentile-based quantization
            key = f"{quantize_bits}bit"
            if key not in quantile_config:
                raise ValueError(
                    f"Quantile config missing '{key}' key for {quantize_bits}-bit quantization"
                )
            logger.info(
                f"Quantizing embeddings to {quantize_bits}-bit using percentile boundaries"
            )
            quantiles = quantile_config[key]["quantiles"]
            embeddings = quantize_embeddings_percentile(
                embeddings, quantiles, quantize_bits
            )
        else:
            # Legacy power-based int8 quantization
            logger.info(f"Quantizing embeddings from {embeddings.dtype} to int8")
            embeddings = quantize_embeddings(embeddings)

    return embeddings, labels

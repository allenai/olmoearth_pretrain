"""Embeddings from models."""

import logging
from collections.abc import Callable, Iterable

import torch

from olmoearth_pretrain.evals.embedding_diagnostics import (
    compute_pipeline_diagnostics,
    flatten_rows,
    sample_row_indices,
)
from olmoearth_pretrain.evals.embedding_transforms import (
    EmbeddingNormalizer,
    QuantizationScheme,
    dequantize_embeddings,
    dequantize_embeddings_percentile,
    quantize_embeddings,
    quantize_embeddings_percentile,
    roundtrip_embeddings_tessera,
)
from olmoearth_pretrain.evals.eval_wrapper import EvalWrapper
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

logger = logging.getLogger(__name__)


def normalize_and_quantize(
    embeddings: torch.Tensor,
    normalizer: EmbeddingNormalizer | None = None,
    quantize: bool = False,
    quantize_bits: int | None = None,
    quantile_config: dict | None = None,
    diagnostics_out: dict[str, float] | None = None,
    quantization_scheme: QuantizationScheme = QuantizationScheme.AEF_POWER,
) -> torch.Tensor:
    """Normalize then (optionally) quantize one split's embeddings.

    Normalization runs BEFORE quantization: the int8 schemes assume a value
    range (see ``QUANTIZE_CLIP_THRESHOLD``), so normalizing afterwards would
    rescale damage that has already been done.

    Args:
        embeddings: Float embeddings ``[N, ..., D]``.
        normalizer: Fitted (or stateless) normalization; None leaves the
            embeddings exactly as the model emitted them.
        quantize: If True, quantize to int8 for storage-efficiency parity with
            the precomputed embedding products.
        quantize_bits: If set (1, 2, 4, or 8), use percentile-based
            quantization with ``quantile_config`` instead of the power scheme.
        quantile_config: Precomputed quantile boundaries for the above.
        diagnostics_out: If provided, filled with geometry diagnostics for each
            stage of the pipeline (a bounded row subsample, so the cost does not
            scale with the split size).
        quantization_scheme: Which int8 scheme to apply. ``TESSERA_PER_VECTOR``
            returns **float32** already round-tripped, because its per-vector
            scales are needed to reconstruct; the other schemes return int8
            codes for a caller-side dequantize.

    Returns:
        The transformed embeddings: int8 under the code-returning schemes,
        float32 under ``TESSERA_PER_VECTOR``, and float if quantization is off.
    """
    raw = embeddings if diagnostics_out is not None else None

    if normalizer is not None:
        embeddings = normalizer(embeddings)
    normalized = embeddings if raw is not None and normalizer is not None else None

    tessera_scheme = quantization_scheme == QuantizationScheme.TESSERA_PER_VECTOR
    if quantize:
        if tessera_scheme:
            logger.info(
                "Quantizing embeddings through Tessera's int8 scheme "
                "(linear, per-vector scale); returns float32"
            )
            embeddings = roundtrip_embeddings_tessera(embeddings)
        elif quantize_bits is not None and quantile_config is not None:
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
            logger.info(f"Quantizing embeddings from {embeddings.dtype} to int8")
            embeddings = quantize_embeddings(embeddings)

    if diagnostics_out is None:
        return embeddings

    assert raw is not None
    # Subsample once, here, and hand the diagnostics the small views: the
    # round-trip view has to be dequantized, and dequantizing the whole split
    # would duplicate the probe's tensor in float32 for no benefit.
    raw_rows = flatten_rows(raw)
    idx = sample_row_indices(raw_rows.shape[0])
    if idx is not None:
        raw_rows = raw_rows[idx]
    norm_rows = None
    if normalized is not None:
        norm_rows = flatten_rows(normalized)
        if idx is not None:
            norm_rows = norm_rows[idx]
    round_tripped = None
    if quantize:
        quantized_rows = embeddings.reshape(-1, embeddings.shape[-1])
        if idx is not None:
            quantized_rows = quantized_rows[idx]
        if tessera_scheme:
            # Already float32 and already round-tripped.
            round_tripped = quantized_rows
        elif quantize_bits is not None and quantile_config is not None:
            midpoints = quantile_config[f"{quantize_bits}bit"]["midpoints"]
            round_tripped = dequantize_embeddings_percentile(quantized_rows, midpoints)
        else:
            round_tripped = dequantize_embeddings(quantized_rows)
    diagnostics_out.update(
        compute_pipeline_diagnostics(
            raw=raw_rows, normalized=norm_rows, round_tripped=round_tripped
        )
    )
    return embeddings


def get_embeddings(
    data_loader: Iterable[tuple[MaskedOlmoEarthSample, torch.Tensor]],
    model: EvalWrapper,
    is_train: bool = True,
    quantize: bool = False,
    quantize_bits: int | None = None,
    quantile_config: dict | None = None,
    sample_transform: Callable[[MaskedOlmoEarthSample], MaskedOlmoEarthSample]
    | None = None,
    normalizer: EmbeddingNormalizer | None = None,
    diagnostics_out: dict[str, float] | None = None,
    quantization_scheme: QuantizationScheme = QuantizationScheme.AEF_POWER,
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
        normalizer: Applied to the extracted embeddings before quantization.
            Fitted modes must already be fitted (on the train split, or on
            precomputed constants); None keeps the model's raw output.
        diagnostics_out: If provided, filled with per-stage geometry diagnostics
            for this split (see ``normalize_and_quantize``).
        quantization_scheme: Which int8 scheme to apply when quantizing.

    Returns:
        Tuple of (embeddings, labels). If quantize=True, embeddings are int8 —
        except under ``TESSERA_PER_VECTOR``, which returns round-tripped float32
        (see ``normalize_and_quantize``).
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

    embeddings = normalize_and_quantize(
        embeddings,
        normalizer=normalizer,
        quantize=quantize,
        quantize_bits=quantize_bits,
        quantile_config=quantile_config,
        diagnostics_out=diagnostics_out,
        quantization_scheme=quantization_scheme,
    )
    return embeddings, labels

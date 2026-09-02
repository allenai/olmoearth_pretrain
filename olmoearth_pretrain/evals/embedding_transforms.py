"""Post-extraction transforms for embeddings (normalization, quantization, dim reduction)."""

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import h5py
import torch
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

# === Quantization ===
# Constants matching AlphaEarth's scheme
QUANTIZE_POWER = 2.0
QUANTIZE_SCALE = 127.5

# The power scheme saturates where |x|^(1/power) * scale exceeds the int8 range:
# |x| > (127 / scale)^power. AEF's embeddings are 64-d unit-L2 vectors, so their
# coordinates sit far below this; anything coming off a LayerNorm (per-coordinate
# std ~ 1) clips a large fraction of its coordinates instead. Diagnostics report
# the clipped fraction so the mismatch is visible rather than silent.
QUANTIZE_CLIP_THRESHOLD = (127.0 / QUANTIZE_SCALE) ** QUANTIZE_POWER


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


# === Tessera's quantization scheme ===
# Verified against the shipped client, geotessera/store.py:
#
#     def dequantise(emb_int8, scales):   # (B,H,W) + (H,W) -> (H,W,B) float32
#         f32 = emb_int8.astype(np.float32) * scales[np.newaxis, :, :]
#
# i.e. LINEAR with one float32 scale per PIXEL (broadcast over all 128 bands),
# published as `grid_{lon}_{lat}.npy` + `_scales.npy`. Two ways this differs
# from the AEF power scheme above, both of which matter for a LayerNorm-geometry
# embedding: the scale is fitted per vector instead of being the global constant
# QUANTIZE_SCALE, and there is no companding curve. Together they make it
# **clip-free by construction** -- the largest coordinate of every vector lands
# exactly on +/-127 -- where the power scheme saturates everything beyond
# QUANTIZE_CLIP_THRESHOLD. Measured on d128 register embeddings: cos(orig,
# round-trip) 0.99998 here vs 0.95058 under the power scheme.
#
# Only their DECODER is in the client, so the encoder below is the natural
# inverse (scale = max|x| / 127). Any per-vector scale is clip-free; the exact
# choice moves the step size by a few percent, not the conclusion.
TESSERA_INT8_MAX = 127.0


def quantize_embeddings_tessera(
    embeddings: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize to int8 with Tessera's per-vector scale.

    Args:
        embeddings: Float tensor of shape (N, dim) or (N, H, W, dim).

    Returns:
        ``(quantized, scales)``: int8 codes of the input shape, and the float32
        scales with the last dim kept as 1 so they broadcast on dequantization.
    """
    scales = embeddings.abs().amax(dim=-1, keepdim=True) / TESSERA_INT8_MAX
    # An all-zero vector has no scale; 1.0 leaves it at zero and keeps the
    # round-trip finite (their product marks such pixels non-finite instead).
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    quantized = (
        torch.round(embeddings / scales)
        .clamp(-TESSERA_INT8_MAX, TESSERA_INT8_MAX)
        .to(torch.int8)
    )
    return quantized, scales


def dequantize_embeddings_tessera(
    quantized: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """Dequantize Tessera-scheme int8 codes: ``codes * scales``."""
    return quantized.float() * scales


def roundtrip_embeddings_tessera(embeddings: torch.Tensor) -> torch.Tensor:
    """Push embeddings through Tessera's int8 scheme, returning float32.

    Fused because the per-vector scales are needed to reconstruct, and unlike
    the power scheme the int8 codes alone are not a faithful stand-in for the
    stored product (they have had each vector's magnitude divided out). The
    result carries exactly the information a consumer of their published
    ``int8 + _scales.npy`` pair gets after ``geotessera``'s ``dequantise``.
    """
    quantized, scales = quantize_embeddings_tessera(embeddings)
    return dequantize_embeddings_tessera(quantized, scales)


class QuantizationScheme(StrEnum):
    """Which int8 scheme ``quantize_embeddings=True`` applies.

    Per-product, because the point of the embedding-eval convention is to score
    each arm at the precision it actually ships.
    """

    # AlphaEarth's published scheme (POWER/SCALE above). The default, and the
    # right choice for AEF-geometry (unit-L2) embeddings.
    AEF_POWER = "aef_power"
    # Tessera's scheme: linear, per-vector scale, clip-free. Used for the
    # tessera_v2 arm, which we bake ourselves in float32.
    TESSERA_PER_VECTOR = "tessera_per_vector"


# === Normalization ===


class EmbeddingNormalization(StrEnum):
    """How to normalize extracted embeddings before the int8 round-trip / probe.

    Nothing in pretraining pins the geometry of an embedding head's output: the
    register bottleneck and its distilled student both end in a LayerNorm
    (per-token scale), but the distillation losses are invariant to any
    invertible linear map of the student space, so per-dimension scale, the
    shared mean component, and the absolute magnitude are all free. Three
    consumers care:

    - ``quantize_embeddings`` assumes AEF's convention (coordinates well inside
      [-1, 1]); a LayerNorm-scale embedding saturates instead (see
      ``QUANTIZE_CLIP_THRESHOLD``).
    - KNN scores cosine similarity without centering, so a large shared mean
      component compresses every pair toward 1 (hubness).
    - The segmentation linear probe has no BatchNorm in front (the
      classification one does), so it sees raw per-dimension scale.

    Fitted modes take their statistics from the TRAIN split only and apply them
    to val/test.

    Combining a variance-scaling mode (``ZSCORE``) with the legacy int8
    round-trip is self-defeating -- it lands the coordinates at std 1, right
    where the power scheme clips. Use ``L2``/``CENTER_L2`` on quantized tasks
    and ``ZSCORE`` to isolate probe conditioning with quantization off.
    """

    # Current behavior: embeddings are consumed exactly as the model emits them.
    NONE = "none"
    # Per-embedding L2 normalization (AEF's convention; stateless).
    L2 = "l2"
    # Subtract the train-split mean embedding (kills the shared offset).
    CENTER = "center"
    # Center, then L2 normalize -- the pairing for quantized embedding tasks.
    CENTER_L2 = "center_l2"
    # Center and divide by the train-split per-dimension std (isotropic dims).
    ZSCORE = "zscore"


# Modes whose statistics must be fitted on the train split before use.
FITTED_NORMALIZATIONS = frozenset(
    {
        EmbeddingNormalization.CENTER,
        EmbeddingNormalization.CENTER_L2,
        EmbeddingNormalization.ZSCORE,
    }
)

# Floor on the per-dimension std, so a dead dimension cannot blow up under ZSCORE.
_STD_EPS = 1e-6


@dataclass
class EmbeddingNormalizer:
    """A fitted (or stateless) embedding normalization, applied on the last dim.

    Accepts ``[N, D]`` and spatial ``[N, ..., D]`` embeddings alike; statistics
    are pooled over every leading dimension.
    """

    mode: EmbeddingNormalization
    mean: torch.Tensor | None = None
    std: torch.Tensor | None = None

    @classmethod
    def fit(
        cls, mode: EmbeddingNormalization, embeddings: torch.Tensor
    ) -> "EmbeddingNormalizer":
        """Fit on ``embeddings`` (one split); stateless modes ignore them.

        Statistics fitted per eval dataset answer "is the geometry the problem?"
        but are not themselves deployable -- a global embedding run has no
        per-dataset train split to fit on. Fit once on a representative sample,
        ``save`` the constants, and pass them back through ``load`` for the
        deployable form (see ``scripts/tools/fit_embedding_norm_stats.py``).
        """
        if mode not in FITTED_NORMALIZATIONS:
            return cls(mode=mode)
        rows = embeddings.reshape(-1, embeddings.shape[-1]).float()
        mean = rows.mean(dim=0)
        std = None
        if mode == EmbeddingNormalization.ZSCORE:
            std = rows.std(dim=0).clamp_min(_STD_EPS)
        return cls(mode=mode, mean=mean, std=std)

    def __call__(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply the normalization, leaving NONE (and dtype) untouched."""
        if self.mode == EmbeddingNormalization.NONE:
            return embeddings
        out = embeddings.float()
        if self.mode in (
            EmbeddingNormalization.CENTER,
            EmbeddingNormalization.CENTER_L2,
            EmbeddingNormalization.ZSCORE,
        ):
            if self.mean is None:
                raise ValueError(
                    f"{self.mode} requires a fitted mean; call fit() first"
                )
            out = out - self.mean.to(out.device)
        if self.mode == EmbeddingNormalization.ZSCORE:
            if self.std is None:
                raise ValueError(f"{self.mode} requires a fitted std; call fit() first")
            out = out / self.std.to(out.device)
        if self.mode in (EmbeddingNormalization.L2, EmbeddingNormalization.CENTER_L2):
            out = torch.nn.functional.normalize(out, dim=-1)
        return out

    def save(self, path: str) -> None:
        """Persist the fitted constants so every dataset can share one set."""
        torch.save(
            {"mode": str(self.mode), "mean": self.mean, "std": self.std},
            path,
        )
        logger.info(f"Wrote {self.mode} embedding norm stats to {path}")

    @classmethod
    def load(cls, path: str, mode: EmbeddingNormalization) -> "EmbeddingNormalizer":
        """Load constants written by ``save``, checking they match ``mode``.

        The stats are per-model, not per-dataset: a normalizer fitted for one
        checkpoint's embedding space is meaningless for another's, so the
        caller is responsible for pointing at the right file.
        """
        state = torch.load(path, map_location="cpu", weights_only=True)
        saved_mode = EmbeddingNormalization(state["mode"])
        if saved_mode != mode:
            raise ValueError(
                f"Embedding norm stats at {path} were fitted for {saved_mode}, "
                f"but the task requests {mode}"
            )
        return cls(mode=mode, mean=state["mean"], std=state["std"])


# === Percentile-based Quantization ===


def load_quantile_config(path: str) -> dict[str, Any]:
    """Load quantile boundaries and midpoints from HDF5 file.

    The HDF5 file is generated by scripts/tools/compute_embedding_quantiles.py.

    The quantiles are used for quantization, while midpoints are used for
    de-quantization.

    Args:
        path: Path to quantiles.h5 file

    Returns:
        Dictionary with keys like "8bit", "4bit", "2bit", "1bit", each containing:
            - "quantiles": torch.Tensor of shape (dim, num_buckets+1)
            - "midpoints": torch.Tensor of shape (dim, num_buckets)
    """
    config: dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        for bits in [8, 4, 2, 1]:
            key = f"{bits}bit"
            if key in f:
                config[key] = {
                    "quantiles": torch.from_numpy(f[key]["quantiles"][:]),
                    "midpoints": torch.from_numpy(f[key]["midpoints"][:]),
                }
        if "dim" in f:
            config["dim"] = int(f["dim"][()])
    return config


def quantize_embeddings_percentile(
    embeddings: torch.Tensor,
    quantiles: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    """Quantize embeddings using precomputed percentile boundaries.

    For each dimension, finds which bucket each value falls into based on
    the precomputed quantile boundaries.

    Args:
        embeddings: Float tensor of shape (N, dim) or (N, H, W, dim)
        quantiles: Boundary values of shape (dim, num_buckets+1)
        bits: Number of bits (1, 2, 4, or 8)

    Returns:
        Int8 tensor of same shape with values in [0, 2^bits - 1]
    """
    num_buckets = 2**bits
    original_shape = embeddings.shape
    dim = original_shape[-1]

    # Flatten to (N_total, dim)
    flat = embeddings.reshape(-1, dim)

    # Move quantiles to same device
    quantiles = quantiles.to(embeddings.device)

    # Vectorized searchsorted: transpose to (dim, N_total) for batched search
    # quantiles: (dim, num_buckets+1), flat.T: (dim, N_total)
    # searchsorted returns index where value would be inserted
    bucket_idx = torch.searchsorted(quantiles, flat.T) - 1  # (dim, N_total)
    bucket_idx = bucket_idx.clamp(0, num_buckets - 1)
    quantized = bucket_idx.T.to(torch.int8)  # (N_total, dim)

    return quantized.reshape(original_shape)


def dequantize_embeddings_percentile(
    quantized: torch.Tensor,
    midpoints: torch.Tensor,
) -> torch.Tensor:
    """Dequantize embeddings using precomputed midpoint values.

    Maps each bucket index back to its corresponding midpoint value
    (the value at the center percentile of that bucket).

    Args:
        quantized: Int8 tensor of shape (N, dim) or (N, H, W, dim)
            with values in [0, 2^bits - 1]
        midpoints: Dequantization values of shape (dim, num_buckets)

    Returns:
        Float32 tensor of same shape
    """
    original_shape = quantized.shape
    dim = original_shape[-1]

    # Flatten to (N_total, dim)
    # Convert to uint8 first to handle int8 wrap-around (128-255 stored as -128 to -1)
    flat = quantized.reshape(-1, dim).to(torch.uint8).long()
    n_total = flat.shape[0]

    # Move midpoints to same device
    midpoints = midpoints.to(quantized.device)

    # Vectorized lookup using advanced indexing
    # d_indices: (N_total, dim) where each row is [0, 1, 2, ..., dim-1]
    d_indices = (
        torch.arange(dim, device=quantized.device).unsqueeze(0).expand(n_total, -1)
    )
    dequantized = midpoints[d_indices, flat]  # (N_total, dim)

    return dequantized.reshape(original_shape)


# === Dimensionality Reduction ===


def reduce_embedding_dim(
    train_embeddings: torch.Tensor,
    val_embeddings: torch.Tensor,
    test_embeddings: torch.Tensor | None,
    target_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, float]:
    """Reduce embedding dimensionality via PCA.

    Fits PCA on train embeddings and applies the same transform to val/test.
    Handles spatial dimensions (N, H, W, C) by flattening before PCA and
    reshaping after.

    Args:
        train_embeddings: Training embeddings, shape (N, dim) or (N, H, W, dim)
        val_embeddings: Validation embeddings, same shape structure as train
        test_embeddings: Test embeddings (optional), same shape structure as train
        target_dim: Target dimensionality after PCA

    Returns:
        Tuple of (train_reduced, val_reduced, test_reduced, variance_retained)
        where variance_retained is the sum of explained variance ratios.
    """
    original_dim = train_embeddings.shape[-1]
    train_shape = train_embeddings.shape
    val_shape = val_embeddings.shape
    test_shape = test_embeddings.shape if test_embeddings is not None else None

    # Flatten spatial dimensions if present (for segmentation tasks)
    if len(train_shape) > 2:
        # Shape is (N, H, W, C) or similar - flatten to (N*H*W, C)
        train_flat = train_embeddings.reshape(-1, original_dim)
        val_flat = val_embeddings.reshape(-1, original_dim)
        test_flat = (
            test_embeddings.reshape(-1, original_dim)
            if test_embeddings is not None
            else None
        )
    else:
        train_flat = train_embeddings
        val_flat = val_embeddings
        test_flat = test_embeddings

    # Fit PCA on train embeddings
    pca = PCA(n_components=target_dim)
    train_reduced = pca.fit_transform(train_flat.cpu().numpy())
    val_reduced = pca.transform(val_flat.cpu().numpy())
    test_reduced = (
        pca.transform(test_flat.cpu().numpy()) if test_flat is not None else None
    )

    variance_retained = float(sum(pca.explained_variance_ratio_))

    # Convert back to tensors and reshape if needed
    device = train_embeddings.device
    dtype = train_embeddings.dtype

    if len(train_shape) > 2:
        new_train_shape = train_shape[:-1] + (target_dim,)
        new_val_shape = val_shape[:-1] + (target_dim,)
        train_out = (
            torch.from_numpy(train_reduced)
            .to(device=device, dtype=dtype)
            .reshape(new_train_shape)
        )
        val_out = (
            torch.from_numpy(val_reduced)
            .to(device=device, dtype=dtype)
            .reshape(new_val_shape)
        )
        if test_reduced is not None and test_shape is not None:
            new_test_shape = test_shape[:-1] + (target_dim,)
            test_out = (
                torch.from_numpy(test_reduced)
                .to(device=device, dtype=dtype)
                .reshape(new_test_shape)
            )
        else:
            test_out = None
    else:
        train_out = torch.from_numpy(train_reduced).to(device=device, dtype=dtype)
        val_out = torch.from_numpy(val_reduced).to(device=device, dtype=dtype)
        test_out = (
            torch.from_numpy(test_reduced).to(device=device, dtype=dtype)
            if test_reduced is not None
            else None
        )

    return train_out, val_out, test_out, variance_retained

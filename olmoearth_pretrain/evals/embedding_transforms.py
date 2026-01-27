"""Post-extraction transforms for embeddings (quantization, dim reduction)."""

import torch
from sklearn.decomposition import PCA

# === Quantization ===
# Constants matching AlphaEarth's scheme
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

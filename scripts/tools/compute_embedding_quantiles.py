"""Compute per-band quantile boundaries for percentile-based embedding quantization.

This script extracts embeddings from a subset of pretraining samples and computes
percentile boundaries for 8-bit, 4-bit, 2-bit, and 1-bit quantization.

Usage:
    python scripts/tools/compute_embedding_quantiles.py \
        --checkpoint_path=/path/to/checkpoint \
        --h5py_dir=/path/to/pretrain/h5py \
        --output=quantiles.h5 \
        --num_samples=10000
"""

import argparse
import json
import logging

import h5py
import numpy as np
import torch
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from tqdm import tqdm
from upath import UPath

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataset import GetItemArgs, OlmoEarthDataset
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.nn.flexi_vit import PoolingType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_embeddings(
    model: torch.nn.Module,
    dataset: OlmoEarthDataset,
    num_samples: int,
    batch_size: int,
    device: torch.device,
    patch_size: int = 4,
    pooling_type: PoolingType = PoolingType.MEAN,
) -> torch.Tensor:
    """Extract embeddings from a subset of pretrain samples.

    Args:
        model: The pretrained model (encoder)
        dataset: OlmoEarthDataset for pretrain data
        num_samples: Number of samples to extract embeddings from
        batch_size: Batch size for extraction
        device: Device to run model on
        patch_size: Patch size for the model
        pooling_type: Pooling type for aggregating tokens (MEAN or MAX)

    Returns:
        Tensor of shape (num_samples, embedding_dim) with pooled embeddings
    """
    model.eval()
    embeddings_list = []

    # Limit to available samples
    num_samples = min(num_samples, len(dataset))
    logger.info(f"Extracting embeddings from {num_samples} samples")

    # Sample random indices
    indices = np.random.choice(len(dataset), size=num_samples, replace=False)

    # Create a simple dataloader by iterating through indices
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="Extracting embeddings"):
            batch_indices = indices[i : i + batch_size]
            batch_embeddings = []

            for idx in batch_indices:
                # Get sample from dataset with random spatial size and token budget
                sampled_hw_p = np.random.randint(8, 17)
                token_budget = np.random.randint(1000, 3000)
                args = GetItemArgs(
                    idx=idx,
                    patch_size=patch_size,
                    sampled_hw_p=sampled_hw_p,
                    token_budget=token_budget,
                )
                _, sample = dataset[args]

                # Convert to MaskedOlmoEarthSample
                sample_dict = sample.as_dict(ignore_nones=True)

                # Create masks for all modalities
                masked_dict = {}
                for key, val in sample_dict.items():
                    if key == "timestamps":
                        masked_dict[key] = torch.from_numpy(val).unsqueeze(0).to(device)
                        continue

                    # Add batch dimension and move to device
                    val_tensor = torch.from_numpy(val).unsqueeze(0).float().to(device)
                    masked_dict[key] = val_tensor

                    # Create mask (all visible)
                    # Mask shape depends on modality - use num_band_sets dimension
                    modality_spec = Modality.get(key)
                    if modality_spec.is_spatial:
                        # Shape: (B, H, W, T, S) where S = num_band_sets
                        if len(val.shape) == 4:  # (H, W, T, C)
                            mask_shape = (
                                1,
                                val.shape[0],
                                val.shape[1],
                                val.shape[2],
                                modality_spec.num_band_sets,
                            )
                        else:
                            mask_shape = (
                                1,
                                *val.shape[:-1],
                                modality_spec.num_band_sets,
                            )
                    else:
                        # Non-spatial modalities
                        mask_shape = (1, *val.shape[:-1], modality_spec.num_band_sets)

                    mask = (
                        torch.ones(mask_shape, device=device)
                        * MaskValue.ONLINE_ENCODER.value
                    )
                    masked_dict[f"{key}_mask"] = mask

                masked_sample = MaskedOlmoEarthSample(**masked_dict)

                # Run through encoder
                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                    output = model.encoder(
                        masked_sample, fast_pass=True, patch_size=patch_size
                    )

                # Get tokens and pool them using the same method as evaluation
                tokens_and_masks = output["tokens_and_masks"]

                # Use pool_unmasked_tokens with spatial_pooling=True to keep H/W dims
                # This pools over T (time) and S (band sets), then aggregates across modalities
                # Result shape: (B, H', W', D)
                pooled_embedding = tokens_and_masks.pool_unmasked_tokens(
                    pooling_type=pooling_type,
                    spatial_pooling=True,
                    concat_features=False,
                )

                # Flatten spatial dims to get all pixel embeddings -> (B*H'*W', D)
                b, h, w, d = pooled_embedding.shape
                pixel_embeddings = pooled_embedding.reshape(b * h * w, d)
                batch_embeddings.append(pixel_embeddings.float().cpu())

            if batch_embeddings:
                embeddings_list.append(torch.cat(batch_embeddings, dim=0))

    embeddings = torch.cat(embeddings_list, dim=0)
    logger.info(f"Extracted embeddings shape: {embeddings.shape}")
    return embeddings


def compute_quantiles(
    embeddings: torch.Tensor,
) -> dict[str, dict[str, np.ndarray]]:
    """Compute quantile boundaries for different bit levels.

    Args:
        embeddings: Tensor of shape (N, dim) with embedding vectors

    Returns:
        Dictionary with keys "8bit", "4bit", "2bit", "1bit", each containing:
            - "quantiles": array of shape (dim, num_buckets+1) with boundary values
            - "midpoints": array of shape (dim, num_buckets) with dequantization values
    """
    embeddings_np = embeddings.numpy()

    result = {}

    for bits in [8, 4, 2, 1]:
        num_buckets = 2**bits
        key = f"{bits}bit"

        logger.info(f"Computing {bits}-bit quantiles ({num_buckets} buckets)...")

        # Compute boundary percentiles: 0, 1/num_buckets, 2/num_buckets, ..., 1
        boundary_percentiles = np.linspace(0, 100, num_buckets + 1)
        quantiles = np.percentile(embeddings_np, boundary_percentiles, axis=0).T
        # Shape: (dim, num_buckets+1)

        # Compute midpoint percentiles for dequantization
        # For bucket i, midpoint is at (i + 0.5) / num_buckets
        midpoint_percentiles = np.array(
            [(i + 0.5) / num_buckets * 100 for i in range(num_buckets)]
        )
        midpoints = np.percentile(embeddings_np, midpoint_percentiles, axis=0).T
        # Shape: (dim, num_buckets)

        result[key] = {
            "quantiles": quantiles.astype(np.float32),
            "midpoints": midpoints.astype(np.float32),
        }

        logger.info(f"  Quantiles shape: {quantiles.shape}")
        logger.info(f"  Midpoints shape: {midpoints.shape}")

    return result


def save_quantiles(quantiles: dict, output_path: str, dim: int) -> None:
    """Save quantile config to HDF5 file.

    Args:
        quantiles: Dictionary with quantile data for each bit level
        output_path: Path to output HDF5 file
        dim: Embedding dimension
    """
    logger.info(f"Saving quantiles to {output_path}")

    with h5py.File(output_path, "w") as f:
        f.create_dataset("dim", data=dim)

        for bits_key, data in quantiles.items():
            grp = f.create_group(bits_key)
            grp.create_dataset("quantiles", data=data["quantiles"])
            grp.create_dataset("midpoints", data=data["midpoints"])

    logger.info("Done saving quantiles")


def main() -> None:
    """Compute and save percentile-based quantile boundaries for embedding quantization."""
    parser = argparse.ArgumentParser(
        description="Compute quantile boundaries for percentile-based embedding quantization"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--h5py_dir",
        type=str,
        required=True,
        help="Path to pretrain h5py directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="quantiles.h5",
        help="Output path for quantiles HDF5 file",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of samples to extract embeddings from",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for embedding extraction (use 1 for simplicity)",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=4,
        help="Patch size for the model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--training_modalities",
        type=str,
        nargs="+",
        default=["sentinel2_l2a"],
        help="Training modalities to use from the dataset",
    )
    parser.add_argument(
        "--pooling_type",
        type=str,
        choices=["mean", "max"],
        default="mean",
        help="Pooling type for aggregating tokens (default: mean)",
    )

    args = parser.parse_args()
    pooling_type = PoolingType(args.pooling_type)

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model from distributed checkpoint
    logger.info(f"Loading model from {args.checkpoint_path}")
    checkpoint_path = UPath(args.checkpoint_path)
    with (checkpoint_path / "config.json").open() as f:
        config_dict = json.load(f)
        model_config = Config.from_dict(config_dict["model"])
    model = model_config.build()
    train_module_dir = checkpoint_path / "model_and_optim"
    load_model_and_optim_state(str(train_module_dir), model)
    model.to(device)
    model.eval()

    # Create dataset
    logger.info(f"Loading dataset from {args.h5py_dir}")
    dataset = OlmoEarthDataset(
        h5py_dir=UPath(args.h5py_dir),
        training_modalities=args.training_modalities,
        dtype=np.float32,
        normalize=True,
    )
    dataset.prepare()
    logger.info(f"Dataset has {len(dataset)} samples")

    # Extract embeddings
    logger.info(f"Using pooling type: {pooling_type}")
    embeddings = extract_embeddings(
        model=model,
        dataset=dataset,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=device,
        patch_size=args.patch_size,
        pooling_type=pooling_type,
    )

    # Compute quantiles
    quantiles = compute_quantiles(embeddings)

    # Save to HDF5
    save_quantiles(quantiles, args.output, dim=embeddings.shape[1])

    logger.info("Done!")


if __name__ == "__main__":
    main()

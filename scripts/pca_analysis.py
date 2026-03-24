"""PCA analysis of encoder embeddings from a distributed checkpoint.

Reproduces the analysis from https://geospatialml.com/posts/compressing-earth-embeddings/
to measure dimensional redundancy in encoder embeddings.

Usage:
    # From a distributed checkpoint — requires the training script to build model config:
    torchrun --nproc_per_node=1 scripts/pca_analysis.py \
        --checkpoint_dir /path/to/checkpoints/step300000 \
        --train_script scripts/vnext/single_bandset_band_dropout/single_bandset_masked_neg.py \
        --experiment masked_neg_decoder_supervision_ic005_sup03x_no_s1_band_dropout \
        --output pca_analysis.png

    # From a converted weights.pth checkpoint (e.g. HuggingFace models):
    python scripts/pca_analysis.py \
        --model_path /path/to/model_dir \
        --output pca_analysis.png
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_from_distributed_checkpoint(
    checkpoint_dir: str, train_script: str, experiment: str
) -> torch.nn.Module:
    """Load model from a distributed training checkpoint.

    Uses the training script to build the model config (same pattern as
    checkpoint_sweep_evals.py), then loads weights from the checkpoint.

    Args:
        checkpoint_dir: Path to a step directory (e.g. .../step300000/).
        train_script: Path to the training script (e.g. scripts/vnext/.../single_bandset_masked_neg.py).
        experiment: Experiment key in the script's EXPERIMENTS dict.
    """
    from olmo_core.distributed.checkpoint import load_model_and_optim_state

    from olmoearth_pretrain.internal.all_evals import load_user_module
    from olmoearth_pretrain.internal.experiment import SubCmd

    user_mod = load_user_module(train_script)

    # Look up experiment builders
    experiments = user_mod.EXPERIMENTS
    if experiment not in experiments:
        raise ValueError(
            f"Unknown experiment: {experiment}. Available: {list(experiments.keys())}"
        )
    common_builder, model_builder, _, _ = experiments[experiment]

    # Build common components (use dummy values for script/cmd/run_name/cluster)
    common = common_builder(train_script, SubCmd.evaluate, "pca_analysis", "local", [])
    model_config = model_builder(common)
    model = model_config.build()

    train_module_dir = str(Path(checkpoint_dir) / "model_and_optim")
    load_model_and_optim_state(train_module_dir, model)
    logger.info(f"Loaded model from distributed checkpoint: {checkpoint_dir}")
    return model


def load_model_from_path(model_path: str) -> torch.nn.Module:
    """Load model from a converted checkpoint (config.json + weights.pth)."""
    from olmoearth_pretrain.model_loader import load_model_from_path as _load

    model = _load(model_path)
    logger.info(f"Loaded model from path: {model_path}")
    return model


def _encoder_forward(
    encoder: torch.nn.Module, sample: torch.nn.Module, patch_size: int
) -> torch.nn.Module:
    """Call encoder forward, handling both FlexiVitBase (returns dict) and STBase (returns tuple)."""
    from olmoearth_pretrain.nn.flexi_vit import FlexiVitBase

    if isinstance(encoder, FlexiVitBase):
        output_dict = encoder(sample, patch_size=patch_size, fast_pass=True)
        return output_dict["tokens_and_masks"]
    else:
        # STEncoder returns (TokensAndMasks, projected_pooled)
        tokens_and_masks, _ = encoder(sample, patch_size=patch_size)
        return tokens_and_masks


def get_embeddings(
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 32,
    patch_size: int = 8,
) -> np.ndarray:
    """Get mean-pooled encoder embeddings on the EuroSAT eval dataset."""
    from olmoearth_pretrain.evals.datasets import get_eval_dataset
    from olmoearth_pretrain.nn.pooling import PoolingType, pool_unmasked_tokens
    from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

    dataset = get_eval_dataset("m-eurosat", split="test")
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    model.eval()
    embeddings_list: list[torch.Tensor] = []
    with torch.no_grad():
        for i, (masked_sample, label) in enumerate(data_loader):
            sample_dict = masked_sample.as_dict()
            for key, val in sample_dict.items():
                sample_dict[key] = val.to(device=device)
            masked_sample = MaskedOlmoEarthSample.from_dict(sample_dict)

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                tokens_and_masks = _encoder_forward(model, masked_sample, patch_size)

            # Mean-pool unmasked encoder tokens -> (batch, dim)
            batch_embeddings = pool_unmasked_tokens(
                tokens_and_masks, PoolingType.MEAN, spatial_pooling=False
            )
            embeddings_list.append(batch_embeddings.float().cpu())

            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1} batches")

    embeddings = torch.cat(embeddings_list, dim=0).numpy()
    logger.info(f"Collected embeddings: shape={embeddings.shape}")
    return embeddings


def pca_analysis(embeddings: np.ndarray) -> dict:
    """Run PCA analysis and return results."""
    # Center the embeddings
    centered = embeddings - embeddings.mean(axis=0)

    # SVD of centered embeddings
    _, singular_values, Vt = np.linalg.svd(centered, full_matrices=False)

    # Variance explained by each component
    variance = singular_values**2 / (len(embeddings) - 1)
    total_variance = variance.sum()
    variance_ratio = variance / total_variance
    cumulative_variance = np.cumsum(variance_ratio)

    # Key metrics
    dims_for_90 = int(np.searchsorted(cumulative_variance, 0.90) + 1)
    dims_for_95 = int(np.searchsorted(cumulative_variance, 0.95) + 1)
    dims_for_98 = int(np.searchsorted(cumulative_variance, 0.98) + 1)
    dims_for_99 = int(np.searchsorted(cumulative_variance, 0.99) + 1)

    results = {
        "singular_values": singular_values,
        "variance_ratio": variance_ratio,
        "cumulative_variance": cumulative_variance,
        "embedding_dim": embeddings.shape[1],
        "num_samples": embeddings.shape[0],
        "dims_for_90": dims_for_90,
        "dims_for_95": dims_for_95,
        "dims_for_98": dims_for_98,
        "dims_for_99": dims_for_99,
    }
    return results


def plot_results(results: dict, output_path: str) -> None:
    """Plot PCA analysis results."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    d = results["embedding_dim"]

    # 1. Singular value spectrum
    ax = axes[0]
    ax.semilogy(range(1, d + 1), results["singular_values"], "b-", linewidth=1.5)
    ax.set_xlabel("Component")
    ax.set_ylabel("Singular Value (log scale)")
    ax.set_title("Singular Value Spectrum")
    ax.grid(True, alpha=0.3)

    # 2. Variance explained per component
    ax = axes[1]
    ax.bar(range(1, min(d, 50) + 1), results["variance_ratio"][:50], color="steelblue")
    ax.set_xlabel("Component")
    ax.set_ylabel("Variance Explained")
    ax.set_title("Variance Explained per Component (first 50)")
    ax.grid(True, alpha=0.3, axis="y")

    # 3. Cumulative variance
    ax = axes[2]
    ax.plot(range(1, d + 1), results["cumulative_variance"], "b-", linewidth=2)
    ax.axhline(y=0.90, color="gray", linestyle="--", alpha=0.5, label="90%")
    ax.axhline(y=0.95, color="gray", linestyle="-.", alpha=0.5, label="95%")
    ax.axhline(y=0.98, color="gray", linestyle=":", alpha=0.5, label="98%")

    # Mark key thresholds
    for pct, dims, color in [
        (0.90, results["dims_for_90"], "green"),
        (0.95, results["dims_for_95"], "orange"),
        (0.98, results["dims_for_98"], "red"),
    ]:
        ax.axvline(x=dims, color=color, linestyle="--", alpha=0.5)
        ax.annotate(
            f"{int(pct * 100)}%: {dims}d",
            xy=(dims, pct),
            xytext=(dims + d * 0.05, pct - 0.03),
            fontsize=9,
            color=color,
        )

    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Variance Explained")
    ax.set_title("Cumulative Variance Explained")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"PCA Analysis of Encoder Embeddings ({results['num_samples']} samples, {d}d)\n"
        f"90%: {results['dims_for_90']}d | "
        f"95%: {results['dims_for_95']}d | "
        f"98%: {results['dims_for_98']}d | "
        f"99%: {results['dims_for_99']}d",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved plot to {output_path}")
    plt.close()


def main() -> None:
    """Run pca analysis."""
    parser = argparse.ArgumentParser(description="PCA analysis of encoder embeddings")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Path to distributed checkpoint step directory (e.g. .../step300000/)",
    )
    group.add_argument(
        "--model_path",
        type=str,
        help="Path to converted model directory (config.json + weights.pth)",
    )
    parser.add_argument(
        "--train_script",
        type=str,
        default=None,
        help="Path to training script (required for --checkpoint_dir)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment key in the training script's EXPERIMENTS dict (required for --checkpoint_dir)",
    )
    parser.add_argument("--output", type=str, default="pca_analysis.png")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patch_size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoint_dir:
        if not args.train_script or not args.experiment:
            parser.error(
                "--train_script and --experiment are required with --checkpoint_dir"
            )
        model = load_model_from_distributed_checkpoint(
            args.checkpoint_dir, args.train_script, args.experiment
        )
    else:
        model = load_model_from_path(args.model_path)

    model = model.to(device)
    model.eval()

    # For LatentMIM models, use the encoder
    if hasattr(model, "encoder"):
        encoder = model.encoder
    else:
        encoder = model

    embeddings = get_embeddings(encoder, device, args.batch_size, args.patch_size)
    results = pca_analysis(embeddings)

    # Print summary
    print(f"\n{'=' * 60}")
    print("PCA Analysis Summary")
    print(f"{'=' * 60}")
    print(f"Embedding dimension: {results['embedding_dim']}")
    print(f"Number of samples:   {results['num_samples']}")
    print("")
    print(f"Dimensions for 90% variance: {results['dims_for_90']}")
    print(f"Dimensions for 95% variance: {results['dims_for_95']}")
    print(f"Dimensions for 98% variance: {results['dims_for_98']}")
    print(f"Dimensions for 99% variance: {results['dims_for_99']}")
    print("")
    print(f"Top-5 component variance:    {results['variance_ratio'][:5]}")
    print(f"{'=' * 60}")

    plot_results(results, args.output)


if __name__ == "__main__":
    main()

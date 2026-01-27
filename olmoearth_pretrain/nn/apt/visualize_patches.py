"""Visualize APT patchification on EuroSAT samples.

Usage:
    python -m olmoearth_pretrain.nn.apt.visualize_patches --output_dir /tmp/apt_viz --num_samples 10 --threshold 0.8
"""

import argparse
import logging
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.evals.datasets import get_eval_dataset
from olmoearth_pretrain.evals.datasets.configs import dataset_to_config
from olmoearth_pretrain.evals.datasets.normalize import NormMethod
from olmoearth_pretrain.nn.apt.config import APTConfig
from olmoearth_pretrain.nn.apt.partitioner import QuadtreePartitioner
from olmoearth_pretrain.nn.apt.scorers import EntropyScorer

logger = logging.getLogger(__name__)


def visualize_sample(
    image: np.ndarray,
    patches: list,
    base_patch_size: int,
    threshold: float,
    title: str = "",
    save_path: str | None = None,
) -> None:
    """Visualize a sample with APT patchification overlay.

    Args:
        image: RGB image [H, W, 3] normalized to [0, 1]
        patches: List of PatchDescriptor objects
        base_patch_size: Base patch size in pixels
        threshold: Entropy threshold used
        title: Title for the plot
        save_path: Path to save the figure (if None, displays)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # 2. Image with patch grid overlay
    axes[1].imshow(image)

    # Color patches by scale
    colors = {
        0: (0, 1, 0, 0.3),  # Green for 4px (base)
        1: (1, 0, 0, 0.3),  # Red for 8px
        2: (0, 0, 1, 0.3),  # Blue for 16px (if any)
    }
    edge_colors = {
        0: "green",
        1: "red",
        2: "blue",
    }

    for patch in patches:
        x_px = patch.x * patch.size
        y_px = patch.y * patch.size
        size = patch.size
        scale = patch.scale

        # Draw filled rectangle
        rect = mpatches.Rectangle(
            (x_px, y_px),
            size,
            size,
            linewidth=1,
            edgecolor=edge_colors.get(scale, "white"),
            facecolor=colors.get(scale, (0.5, 0.5, 0.5, 0.3)),
        )
        axes[1].add_patch(rect)

    # Add legend
    legend_patches = [
        mpatches.Patch(color="green", alpha=0.5, label=f"{base_patch_size}px (fine)"),
        mpatches.Patch(color="red", alpha=0.5, label=f"{base_patch_size * 2}px (coarse)"),
    ]
    axes[1].legend(handles=legend_patches, loc="upper right")
    axes[1].set_title(f"APT Patches (threshold={threshold:.2f})")
    axes[1].axis("off")

    # 3. Entropy heatmap
    h, w = image.shape[:2]
    entropy_map = np.zeros((h, w))

    for patch in patches:
        x_px = patch.x * patch.size
        y_px = patch.y * patch.size
        size = patch.size
        entropy_map[y_px : y_px + size, x_px : x_px + size] = patch.score

    im = axes[2].imshow(entropy_map, cmap="hot")
    axes[2].axhline(y=0, color="white", linestyle="--", alpha=0)  # dummy for colorbar alignment
    plt.colorbar(im, ax=axes[2], label="Entropy Score")
    axes[2].axhline(y=h // 2, color="cyan", linestyle="--", alpha=0.5, label=f"threshold={threshold}")
    axes[2].set_title(f"Entropy Map (threshold={threshold:.2f} shown in title)")
    axes[2].axis("off")

    # Add overall title with stats
    scale_counts = {}
    for p in patches:
        scale_counts[p.scale] = scale_counts.get(p.scale, 0) + 1

    fine_count = scale_counts.get(0, 0)
    coarse_count = scale_counts.get(1, 0)
    total = fine_count + coarse_count

    fig.suptitle(
        f"{title}\n"
        f"Patches: {fine_count} fine ({base_patch_size}px), {coarse_count} coarse ({base_patch_size * 2}px) | "
        f"Fine ratio: {100 * fine_count / max(1, total):.1f}%",
        fontsize=12,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved visualization to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize APT patchification on EuroSAT")
    parser.add_argument("--output_dir", type=str, default="/tmp/apt_viz", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to visualize")
    parser.add_argument("--threshold", type=float, default=0.8, help="Entropy threshold")
    parser.add_argument("--base_patch_size", type=int, default=4, help="Base patch size")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load EuroSAT dataset
    dataset_name = "m-eurosat"
    config = dataset_to_config(dataset_name)

    logger.info(f"Loading {dataset_name} dataset...")
    dataset = get_eval_dataset(
        eval_dataset=dataset_name,
        split=args.split,
        norm_method=NormMethod.NORM_NO_CLIP,
        norm_stats_from_pretrained=True,
    )

    # Build APT components
    modality_spec = Modality.SENTINEL2_L2A
    rgb_bands = [modality_spec.band_order.index(b) for b in ["B04", "B03", "B02"]]  # RGB
    scorer_bands = tuple(modality_spec.band_order.index(b) for b in ["B02", "B03", "B04"])

    scorer = EntropyScorer(
        num_bins=32,
        bands=scorer_bands,
        normalizer=None,
    )

    partitioner = QuadtreePartitioner(
        scorer=scorer,
        base_patch_size=args.base_patch_size,
        num_scales=2,
        thresholds=[args.threshold],
    )

    logger.info(f"Visualizing {args.num_samples} samples with threshold={args.threshold}...")

    # Collect stats
    all_scores = []
    all_fine_ratios = []

    for i in range(min(args.num_samples, len(dataset))):
        sample = dataset[i]
        label = None

        # Dataset returns (MaskedOlmoEarthSample, label) tuple
        if isinstance(sample, tuple) and len(sample) == 2:
            masked_sample, label = sample
            if hasattr(label, 'item'):
                label = label.item()
        else:
            masked_sample = sample

        # Get sentinel2 data from the masked sample
        if hasattr(masked_sample, "sentinel2_l2a") and masked_sample.sentinel2_l2a is not None:
            image_data = masked_sample.sentinel2_l2a
        else:
            logger.warning(f"Sample {i} has no sentinel2_l2a data, skipping")
            continue

        # Convert to numpy
        if isinstance(image_data, torch.Tensor):
            image_data = image_data.numpy()

        # Handle shape: [H, W, T, C] -> [H, W, C]
        if image_data.ndim == 4:
            image_data = image_data[:, :, 0, :]  # Take first timestep
        elif image_data.ndim == 3 and image_data.shape[0] < image_data.shape[-1]:
            # [C, H, W] -> [H, W, C]
            image_data = np.transpose(image_data, (1, 2, 0))

        # Run partitioner
        patches = partitioner.partition(image_data.astype(np.float32))

        # Collect scores
        scores = [p.score for p in patches]
        all_scores.extend(scores)

        fine_count = sum(1 for p in patches if p.scale == 0)
        total = len(patches)
        fine_ratio = fine_count / max(1, total)
        all_fine_ratios.append(fine_ratio)

        # Create RGB visualization
        rgb_image = image_data[..., rgb_bands].copy()

        # Normalize for display
        for c in range(3):
            channel = rgb_image[..., c]
            p2, p98 = np.percentile(channel, [2, 98])
            rgb_image[..., c] = np.clip((channel - p2) / (p98 - p2 + 1e-8), 0, 1)

        # Set title with label if available
        if label is not None:
            title = f"Sample {i} (label={label})"
        else:
            title = f"Sample {i}"

        save_path = output_dir / f"sample_{i:04d}.png"
        visualize_sample(
            image=rgb_image,
            patches=patches,
            base_patch_size=args.base_patch_size,
            threshold=args.threshold,
            title=title,
            save_path=str(save_path),
        )

    # Print summary statistics
    all_scores = np.array(all_scores)
    percentiles = np.percentile(all_scores, [0, 10, 25, 50, 75, 90, 100])

    print("\n" + "=" * 60)
    print("ENTROPY SCORE SUMMARY")
    print("=" * 60)
    print(f"Threshold: {args.threshold}")
    print(f"Total patches analyzed: {len(all_scores)}")
    print(f"\nPercentiles:")
    print(f"  Min:    {percentiles[0]:.3f}")
    print(f"  p10:    {percentiles[1]:.3f}")
    print(f"  p25:    {percentiles[2]:.3f}")
    print(f"  Median: {percentiles[3]:.3f}")
    print(f"  p75:    {percentiles[4]:.3f}")
    print(f"  p90:    {percentiles[5]:.3f}")
    print(f"  Max:    {percentiles[6]:.3f}")
    print(f"\nMean: {all_scores.mean():.3f}, Std: {all_scores.std():.3f}")
    print(f"\nAbove threshold (→fine): {np.sum(all_scores > args.threshold)} ({100 * np.mean(all_scores > args.threshold):.1f}%)")
    print(f"Below threshold (→coarse): {np.sum(all_scores <= args.threshold)} ({100 * np.mean(all_scores <= args.threshold):.1f}%)")
    print(f"\nAvg fine patch ratio per sample: {100 * np.mean(all_fine_ratios):.1f}%")
    print("=" * 60)
    print(f"\nVisualizations saved to: {output_dir}")

    # Also create a histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(all_scores, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(x=args.threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold={args.threshold}")
    ax.axvline(x=percentiles[3], color="green", linestyle="--", linewidth=2, label=f"Median={percentiles[3]:.2f}")
    ax.set_xlabel("Entropy Score")
    ax.set_ylabel("Count")
    ax.set_title(f"Entropy Score Distribution (n={len(all_scores)} patches)")
    ax.legend()
    plt.savefig(output_dir / "entropy_histogram.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved histogram to {output_dir / 'entropy_histogram.png'}")


if __name__ == "__main__":
    main()

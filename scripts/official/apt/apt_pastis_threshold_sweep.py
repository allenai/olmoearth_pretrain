"""Score-only APT threshold sweep for PASTIS (no finetuning).

This script helps choose thresholds for 3-scale or 4-scale APT by:
1) Scoring PASTIS samples and saving per-sample score maps + preview images.
2) Sweeping candidate threshold sets and estimating token usage.

Usage:
    python scripts/official/apt/apt_pastis_threshold_sweep.py \
        --num_scales 3 \
        --split train \
        --num_samples 1000 \
        --output_dir /tmp/apt_pastis_sweep_3scale

    python scripts/official/apt/apt_pastis_threshold_sweep.py \
        --num_scales 4 \
        --split train \
        --num_samples 1000 \
        --output_dir /tmp/apt_pastis_sweep_4scale

    # Optional explicit threshold sets to evaluate
    # (repeat flag; values correspond to finer->coarser split thresholds)
    --threshold_set 0.5,0.8 --threshold_set 0.6,0.9
"""

import argparse
import concurrent.futures
import csv
import itertools
import json
import logging
import os
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.evals.datasets import EvalDatasetPartition, get_eval_dataset
from olmoearth_pretrain.evals.datasets.normalize import NormMethod
from olmoearth_pretrain.nn.apt.config import APTConfig
from olmoearth_pretrain.nn.apt.partitioner import QuadtreePartitioner
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

logger = logging.getLogger(__name__)


def _get_default_config(num_scales: int) -> APTConfig:
    if num_scales == 3:
        return APTConfig.default_s2_finetune_config_3scale()
    if num_scales == 4:
        return APTConfig.default_s2_finetune_config_4scale()
    raise ValueError(f"Unsupported num_scales={num_scales}. Expected 3 or 4.")


def _parse_threshold_set(threshold_text: str, expected_len: int) -> list[float]:
    values = [float(x.strip()) for x in threshold_text.split(",") if x.strip()]
    if len(values) != expected_len:
        raise ValueError(
            f"Expected {expected_len} thresholds, got {len(values)} from '{threshold_text}'"
        )
    return values


def _to_numpy_s2(sample: MaskedOlmoEarthSample) -> np.ndarray:
    image = sample.sentinel2_l2a
    if image is None:
        raise ValueError("Sample has no sentinel2_l2a.")
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    return image.astype(np.float32)


def _rgb_from_first_timestep(image: np.ndarray) -> np.ndarray:
    # image shape: [H, W, T, C]
    first_frame = image[:, :, 0, :]
    rgb_bands = [
        Modality.SENTINEL2_L2A.band_order.index("B04"),
        Modality.SENTINEL2_L2A.band_order.index("B03"),
        Modality.SENTINEL2_L2A.band_order.index("B02"),
    ]
    rgb = first_frame[..., rgb_bands].copy()
    for c in range(3):
        channel = rgb[..., c]
        p2, p98 = np.percentile(channel, [2, 98])
        rgb[..., c] = np.clip((channel - p2) / (p98 - p2 + 1e-8), 0, 1)
    return rgb


def _scale_to_color(scale: int) -> tuple[float, float, float, float]:
    palette = [
        (0.2, 0.8, 0.2, 0.25),
        (1.0, 0.2, 0.2, 0.25),
        (0.2, 0.4, 1.0, 0.25),
        (1.0, 0.7, 0.2, 0.25),
    ]
    return palette[scale] if scale < len(palette) else (0.7, 0.7, 0.7, 0.25)


def _scale_to_edge_color(scale: int) -> str:
    edge_colors = ["green", "red", "blue", "orange"]
    return edge_colors[scale] if scale < len(edge_colors) else "white"


def _save_preview(
    image: np.ndarray,
    patches: list,
    score_grids: dict[int, np.ndarray],
    threshold_set: list[float],
    base_patch_size: int,
    save_path: Path,
) -> None:
    rgb = _rgb_from_first_timestep(image)
    num_scales = len(score_grids)
    fig, axes = plt.subplots(1, 2 + num_scales, figsize=(6 + 4 * num_scales, 5))

    axes[0].imshow(rgb)
    axes[0].set_title("RGB (t=0)")
    axes[0].axis("off")

    axes[1].imshow(rgb)
    scale_counts: dict[int, int] = {}
    for patch in patches:
        x_px = patch.x * base_patch_size
        y_px = patch.y * base_patch_size
        rect = mpatches.Rectangle(
            (x_px, y_px),
            patch.size,
            patch.size,
            linewidth=1.0,
            edgecolor=_scale_to_edge_color(patch.scale),
            facecolor=_scale_to_color(patch.scale),
        )
        axes[1].add_patch(rect)
        scale_counts[patch.scale] = scale_counts.get(patch.scale, 0) + 1
    axes[1].set_title(f"APT patches\nthr={threshold_set}")
    axes[1].axis("off")

    legend_handles = []
    for scale in sorted(scale_counts):
        patch_size = base_patch_size * (2**scale)
        legend_handles.append(
            mpatches.Patch(
                color=_scale_to_edge_color(scale),
                alpha=0.6,
                label=f"{patch_size}px: {scale_counts[scale]}",
            )
        )
    if legend_handles:
        axes[1].legend(handles=legend_handles, loc="upper right", fontsize=8)

    for i, scale in enumerate(sorted(score_grids)):
        score_grid = score_grids[scale][:, :, 0]
        ax = axes[2 + i]
        im = ax.imshow(score_grid, cmap="hot")
        patch_size = base_patch_size * (2**scale)
        ax.set_title(f"Scores @ {patch_size}px")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _build_threshold_candidates(
    score_values: dict[int, list[float]],
    quantiles: list[float],
    num_scales: int,
) -> dict[int, list[float]]:
    candidates: dict[int, list[float]] = {}
    for scale in range(1, num_scales):
        values = np.array(score_values[scale], dtype=np.float32)
        if values.size == 0:
            raise ValueError(f"No score values collected for scale={scale}")
        cands = sorted(set(float(np.quantile(values, q)) for q in quantiles))
        candidates[scale] = cands
    return candidates


def _build_monotonic_threshold_sets(candidates: dict[int, list[float]]) -> list[list[float]]:
    scales = sorted(candidates.keys())  # 1..num_scales-1 (fine->coarse)
    threshold_sets: list[list[float]] = []
    for values in itertools.product(*(candidates[s] for s in scales)):
        # Keep monotonic threshold schedule for stability:
        # finer split thresholds <= coarser split thresholds.
        if all(values[i] <= values[i + 1] for i in range(len(values) - 1)):
            threshold_sets.append([float(v) for v in values])
    return threshold_sets


def _token_stats_from_score_grids(
    sample_scores: dict[int, np.ndarray],
    thresholds: list[float],
    num_scales: int,
) -> tuple[float, float]:
    """Compute token stats from score grids only (no image reload/repartition)."""
    if num_scales < 2:
        raise ValueError("num_scales must be >= 2")
    if len(thresholds) != num_scales - 1:
        raise ValueError(
            f"Expected {num_scales - 1} thresholds, got {len(thresholds)}"
        )

    top_scale = num_scales - 1
    active = np.ones_like(sample_scores[top_scale], dtype=bool)
    total_tokens = 0

    # Walk coarse -> fine and apply split/keep decisions.
    for scale in range(top_scale, 0, -1):
        scores = sample_scores[scale]
        if scores.shape != active.shape:
            raise ValueError(
                f"Shape mismatch at scale {scale}: scores={scores.shape}, active={active.shape}"
            )
        keep = active & (scores < thresholds[scale - 1])
        split = active & ~keep
        total_tokens += int(keep.sum())
        # Every split patch creates 4 children at the next finer scale.
        active = np.repeat(np.repeat(split, 2, axis=0), 2, axis=1)

    # Finest scale (scale 0) always kept.
    total_tokens += int(active.sum())

    num_timesteps = sample_scores[top_scale].shape[2]
    uniform_tokens_per_frame = int((sample_scores[1].shape[0] * 2) * (sample_scores[1].shape[1] * 2))
    tokens_per_frame = total_tokens / max(1, num_timesteps)
    token_ratio = tokens_per_frame / max(1, uniform_tokens_per_frame)
    return tokens_per_frame, token_ratio


def _evaluate_threshold_set(
    threshold_id: int,
    thresholds: list[float],
    sample_indices: list[int],
    sample_score_grids: list[dict[int, np.ndarray]],
    num_scales: int,
) -> tuple[int, list[dict], dict]:
    per_sample_rows: list[dict] = []
    per_sample_tokens: list[float] = []
    per_sample_ratios: list[float] = []

    for sample_idx, sample_scores in zip(sample_indices, sample_score_grids):
        tokens_per_frame, token_ratio = _token_stats_from_score_grids(
            sample_scores=sample_scores,
            thresholds=thresholds,
            num_scales=num_scales,
        )
        per_sample_tokens.append(tokens_per_frame)
        per_sample_ratios.append(token_ratio)
        per_sample_rows.append(
            {
                "threshold_id": threshold_id,
                "thresholds": json.dumps(thresholds),
                "sample_index": sample_idx,
                "tokens_per_frame": f"{tokens_per_frame:.6f}",
                "token_ratio_vs_uniform": f"{token_ratio:.6f}",
            }
        )

    ratios = np.array(per_sample_ratios, dtype=np.float32)
    tokens = np.array(per_sample_tokens, dtype=np.float32)
    summary_row = {
        "threshold_id": threshold_id,
        "thresholds": json.dumps(thresholds),
        "num_samples": len(per_sample_ratios),
        "mean_tokens_per_frame": f"{tokens.mean():.6f}",
        "median_tokens_per_frame": f"{np.median(tokens):.6f}",
        "p90_tokens_per_frame": f"{np.percentile(tokens, 90):.6f}",
        "mean_token_ratio_vs_uniform": f"{ratios.mean():.6f}",
        "median_token_ratio_vs_uniform": f"{np.median(ratios):.6f}",
        "p90_token_ratio_vs_uniform": f"{np.percentile(ratios, 90):.6f}",
    }
    return threshold_id, per_sample_rows, summary_row


def _save_json(path: Path, data: dict) -> None:
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def _write_summary_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _pick_thresholds_for_targets(
    summary_rows: list[dict],
    target_token_ratios: list[float],
) -> list[dict]:
    if not summary_rows:
        return []

    recommendations: list[dict] = []
    for target in target_token_ratios:
        best_row = min(
            summary_rows,
            key=lambda row: abs(float(row["mean_token_ratio_vs_uniform"]) - target),
        )
        realized = float(best_row["mean_token_ratio_vs_uniform"])
        recommendations.append(
            {
                "target_token_ratio_vs_uniform": f"{target:.6f}",
                "selected_threshold_id": best_row["threshold_id"],
                "selected_thresholds": best_row["thresholds"],
                "realized_mean_token_ratio_vs_uniform": f"{realized:.6f}",
                "abs_error": f"{abs(realized - target):.6f}",
                "mean_tokens_per_frame": best_row["mean_tokens_per_frame"],
                "median_tokens_per_frame": best_row["median_tokens_per_frame"],
                "p90_tokens_per_frame": best_row["p90_tokens_per_frame"],
            }
        )
    return recommendations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score-only threshold sweep for PASTIS APT."
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--partition", type=str, default=EvalDatasetPartition.TRAIN1X)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--num_scales", type=int, choices=[3, 4], default=3)
    parser.add_argument(
        "--quantiles",
        type=str,
        default="0.4,0.5,0.6,0.7,0.8,0.9",
        help="Comma-separated quantiles used to build threshold candidates.",
    )
    parser.add_argument(
        "--threshold_set",
        action="append",
        default=[],
        help="Explicit threshold set, e.g. '0.5,0.8' for 3-scale. Repeatable.",
    )
    parser.add_argument(
        "--save_preview_images",
        action="store_true",
        help="Save per-sample preview PNGs (RGB + patches + score maps).",
    )
    parser.add_argument(
        "--max_preview_images",
        type=int,
        default=200,
        help="Limit preview image count when --save_preview_images is enabled.",
    )
    parser.add_argument(
        "--target_token_ratios",
        type=str,
        default="0.35,0.50,0.65",
        help=(
            "Comma-separated target mean token ratios (vs uniform) used to pick "
            "recommended threshold sets."
        ),
    )
    parser.add_argument(
        "--num_threshold_workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Parallel workers for threshold evaluation stage.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    output_dir = Path(args.output_dir)
    score_dir = output_dir / "sample_scores"
    preview_dir = output_dir / "sample_previews"
    score_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    apt_config = _get_default_config(args.num_scales)
    base_patch_size = apt_config.partitioner.base_patch_size
    default_thresholds = list(apt_config.partitioner.thresholds)
    scorer = apt_config.scorer.build()

    logger.info(
        "Loading PASTIS split=%s partition=%s num_scales=%d base_patch_size=%d",
        args.split,
        args.partition,
        args.num_scales,
        base_patch_size,
    )
    dataset = get_eval_dataset(
        eval_dataset="pastis",
        split=args.split,
        norm_stats_from_pretrained=True,
        input_modalities=[Modality.SENTINEL2_L2A.name],
        partition=args.partition,
        norm_method=NormMethod.NORM_NO_CLIP,
    )

    sample_indices = list(range(len(dataset)))
    if args.num_samples > 0:
        sample_indices = sample_indices[: min(args.num_samples, len(sample_indices))]
    logger.info("Processing %d samples", len(sample_indices))

    quantiles = [float(x.strip()) for x in args.quantiles.split(",") if x.strip()]
    score_values: dict[int, list[float]] = {scale: [] for scale in range(1, args.num_scales)}
    metadata_rows: list[dict] = []
    sample_score_grids: list[dict[int, np.ndarray]] = []

    preview_partitioner = QuadtreePartitioner(
        scorer=scorer,
        base_patch_size=base_patch_size,
        num_scales=args.num_scales,
        thresholds=default_thresholds,
    )

    for i, sample_idx in enumerate(sample_indices):
        sample, _ = dataset[sample_idx]
        image = _to_numpy_s2(sample)

        sample_scores: dict[int, np.ndarray] = {}
        for scale in range(1, args.num_scales):
            patch_size = base_patch_size * (2**scale)
            score_grid = scorer.compute_scores_grid(image, patch_size)
            sample_scores[scale] = score_grid.astype(np.float32)
            score_values[scale].extend(score_grid.reshape(-1).astype(float).tolist())
        sample_score_grids.append(sample_scores)

        npz_path = score_dir / f"sample_{sample_idx:06d}.npz"
        np.savez_compressed(
            npz_path,
            sample_index=sample_idx,
            image_shape=np.array(image.shape, dtype=np.int32),
            **{
                f"scale_{scale}_scores": sample_scores[scale]
                for scale in sorted(sample_scores)
            },
        )

        metadata_rows.append(
            {
                "sample_index": sample_idx,
                "height": image.shape[0],
                "width": image.shape[1],
                "timesteps": image.shape[2],
                "channels": image.shape[3],
                "score_npz": str(npz_path),
            }
        )

        if args.save_preview_images and i < args.max_preview_images:
            t0_patches = preview_partitioner.partition(image[:, :, 0, :], timestep=0)
            _save_preview(
                image=image,
                patches=t0_patches,
                score_grids=sample_scores,
                threshold_set=default_thresholds,
                base_patch_size=base_patch_size,
                save_path=preview_dir / f"sample_{sample_idx:06d}.png",
            )

        if (i + 1) % 50 == 0:
            logger.info("Scored %d/%d samples", i + 1, len(sample_indices))

    _write_summary_csv(output_dir / "sample_metadata.csv", metadata_rows)

    threshold_sets: list[list[float]] = []
    expected_len = args.num_scales - 1
    if args.threshold_set:
        for threshold_text in args.threshold_set:
            threshold_sets.append(_parse_threshold_set(threshold_text, expected_len))
        logger.info("Using %d explicit threshold sets", len(threshold_sets))
        threshold_candidates: dict[int, list[float]] = {}
    else:
        threshold_candidates = _build_threshold_candidates(
            score_values=score_values,
            quantiles=quantiles,
            num_scales=args.num_scales,
        )
        threshold_sets = _build_monotonic_threshold_sets(threshold_candidates)
        logger.info(
            "Generated %d monotonic threshold sets from quantiles %s",
            len(threshold_sets),
            quantiles,
        )

    logger.info(
        "Evaluating %d threshold sets with %d worker(s) using score-grid-only token counting",
        len(threshold_sets),
        args.num_threshold_workers,
    )
    threshold_results: dict[int, tuple[list[dict], dict]] = {}
    max_workers = max(1, args.num_threshold_workers)
    if max_workers == 1 or len(threshold_sets) <= 1:
        for threshold_id, thresholds in enumerate(threshold_sets):
            tid, per_rows, summary_row = _evaluate_threshold_set(
                threshold_id=threshold_id,
                thresholds=thresholds,
                sample_indices=sample_indices,
                sample_score_grids=sample_score_grids,
                num_scales=args.num_scales,
            )
            threshold_results[tid] = (per_rows, summary_row)
            logger.info(
                "Threshold %s -> mean token ratio %.4f",
                thresholds,
                float(summary_row["mean_token_ratio_vs_uniform"]),
            )
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _evaluate_threshold_set,
                    threshold_id,
                    thresholds,
                    sample_indices,
                    sample_score_grids,
                    args.num_scales,
                ): (threshold_id, thresholds)
                for threshold_id, thresholds in enumerate(threshold_sets)
            }
            completed = 0
            total = len(futures)
            for future in concurrent.futures.as_completed(futures):
                threshold_id, thresholds = futures[future]
                tid, per_rows, summary_row = future.result()
                threshold_results[tid] = (per_rows, summary_row)
                completed += 1
                logger.info(
                    "[%d/%d] Threshold %s -> mean token ratio %.4f",
                    completed,
                    total,
                    thresholds,
                    float(summary_row["mean_token_ratio_vs_uniform"]),
                )

    per_sample_rows: list[dict] = []
    summary_rows: list[dict] = []
    for threshold_id in sorted(threshold_results):
        per_rows, summary_row = threshold_results[threshold_id]
        per_sample_rows.extend(per_rows)
        summary_rows.append(summary_row)

    summary_rows.sort(key=lambda row: float(row["mean_token_ratio_vs_uniform"]))

    _write_summary_csv(output_dir / "threshold_sweep_summary.csv", summary_rows)
    _write_summary_csv(output_dir / "threshold_sweep_per_sample.csv", per_sample_rows)

    target_token_ratios = [
        float(x.strip()) for x in args.target_token_ratios.split(",") if x.strip()
    ]
    recommended_rows = _pick_thresholds_for_targets(
        summary_rows=summary_rows,
        target_token_ratios=target_token_ratios,
    )
    _write_summary_csv(output_dir / "threshold_recommendations.csv", recommended_rows)

    config_payload = {
        "split": args.split,
        "partition": args.partition,
        "num_samples": len(sample_indices),
        "num_scales": args.num_scales,
        "base_patch_size": base_patch_size,
        "default_thresholds": default_thresholds,
        "quantiles": quantiles,
        "target_token_ratios": target_token_ratios,
        "threshold_candidates_by_scale": threshold_candidates
        if not args.threshold_set
        else "explicit_threshold_set_input",
    }
    _save_json(output_dir / "run_config.json", config_payload)

    logger.info("Done. Outputs written to %s", output_dir)
    logger.info(
        "Use threshold_sweep_summary.csv to pick thresholds by token budget before finetuning."
    )
    logger.info(
        "Threshold recommendations saved to %s",
        output_dir / "threshold_recommendations.csv",
    )


if __name__ == "__main__":
    main()

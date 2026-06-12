"""Analyze cached Presto map-label metadata for balanced eval design."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEGMENTATION_TARGETS = {
    "worldcover",
    "openstreetmap_raster",
    "cdl",
    "worldcereal",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("analysis/pretrain_eval_label_metadata"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/presto_label_metadata_analysis"),
    )
    parser.add_argument("--dataset", default="presto")
    parser.add_argument("--rare-quantile", type=float, default=0.25)
    parser.add_argument("--top-k-tiles", type=int, default=50000)
    return parser.parse_args()


class LabelMetadata:
    """Sparse per-sample label/bin metadata."""

    def __init__(self, path: Path) -> None:
        """Load metadata from an NPZ cache."""
        data = np.load(path, allow_pickle=False)
        self.path = path
        self.sample_indices = data["sample_indices"]
        self.row_offsets = data["row_offsets"]
        self.label_ids = data["label_ids"]
        self.counts = data["counts"]
        self.finite_counts = data["finite_counts"]
        self.value_sums = data["value_sums"]
        self.value_sumsqs = data["value_sumsqs"]
        self.value_mins = data["value_mins"]
        self.value_maxs = data["value_maxs"]
        self.target = str(data["target"])
        self.kind = str(data["kind"])
        self.regression_bin_edges = data["regression_bin_edges"]

    def row_bounds(self, row_idx: int) -> tuple[int, int]:
        """Return sparse start/end offsets for one row."""
        return int(self.row_offsets[row_idx]), int(self.row_offsets[row_idx + 1])

    def row_labels_counts(self, row_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return label ids and counts for one sample row."""
        start, end = self.row_bounds(row_idx)
        return self.label_ids[start:end], self.counts[start:end]


def cache_files(metadata_dir: Path, dataset: str) -> list[Path]:
    """Return all target metadata files for a dataset."""
    return sorted(metadata_dir.glob(f"{dataset}_*_label_metadata.npz"))


def aggregate_counts(metadata: LabelMetadata) -> pd.DataFrame:
    """Aggregate global label/bin counts."""
    labels, counts = np.unique(metadata.label_ids, return_counts=False), []
    for label_id in labels:
        counts.append(int(metadata.counts[metadata.label_ids == label_id].sum()))
    df = pd.DataFrame({"label_id": labels.astype(int), "pixel_count": counts})
    total = df["pixel_count"].sum()
    df["pixel_fraction"] = df["pixel_count"] / total if total else 0.0
    return df.sort_values("pixel_count", ascending=False).reset_index(drop=True)


def regression_stats(metadata: LabelMetadata) -> dict[str, float]:
    """Compute global regression stats from cached moments."""
    count = int(metadata.finite_counts.sum())
    value_sum = float(metadata.value_sums.sum())
    value_sumsq = float(metadata.value_sumsqs.sum())
    mean = value_sum / count if count else float("nan")
    variance = value_sumsq / count - mean**2 if count else float("nan")
    return {
        "finite_count": count,
        "mean": mean,
        "std": float(np.sqrt(max(variance, 0.0))) if count else float("nan"),
        "min": float(np.nanmin(metadata.value_mins)),
        "max": float(np.nanmax(metadata.value_maxs)),
    }


def tile_interest_scores(
    metadata: LabelMetadata,
    global_counts: pd.DataFrame,
    rare_quantile: float,
) -> pd.DataFrame:
    """Rank segmentation tiles by class diversity and rare-class mass."""
    rare_cutoff = global_counts["pixel_count"].quantile(rare_quantile)
    rare_labels = set(
        global_counts.loc[global_counts["pixel_count"] <= rare_cutoff, "label_id"]
        .astype(int)
        .tolist()
    )

    rows = []
    for row_idx, sample_index in enumerate(metadata.sample_indices.tolist()):
        labels, counts = metadata.row_labels_counts(row_idx)
        valid_pixels = int(counts.sum())
        if valid_pixels == 0:
            continue
        fractions = counts / valid_pixels
        entropy = float(-(fractions * np.log2(fractions + 1e-12)).sum())
        max_entropy = float(np.log2(len(labels))) if len(labels) > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy else 0.0
        rare_mask = np.asarray([int(label) in rare_labels for label in labels])
        rare_fraction = float(counts[rare_mask].sum() / valid_pixels)
        top_class_fraction = float(fractions.max())
        score = (
            0.45 * normalized_entropy
            + 0.35 * rare_fraction
            + 0.20 * min(len(labels) / 8.0, 1.0)
        )
        rows.append(
            {
                "sample_index": int(sample_index),
                "valid_pixels": valid_pixels,
                "num_classes": int(len(labels)),
                "entropy": entropy,
                "normalized_entropy": normalized_entropy,
                "rare_fraction": rare_fraction,
                "top_class_fraction": top_class_fraction,
                "interesting_score": score,
                "labels": " ".join(str(int(label)) for label in labels.tolist()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["interesting_score", "num_classes", "valid_pixels"],
        ascending=[False, False, False],
    )


def plot_distribution(target: str, counts_df: pd.DataFrame, output_path: Path) -> None:
    """Plot global label/bin distribution."""
    fig, ax = plt.subplots(figsize=(max(8, len(counts_df) * 0.25), 5))
    sorted_df = counts_df.sort_values("label_id")
    ax.bar(sorted_df["label_id"].astype(str), sorted_df["pixel_count"])
    ax.set_title(f"Presto {target} global label distribution")
    ax.set_xlabel("Class id / regression bin id")
    ax.set_ylabel("Valid pixel count")
    ax.set_yscale("log")
    ax.tick_params(axis="x", labelrotation=90)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def recommendation(
    target: str,
    counts_df: pd.DataFrame,
    metadata: LabelMetadata,
    top_tiles_df: pd.DataFrame | None,
) -> dict:
    """Produce a compact balanced-eval recommendation."""
    present_labels = counts_df["label_id"].astype(int).tolist()
    tail_labels = (
        counts_df.sort_values("pixel_count")
        .head(max(1, len(counts_df) // 4))["label_id"]
        .astype(int)
        .tolist()
    )
    rec = {
        "target": target,
        "num_labels_or_bins": len(present_labels),
        "dominant_labels_or_bins": counts_df.head(10)["label_id"].astype(int).tolist(),
        "tail_labels_or_bins": tail_labels,
        "suggested_selection": (
            "Use per-class round-robin balanced sampling, but require class-diverse "
            "tiles for train/valid/test so the eval tests boundary/context reasoning."
        )
        if target in SEGMENTATION_TARGETS
        else (
            "Use fixed-bin balanced sampling over the regression bins, with separate "
            "reporting for zero/low and high-value bins."
        ),
    }
    if target in SEGMENTATION_TARGETS and top_tiles_df is not None:
        rec["interesting_tile_thresholds"] = {
            "min_num_classes": int(max(2, top_tiles_df["num_classes"].quantile(0.5))),
            "min_normalized_entropy": float(
                top_tiles_df["normalized_entropy"].quantile(0.25)
            ),
            "prefer_rare_fraction_at_least": float(
                top_tiles_df["rare_fraction"].quantile(0.25)
            ),
        }
    if metadata.kind == "regression":
        rec["regression_stats"] = regression_stats(metadata)
    return rec


def main() -> None:
    """Analyze Presto label metadata caches."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = cache_files(args.metadata_dir, args.dataset)
    if not files:
        raise FileNotFoundError(
            f"No metadata caches found for dataset '{args.dataset}' in {args.metadata_dir}"
        )

    recommendations = []
    for path in files:
        metadata = LabelMetadata(path)
        target = metadata.target
        counts_df = aggregate_counts(metadata)
        counts_path = (
            args.output_dir / f"{args.dataset}_{target}_global_distribution.csv"
        )
        counts_df.to_csv(counts_path, index=False)
        plot_distribution(
            target=target,
            counts_df=counts_df,
            output_path=args.output_dir
            / f"{args.dataset}_{target}_global_distribution.png",
        )

        top_tiles_df = None
        if target in SEGMENTATION_TARGETS:
            scores_df = tile_interest_scores(
                metadata=metadata,
                global_counts=counts_df,
                rare_quantile=args.rare_quantile,
            )
            top_tiles_df = scores_df.head(args.top_k_tiles)
            top_tiles_df.to_csv(
                args.output_dir / f"{args.dataset}_{target}_interesting_tiles.csv",
                index=False,
            )

        recommendations.append(
            recommendation(
                target=target,
                counts_df=counts_df,
                metadata=metadata,
                top_tiles_df=top_tiles_df,
            )
        )

    rec_path = args.output_dir / f"{args.dataset}_balanced_eval_recommendations.json"
    rec_path.write_text(json.dumps(recommendations, indent=2) + "\n")
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()

"""Test whether each embedding dimension follows a Gaussian distribution.

Metrics computed per dimension:
  1. Skewness           -- 0 for a perfect Gaussian
  2. Excess kurtosis    -- 0 for a perfect Gaussian
  3. Q-Q correlation    -- 1.0 for a perfect Gaussian

Each dimension is classified into one of three buckets based on configurable
strict / relaxed thresholds applied to all three metrics simultaneously:
  - strict pass   (all metrics within strict bounds)
  - relaxed pass  (all metrics within relaxed bounds but not strict)
  - fail          (at least one metric outside relaxed bounds)

Usage:
  python test_gaussianity.py --input-dir /path/to/embeddings
  python test_gaussianity.py --input-dir /path/to/embeddings --output-csv results.csv --plot hist.png
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from cluster_embeddings import load_embeddings

# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_skewness_kurtosis(
    embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-dimension skewness and excess kurtosis (both shape (D,))."""
    print("[metrics] Computing skewness and excess kurtosis ...")
    t0 = time.time()
    skew = stats.skew(embeddings, axis=0)
    kurt = stats.kurtosis(embeddings, axis=0)  # excess kurtosis by default
    elapsed = time.time() - t0
    print(f"[metrics] Skewness & kurtosis done in {elapsed:.1f}s")
    return skew, kurt


def _qq_corr_single(col: np.ndarray) -> float:
    """Q-Q correlation coefficient for a single 1-D array."""
    _, (_, _, r) = stats.probplot(col)
    return r


def compute_qq_correlations(
    embeddings: np.ndarray,
    max_workers: int = 16,
) -> np.ndarray:
    """Return per-dimension Q-Q correlation coefficients (shape (D,))."""
    n_dims = embeddings.shape[1]
    print(
        f"[metrics] Computing Q-Q correlations for {n_dims} dims "
        f"(workers={max_workers}) ..."
    )
    t0 = time.time()

    cols = [embeddings[:, d] for d in range(n_dims)]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        qq_corrs = np.array(list(pool.map(_qq_corr_single, cols)), dtype=np.float64)

    elapsed = time.time() - t0
    print(f"[metrics] Q-Q correlations done in {elapsed:.1f}s")
    return qq_corrs


# ---------------------------------------------------------------------------
# Threshold evaluation
# ---------------------------------------------------------------------------


def evaluate_thresholds(
    skew: np.ndarray,
    kurt: np.ndarray,
    qq_corr: np.ndarray,
    strict_skew: float,
    strict_kurt: float,
    strict_qq: float,
    relaxed_skew: float,
    relaxed_kurt: float,
    relaxed_qq: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify each dimension as strict-pass / relaxed-pass / fail.

    Returns:
        strict_pass: bool array (D,)
        relaxed_pass: bool array (D,)
    """
    strict_pass = (
        (np.abs(skew) < strict_skew)
        & (np.abs(kurt) < strict_kurt)
        & (qq_corr > strict_qq)
    )
    relaxed_pass = (
        (np.abs(skew) < relaxed_skew)
        & (np.abs(kurt) < relaxed_kurt)
        & (qq_corr > relaxed_qq)
    )
    return strict_pass, relaxed_pass


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _stat_line(name: str, values: np.ndarray) -> str:
    return (
        f"  {name:>22s}:  min={values.min():.6f}  max={values.max():.6f}  "
        f"mean={values.mean():.6f}  median={np.median(values):.6f}"
    )


def print_summary(
    skew: np.ndarray,
    kurt: np.ndarray,
    qq_corr: np.ndarray,
    strict_pass: np.ndarray,
    relaxed_pass: np.ndarray,
    strict_thresholds: tuple[float, float, float],
    relaxed_thresholds: tuple[float, float, float],
) -> None:
    """Print aggregate Gaussianity statistics and pass-rate summaries."""
    n_dims = len(skew)
    s_skew, s_kurt, s_qq = strict_thresholds
    r_skew, r_kurt, r_qq = relaxed_thresholds

    print("\n" + "=" * 72)
    print("  GAUSSIANITY TEST SUMMARY")
    print("=" * 72)

    print(f"\n  Dimensions: {n_dims}")
    print()
    print("  Per-metric statistics across all dimensions:")
    print(_stat_line("|skewness|", np.abs(skew)))
    print(_stat_line("|excess kurtosis|", np.abs(kurt)))
    print(_stat_line("Q-Q correlation", qq_corr))

    n_strict_skew = int((np.abs(skew) < s_skew).sum())
    n_strict_kurt = int((np.abs(kurt) < s_kurt).sum())
    n_strict_qq = int((qq_corr > s_qq).sum())
    n_relaxed_skew = int((np.abs(skew) < r_skew).sum())
    n_relaxed_kurt = int((np.abs(kurt) < r_kurt).sum())
    n_relaxed_qq = int((qq_corr > r_qq).sum())

    print("\n  Individual metric pass rates:")
    print(f"    {'Metric':<22s}  {'Strict':>12s}  {'Relaxed':>12s}")
    print(f"    {'-' * 22}  {'-' * 12}  {'-' * 12}")
    print(
        f"    {'|skewness|':<22s}  "
        f"{n_strict_skew:>5d}/{n_dims:<5d}  "
        f"{n_relaxed_skew:>5d}/{n_dims:<5d}"
    )
    print(
        f"    {'|excess kurtosis|':<22s}  "
        f"{n_strict_kurt:>5d}/{n_dims:<5d}  "
        f"{n_relaxed_kurt:>5d}/{n_dims:<5d}"
    )
    print(
        f"    {'Q-Q correlation':<22s}  "
        f"{n_strict_qq:>5d}/{n_dims:<5d}  "
        f"{n_relaxed_qq:>5d}/{n_dims:<5d}"
    )

    n_strict = int(strict_pass.sum())
    n_relaxed = int(relaxed_pass.sum())
    n_fail = int((~relaxed_pass).sum())
    n_relaxed_only = n_relaxed - n_strict

    print("\n  Combined (all 3 metrics must pass):")
    print(
        f"    Strict pass  (<|skew|{s_skew}, <|kurt|{s_kurt}, QQ>{s_qq}):  "
        f"{n_strict:>5d}/{n_dims}  ({n_strict / n_dims:.1%})"
    )
    print(
        f"    Relaxed pass (<|skew|{r_skew}, <|kurt|{r_kurt}, QQ>{r_qq}):  "
        f"{n_relaxed:>5d}/{n_dims}  ({n_relaxed / n_dims:.1%})"
    )
    print(
        f"    Relaxed only (pass relaxed, fail strict):  "
        f"{n_relaxed_only:>5d}/{n_dims}  ({n_relaxed_only / n_dims:.1%})"
    )
    print(
        f"    Fail (outside relaxed):  {n_fail:>5d}/{n_dims}  ({n_fail / n_dims:.1%})"
    )
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def save_csv(
    path: str,
    skew: np.ndarray,
    kurt: np.ndarray,
    qq_corr: np.ndarray,
    strict_pass: np.ndarray,
    relaxed_pass: np.ndarray,
) -> None:
    """Write per-dimension Gaussianity metrics to CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dim",
                "skewness",
                "excess_kurtosis",
                "qq_corr",
                "strict_pass",
                "relaxed_pass",
            ]
        )
        for d in range(len(skew)):
            writer.writerow(
                [
                    d,
                    f"{skew[d]:.6f}",
                    f"{kurt[d]:.6f}",
                    f"{qq_corr[d]:.6f}",
                    int(strict_pass[d]),
                    int(relaxed_pass[d]),
                ]
            )
    print(f"[save] Per-dimension CSV -> {path}")


# ---------------------------------------------------------------------------
# Histogram plot
# ---------------------------------------------------------------------------


def save_plot(
    path: str,
    skew: np.ndarray,
    kurt: np.ndarray,
    qq_corr: np.ndarray,
    strict_thresholds: tuple[float, float, float],
    relaxed_thresholds: tuple[float, float, float],
) -> None:
    """Save histogram plots for skew, kurtosis, and Q-Q correlation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s_skew, s_kurt, s_qq = strict_thresholds
    r_skew, r_kurt, r_qq = relaxed_thresholds

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # -- |Skewness| --
    ax = axes[0]
    ax.hist(np.abs(skew), bins=60, edgecolor="black", linewidth=0.3)
    ax.axvline(s_skew, color="green", ls="--", lw=1.5, label=f"strict ({s_skew})")
    ax.axvline(r_skew, color="orange", ls="--", lw=1.5, label=f"relaxed ({r_skew})")
    ax.set_xlabel("|Skewness|")
    ax.set_ylabel("# dimensions")
    ax.set_title("|Skewness| across dimensions")
    ax.legend()

    # -- |Excess Kurtosis| --
    ax = axes[1]
    ax.hist(np.abs(kurt), bins=60, edgecolor="black", linewidth=0.3)
    ax.axvline(s_kurt, color="green", ls="--", lw=1.5, label=f"strict ({s_kurt})")
    ax.axvline(r_kurt, color="orange", ls="--", lw=1.5, label=f"relaxed ({r_kurt})")
    ax.set_xlabel("|Excess Kurtosis|")
    ax.set_ylabel("# dimensions")
    ax.set_title("|Excess Kurtosis| across dimensions")
    ax.legend()

    # -- Q-Q Correlation --
    ax = axes[2]
    ax.hist(qq_corr, bins=60, edgecolor="black", linewidth=0.3)
    ax.axvline(s_qq, color="green", ls="--", lw=1.5, label=f"strict ({s_qq})")
    ax.axvline(r_qq, color="orange", ls="--", lw=1.5, label=f"relaxed ({r_qq})")
    ax.set_xlabel("Q-Q Correlation")
    ax.set_ylabel("# dimensions")
    ax.set_title("Q-Q Correlation across dimensions")
    ax.legend()

    fig.suptitle("Per-dimension Gaussianity Metrics", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] Histogram plot -> {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for Gaussianity analysis."""
    p = argparse.ArgumentParser(
        description="Test per-dimension Gaussianity of sharded embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input-dir", required=True, help="Directory containing shard_*.npz files"
    )
    p.add_argument(
        "--subsample",
        type=int,
        default=0,
        help="Randomly subsample to N points (0 = all)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Thread-pool workers for Q-Q computation",
    )
    p.add_argument(
        "--output-csv", default=None, help="Path to write per-dimension CSV results"
    )
    p.add_argument("--plot", default=None, help="Path to save histogram plot (PNG/PDF)")

    g = p.add_argument_group("strict thresholds")
    g.add_argument("--strict-skew", type=float, default=0.5)
    g.add_argument("--strict-kurt", type=float, default=1.0)
    g.add_argument("--strict-qq", type=float, default=0.999)

    g = p.add_argument_group("relaxed thresholds")
    g.add_argument("--relaxed-skew", type=float, default=1.0)
    g.add_argument("--relaxed-kurt", type=float, default=2.0)
    g.add_argument("--relaxed-qq", type=float, default=0.995)

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run Gaussianity metrics, summaries, and optional artifact export."""
    args = parse_args(argv)

    embeddings = load_embeddings(args.input_dir, args.subsample, args.seed)

    skew, kurt = compute_skewness_kurtosis(embeddings)
    qq_corr = compute_qq_correlations(embeddings, max_workers=args.workers)

    strict_thresholds = (args.strict_skew, args.strict_kurt, args.strict_qq)
    relaxed_thresholds = (args.relaxed_skew, args.relaxed_kurt, args.relaxed_qq)

    strict_pass, relaxed_pass = evaluate_thresholds(
        skew,
        kurt,
        qq_corr,
        *strict_thresholds,
        *relaxed_thresholds,
    )

    print_summary(
        skew,
        kurt,
        qq_corr,
        strict_pass,
        relaxed_pass,
        strict_thresholds,
        relaxed_thresholds,
    )

    if args.output_csv:
        save_csv(args.output_csv, skew, kurt, qq_corr, strict_pass, relaxed_pass)

    if args.plot:
        save_plot(args.plot, skew, kurt, qq_corr, strict_thresholds, relaxed_thresholds)


if __name__ == "__main__":
    main()

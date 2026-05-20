r"""Greedy max-score with similarity-exclusion diversification of acquisition scores.

For each of the 5 acquisition strategies already present in
``combined_acquisition_scores.parquet`` (novelty, xglobal_bridge,
sparse_infill, xlocal_bridge, prototypes), this script produces a
diversity-filtered version of the per-strategy *_normalized_score* column:

1. Candidates are sorted descending by the strategy's normalized score.
2. We iterate top-down. A candidate is accepted only if its cosine
   similarity to every previously accepted candidate **in the same parent
   cluster** is strictly below a per-parent threshold tau_parent[c].
3. Rejected candidates get a sentinel value (default ``-999.0``) so they
   fall to the bottom of any downstream top-K ranking.

The threshold tau_parent[c] is calibrated from the frozen reference set:
for each reference point in parent c we compute its nearest-neighbor
cosine similarity to other reference points **inside the same parent**.
``tau_parent[c] = percentile(nnsim_c, p)``. Lower ``p`` = stricter filter.

The new diverse scores are written back into
``combined_acquisition_scores.parquet`` in place, as additional columns
named ``<strategy>_diverse_score_p<percentile>``. A summary JSON next to
the parquet records the per-parent thresholds, acceptance counts, and
the original-ranking positions at which the N-th globally accepted
candidate landed (defaults: 50,000 and 100,000) so the operator can see
whether the chosen percentile is too aggressive or too light.

Usage
-----

Moderate diversity filter (recommended starting point)::

    export OPENBLAS_NUM_THREADS=16
    export OMP_NUM_THREADS=16
    python olmoearth_pretrain/dataset_creation/candidate_embeddings/diversify_acquisition.py \
      --input-dir "${RSLEARN_EAI_ROOT}/datasets/globe_land_grid/s50ix24_embeddings" \
      --reference-dir /weka/dfive-default/hadriens/oe_inst_embeddings_ps8_shwp12_4s_s2_ixes/_scores \
      --combined-parquet "${RSLEARN_EAI_ROOT}/datasets/globe_land_grid/s50ix24_embeddings/_scores/combined_acquisition_scores.parquet" \
      --percentile 95 \
      --report-marks 50000 100000

Force recompute of the per-parent reference NN-sim cache (e.g. after
changing the reference set)::

    export OPENBLAS_NUM_THREADS=16
    export OMP_NUM_THREADS=16
    python olmoearth_pretrain/dataset_creation/candidate_embeddings/diversify_acquisition.py \
      --input-dir "${RSLEARN_EAI_ROOT}/datasets/globe_land_grid/s50ix24_embeddings" \
      --reference-dir /weka/dfive-default/hadriens/oe_inst_embeddings_ps8_shwp12_4s_s2_ixes/_scores \
      --combined-parquet "${RSLEARN_EAI_ROOT}/datasets/globe_land_grid/s50ix24_embeddings/_scores/combined_acquisition_scores.parquet" \
      --percentile 95 \
      --recompute-nnsim

Notes:
-----
1. Running the script multiple times with different ``--percentile`` values
   accumulates side-by-side columns in the same parquet
   (``..._diverse_score_p90`` / ``_p95`` / ``_p98`` ...). Old columns are
   never removed.
2. The cached per-parent reference NN-sim distribution
   (``reference_parent_nnsim.npz`` under ``--reference-dir``) is reused
   across percentile runs, so only the first run pays the NN-sim cost.
3. Set ``OPENBLAS_NUM_THREADS`` and ``OMP_NUM_THREADS`` before running
   (machines with >128 cores will segfault otherwise).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import TypedDict

import numpy as np
import pandas as pd
from reference_model import (
    EPS,
    load_reference_artifacts,
    normalize_rows,
    save_summary_json,
)
from score_acquisition import load_candidate_state

STRATEGIES = (
    "novelty",
    "xglobal_bridge",
    "sparse_infill",
    "xlocal_bridge",
    "prototypes",
)

NORMALIZED_SCORE_TEMPLATE = "{strategy}_normalized_score"
DIVERSE_SCORE_TEMPLATE = "{strategy}_diverse_score_{percentile_tag}"

REFERENCE_NNSIM_FILENAME = "reference_parent_nnsim.npz"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _percentile_tag(percentile: float) -> str:
    """Return a parquet-column-friendly tag for the percentile.

    95.0 -> 'p95'
    95.5 -> 'p95_5'
    """
    if float(percentile) == float(int(percentile)):
        return f"p{int(percentile)}"
    return ("p" + repr(float(percentile))).replace(".", "_")


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization with EPS-stabilization (float32)."""
    return normalize_rows(matrix.astype(np.float32, copy=False)).astype(
        np.float32, copy=False
    )


# ---------------------------------------------------------------------------
# Per-parent reference nearest-neighbor cosine similarity distribution
# ---------------------------------------------------------------------------


def _per_parent_nnsim(
    points_normalized: np.ndarray,
    block: int = 4096,
) -> np.ndarray:
    """Top-1 cosine similarity of each row to every other row (no self).

    Computed in blocks to bound memory. Returns a 1D float32 array.
    """
    n = points_normalized.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    if n == 1:
        return np.zeros(1, dtype=np.float32)

    nnsim = np.full(n, -np.inf, dtype=np.float32)
    for start in range(0, n, block):
        stop = min(start + block, n)
        sims = points_normalized[start:stop] @ points_normalized.T
        rows = np.arange(start, stop)
        sims[np.arange(stop - start), rows] = -np.inf
        nnsim[start:stop] = sims.max(axis=1).astype(np.float32)
    nnsim[~np.isfinite(nnsim)] = 0.0
    return nnsim


def compute_reference_parent_nnsim(
    parent_centroids: np.ndarray,
    residuals_by_parent: dict[int, np.ndarray],
    cache_path: str,
    force: bool = False,
) -> dict[int, np.ndarray]:
    """Return per-parent NN cosine sim distributions, caching to disk."""
    if (not force) and os.path.exists(cache_path):
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                cached: dict[int, np.ndarray] = {}
                ok = True
                for parent_idx in range(parent_centroids.shape[0]):
                    key = f"parent_{parent_idx:03d}_nnsim"
                    if key not in data.files:
                        ok = False
                        break
                    expected_n = int(
                        residuals_by_parent.get(
                            parent_idx, np.zeros((0, 0), dtype=np.float32)
                        ).shape[0]
                    )
                    if data[key].shape[0] != expected_n:
                        ok = False
                        break
                    cached[parent_idx] = data[key].astype(np.float32, copy=False)
            if ok:
                print(f"[cache] Reusing reference NN-sim cache -> {cache_path}")
                return cached
            print(f"[cache] Stale NN-sim cache at {cache_path}; recomputing")
        except Exception as exc:
            print(f"[cache] Failed to read {cache_path} ({exc}); recomputing")

    print(
        f"[nnsim] Computing per-parent reference NN cosine sim for "
        f"{parent_centroids.shape[0]} parents ..."
    )
    nnsim_by_parent: dict[int, np.ndarray] = {}
    for parent_idx in range(parent_centroids.shape[0]):
        residuals = residuals_by_parent.get(parent_idx)
        if residuals is None or residuals.shape[0] == 0:
            nnsim_by_parent[parent_idx] = np.zeros(0, dtype=np.float32)
            print(f"[nnsim]   parent {parent_idx:02d}: 0 points")
            continue
        t0 = time.time()
        pca_points = (residuals + parent_centroids[parent_idx]).astype(
            np.float32, copy=False
        )
        normalized = _l2_normalize(pca_points)
        nnsim = _per_parent_nnsim(normalized)
        elapsed = time.time() - t0
        nnsim_by_parent[parent_idx] = nnsim
        print(
            f"[nnsim]   parent {parent_idx:02d}: "
            f"n={nnsim.shape[0]:>7,d}  "
            f"median={np.median(nnsim):.4f}  "
            f"p90={np.percentile(nnsim, 90):.4f}  "
            f"p95={np.percentile(nnsim, 95):.4f}  "
            f"p98={np.percentile(nnsim, 98):.4f}  "
            f"({elapsed:.1f}s)"
        )

    payload = {
        f"parent_{parent_idx:03d}_nnsim": arr
        for parent_idx, arr in nnsim_by_parent.items()
    }
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.savez(cache_path, **payload)
    print(f"[save] Reference NN-sim cache -> {cache_path}")
    return nnsim_by_parent


def per_parent_thresholds(
    nnsim_by_parent: dict[int, np.ndarray],
    percentile: float,
    n_clusters: int,
) -> np.ndarray:
    """tau_parent[c] = np.percentile(nnsim_c, percentile)."""
    thresholds = np.zeros(n_clusters, dtype=np.float32)
    for parent_idx in range(n_clusters):
        nnsim = nnsim_by_parent.get(parent_idx)
        if nnsim is None or nnsim.size == 0:
            thresholds[parent_idx] = 1.0
            continue
        thresholds[parent_idx] = float(np.percentile(nnsim, percentile))
    return thresholds


# ---------------------------------------------------------------------------
# Greedy max-score with similarity exclusion
# ---------------------------------------------------------------------------


class _ParentBuffer:
    """Doubling-capacity row buffer of accepted normalized vectors for one parent."""

    __slots__ = ("dim", "count", "_capacity", "_data")

    def __init__(self, dim: int, initial_capacity: int = 64) -> None:
        self.dim = int(dim)
        self.count = 0
        self._capacity = max(1, int(initial_capacity))
        self._data = np.empty((self._capacity, self.dim), dtype=np.float32)

    def view(self) -> np.ndarray:
        return self._data[: self.count]

    def append(self, row: np.ndarray) -> None:
        if self.count == self._capacity:
            new_capacity = self._capacity * 2
            new_data = np.empty((new_capacity, self.dim), dtype=np.float32)
            new_data[: self.count] = self._data[: self.count]
            self._data = new_data
            self._capacity = new_capacity
        self._data[self.count] = row
        self.count += 1


class DiversifyResult(TypedDict):
    """Return type for greedy_diversify_strategy."""

    diverse_scores: np.ndarray
    accepted_total: int
    rejected_total: int
    accepted_per_parent: list[int]
    report_marks: dict[int, int | None]


class StrategySummary(TypedDict):
    """Per-strategy section of the output summary."""

    score_column: str
    diverse_column: str
    accepted_total: int
    rejected_total: int
    accepted_per_parent: list[int]
    report_marks_original_rank: dict[str, int | None]


def greedy_diversify_strategy(
    normalized_scores: np.ndarray,
    parent_labels: np.ndarray,
    candidates_normalized: np.ndarray,
    tau: np.ndarray,
    n_clusters: int,
    rejected_value: float,
    report_marks: list[int],
    strategy_name: str,
) -> DiversifyResult:
    """Iterate candidates desc by score; reject if cosine sim >= tau.

    Returns the diverse score array (same length and order as the inputs)
    plus diagnostic metadata (per-parent acceptance, original-rank marks).
    """
    n = normalized_scores.shape[0]
    if n == 0:
        return {
            "diverse_scores": np.zeros(0, dtype=np.float32),
            "accepted_total": 0,
            "rejected_total": 0,
            "accepted_per_parent": [0] * n_clusters,
            "report_marks": {int(mark): None for mark in report_marks},
        }

    sorted_idx = np.argsort(-normalized_scores, kind="stable").astype(np.int64)
    diverse_scores = np.full(n, np.float32(rejected_value), dtype=np.float32)
    accepted_per_parent = np.zeros(n_clusters, dtype=np.int64)
    buffers: dict[int, _ParentBuffer] = {}
    dim = candidates_normalized.shape[1]

    report_marks_sorted = sorted({int(m) for m in report_marks if int(m) > 0})
    mark_to_original_rank: dict[int, int | None] = {
        m: None for m in report_marks_sorted
    }
    next_mark_idx = 0

    t0 = time.time()
    accepted_total = 0
    log_every = max(1, n // 20)
    for rank_in_sorted, cand_idx in enumerate(sorted_idx):
        parent = int(parent_labels[cand_idx])
        cand_vec = candidates_normalized[cand_idx]
        buf = buffers.get(parent)
        if buf is None:
            buf = _ParentBuffer(dim)
            buffers[parent] = buf

        accept: bool
        if buf.count == 0:
            accept = True
        else:
            sims = buf.view() @ cand_vec
            max_sim = float(sims.max())
            accept = max_sim < float(tau[parent])

        if accept:
            buf.append(cand_vec)
            diverse_scores[cand_idx] = np.float32(normalized_scores[cand_idx])
            accepted_per_parent[parent] += 1
            accepted_total += 1
            while (
                next_mark_idx < len(report_marks_sorted)
                and accepted_total == report_marks_sorted[next_mark_idx]
            ):
                mark = report_marks_sorted[next_mark_idx]
                mark_to_original_rank[mark] = rank_in_sorted + 1
                next_mark_idx += 1

        if (rank_in_sorted + 1) % log_every == 0:
            elapsed = time.time() - t0
            print(
                f"[diversify:{strategy_name}]   "
                f"processed {rank_in_sorted + 1:>9,d}/{n:,d}  "
                f"accepted={accepted_total:>8,d}  "
                f"({elapsed:.1f}s)"
            )

    rejected_total = int(n - accepted_total)
    elapsed = time.time() - t0
    print(
        f"[diversify:{strategy_name}] done in {elapsed:.1f}s  "
        f"accepted={accepted_total:,d}  rejected={rejected_total:,d}  "
        f"({accepted_total / max(n, 1):.1%} accepted)"
    )

    return {
        "diverse_scores": diverse_scores,
        "accepted_total": int(accepted_total),
        "rejected_total": int(rejected_total),
        "accepted_per_parent": accepted_per_parent.tolist(),
        "report_marks": mark_to_original_rank,
    }


# ---------------------------------------------------------------------------
# Parquet update
# ---------------------------------------------------------------------------


def add_diverse_columns_inplace(
    parquet_path: str,
    new_columns: dict[str, np.ndarray],
) -> None:
    """Atomically add columns to the parquet via tmp file + os.replace."""
    if not new_columns:
        return
    df = pd.read_parquet(parquet_path)
    for column, values in new_columns.items():
        if values.shape[0] != len(df):
            raise ValueError(
                f"Column '{column}' length {values.shape[0]} != {len(df)} parquet rows"
            )
        df[column] = values.astype(np.float32, copy=False)
    tmp_path = parquet_path + ".tmp"
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, parquet_path)
    print(
        f"[save] Added {len(new_columns)} column(s) to {parquet_path}: "
        f"{sorted(new_columns.keys())}"
    )


# ---------------------------------------------------------------------------
# Alignment / validation
# ---------------------------------------------------------------------------


def assert_combined_aligned(
    combined_df: pd.DataFrame,
    candidate_parent_labels: np.ndarray,
    candidate_sample_indices: np.ndarray,
) -> None:
    """Sanity-check the combined parquet aligns row-for-row with candidate state."""
    n = len(combined_df)
    if candidate_parent_labels.shape[0] != n:
        raise ValueError(
            f"Combined parquet has {n} rows but candidate state has "
            f"{candidate_parent_labels.shape[0]} rows"
        )
    parq_parent = combined_df["parent_label"].to_numpy(dtype=np.int32)
    if not np.array_equal(parq_parent, candidate_parent_labels.astype(np.int32)):
        raise ValueError(
            "parent_label mismatch between combined parquet and recomputed "
            "candidate state. Were they produced from the same input_dir / "
            "reference set?"
        )
    parq_sample = combined_df["sample_idx"].to_numpy(dtype=np.int64)
    if not np.array_equal(parq_sample, candidate_sample_indices.astype(np.int64)):
        raise ValueError(
            "sample_idx ordering mismatch between combined parquet and candidate state"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for diversity-filtered acquisition scoring."""
    parser = argparse.ArgumentParser(
        description=(
            "Greedy max-score with similarity exclusion. Adds a diverse "
            "per-strategy score column to combined_acquisition_scores.parquet."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Candidate directory with shard_*.npz (the embeddings to score).",
    )
    parser.add_argument(
        "--reference-dir",
        required=True,
        help="Directory containing reference artifacts from reference_model.py.",
    )
    parser.add_argument(
        "--combined-parquet",
        required=True,
        help=(
            "Path to combined_acquisition_scores.parquet. New columns are added "
            "to this file in place (atomic tmp+replace)."
        ),
    )
    parser.add_argument(
        "--percentile",
        type=float,
        required=True,
        help=(
            "Per-parent percentile of reference NN cosine sim used as the "
            "diversity threshold. 90=stronger, 95=moderate, 98=light filter."
        ),
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=list(STRATEGIES),
        choices=list(STRATEGIES),
        help="Subset of strategies to diversify.",
    )
    parser.add_argument(
        "--report-marks",
        nargs="+",
        type=int,
        default=[50000, 100000],
        help=(
            "Report the original sorted-order rank where the N-th globally "
            "accepted sample landed (one or more positive integers)."
        ),
    )
    parser.add_argument(
        "--rejected-value",
        type=float,
        default=-999.0,
        help="Sentinel value written for rejected (redundant) candidates.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed (used by load_candidate_state; selection is deterministic).",
    )
    parser.add_argument(
        "--recompute-nnsim",
        action="store_true",
        help="Force recomputation of the reference per-parent NN-sim cache.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Run the diversification pipeline for the requested strategies."""
    args = parse_args(argv)

    if not 0.0 <= args.percentile <= 100.0:
        raise ValueError(f"--percentile must be in [0, 100], got {args.percentile}")
    tag = _percentile_tag(args.percentile)
    print(f"[args] percentile={args.percentile} (tag={tag})")

    print(
        "[load] Loading candidate state (embeddings + frozen PCA + parent labels) ..."
    )
    state = load_candidate_state(args.input_dir, args.reference_dir, args.seed)
    model = state["model"]
    n_clusters = int(model["n_clusters"].item())
    parent_labels: np.ndarray = state["parent_labels"]
    pca_data: np.ndarray = state["pca_data"]
    index_rows = state["index_rows"]
    sample_indices = np.array(
        [int(row.get("sample_idx", i)) for i, row in enumerate(index_rows)],
        dtype=np.int64,
    )

    print("[load] Loading reference artifacts (residuals + parent centroids) ...")
    _model_ref, residuals_by_parent = load_reference_artifacts(args.reference_dir)

    cache_path = os.path.join(args.reference_dir, REFERENCE_NNSIM_FILENAME)
    nnsim_by_parent = compute_reference_parent_nnsim(
        parent_centroids=model["parent_centroids"],
        residuals_by_parent=residuals_by_parent,
        cache_path=cache_path,
        force=bool(args.recompute_nnsim),
    )

    tau = per_parent_thresholds(
        nnsim_by_parent, percentile=args.percentile, n_clusters=n_clusters
    )
    print("[tau] Per-parent diversity thresholds:")
    for parent_idx in range(n_clusters):
        n_ref = int(nnsim_by_parent.get(parent_idx, np.zeros(0)).shape[0])
        print(
            f"[tau]   parent {parent_idx:02d}:  tau={tau[parent_idx]:.4f}  "
            f"(reference n={n_ref:,d})"
        )

    print("[normalize] L2-normalizing candidate PCA vectors ...")
    candidates_normalized = _l2_normalize(pca_data)

    print(f"[load] Reading combined parquet: {args.combined_parquet}")
    combined_df = pd.read_parquet(args.combined_parquet)
    assert_combined_aligned(combined_df, parent_labels, sample_indices)

    per_strategy_summary: dict[str, StrategySummary] = {}
    new_columns: dict[str, np.ndarray] = {}

    for strategy in args.strategies:
        score_col = NORMALIZED_SCORE_TEMPLATE.format(strategy=strategy)
        if score_col not in combined_df.columns:
            raise ValueError(
                f"Combined parquet missing required column '{score_col}' for "
                f"strategy '{strategy}'."
            )
        normalized_scores = combined_df[score_col].to_numpy(dtype=np.float32)
        diverse_col = DIVERSE_SCORE_TEMPLATE.format(
            strategy=strategy, percentile_tag=tag
        )
        print(
            f"[diversify:{strategy}] sorting by '{score_col}' and applying "
            f"greedy diversification ..."
        )
        result = greedy_diversify_strategy(
            normalized_scores=normalized_scores,
            parent_labels=parent_labels.astype(np.int32, copy=False),
            candidates_normalized=candidates_normalized,
            tau=tau,
            n_clusters=n_clusters,
            rejected_value=float(args.rejected_value),
            report_marks=list(args.report_marks),
            strategy_name=strategy,
        )
        new_columns[diverse_col] = result["diverse_scores"]
        per_strategy_summary[strategy] = StrategySummary(
            score_column=score_col,
            diverse_column=diverse_col,
            accepted_total=result["accepted_total"],
            rejected_total=result["rejected_total"],
            accepted_per_parent=result["accepted_per_parent"],
            report_marks_original_rank={
                str(mark): (None if rank is None else rank)
                for mark, rank in result["report_marks"].items()
            },
        )

    print(f"[save] Updating combined parquet in place: {args.combined_parquet}")
    add_diverse_columns_inplace(args.combined_parquet, new_columns)

    summary = {
        "input_dir": args.input_dir,
        "reference_dir": args.reference_dir,
        "combined_parquet": args.combined_parquet,
        "percentile": float(args.percentile),
        "percentile_tag": tag,
        "rejected_value": float(args.rejected_value),
        "report_marks": [int(m) for m in args.report_marks],
        "candidate_size": int(pca_data.shape[0]),
        "n_clusters": int(n_clusters),
        "tau_per_parent": tau.astype(float).tolist(),
        "reference_count_per_parent": [
            int(nnsim_by_parent.get(c, np.zeros(0)).shape[0]) for c in range(n_clusters)
        ],
        "strategies": per_strategy_summary,
    }
    summary_path = os.path.join(
        os.path.dirname(os.path.abspath(args.combined_parquet)),
        f"diverse_acquisition_summary_{tag}.json",
    )
    save_summary_json(summary_path, summary)

    print("[report] Greedy diversification summary:")
    for strategy, info in per_strategy_summary.items():
        marks_str = ", ".join(
            f"{m}@orig_rank={r}" for m, r in info["report_marks_original_rank"].items()
        )
        print(
            f"[report]   {strategy:>16s}  accepted={info['accepted_total']:>8,d}  "
            f"rejected={info['rejected_total']:>9,d}  marks: {marks_str}"
        )

    _ = EPS  # silence import-only lints if any


if __name__ == "__main__":
    main(sys.argv[1:])

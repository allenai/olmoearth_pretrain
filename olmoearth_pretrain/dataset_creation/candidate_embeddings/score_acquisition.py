"""Score candidate embeddings with multiple acquisition strategies.

This CLI reuses the frozen reference artifacts produced by
``reference_model.py`` and exposes additional acquisition signals beyond
novelty:

- xglobal_bridge
- sparse-infill
- xlocal_bridge
- prototypes
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import TypedDict

import acquisition_strategies as strat
import numpy as np
import pandas as pd
from reference_model import (
    assign_spherical_kmeans,
    compute_residuals,
    ensure_index_rows,
    load_embeddings,
    load_reference_artifacts,
    save_summary_json,
    transform_with_pca,
)


class CandidateState(TypedDict):
    """Typed container for candidate embeddings plus frozen reference artifacts."""

    model: dict[str, np.ndarray]
    residuals_by_parent: dict[int, np.ndarray]
    index_rows: list[dict[str, str]]
    pca_data: np.ndarray
    parent_labels: np.ndarray
    residuals: np.ndarray


def load_candidate_state(
    input_dir: str,
    reference_dir: str,
    seed: int,
) -> CandidateState:
    """Load candidate embeddings and frozen reference artifacts."""
    model, residuals_by_parent = load_reference_artifacts(reference_dir)
    embeddings, kept_indices = load_embeddings(input_dir, 0, seed)
    if kept_indices is not None:
        raise AssertionError(
            "Acquisition scoring does not support subsampling candidates"
        )
    index_rows = ensure_index_rows(input_dir, embeddings.shape[0], None)

    pca_data = transform_with_pca(
        embeddings,
        model["pca_mean"],
        model["pca_components"],
    )
    parent_labels = assign_spherical_kmeans(pca_data, model["normalized_centers"])
    residuals = compute_residuals(pca_data, parent_labels, model["parent_centroids"])

    return {
        "model": model,
        "residuals_by_parent": residuals_by_parent,
        "index_rows": index_rows,
        "pca_data": pca_data,
        "parent_labels": parent_labels,
        "residuals": residuals,
    }


def _strategy_output_prefix(strategy: str) -> str:
    return strategy.replace("-", "_")


def _coerce_float(value: object) -> float:
    if value is None or value == "":
        return float("nan")
    if isinstance(
        value,
        int | float | str | bytes | bytearray | np.integer | np.floating,
    ):
        try:
            return float(value)
        except ValueError:
            return float("nan")
    return float("nan")


def build_strategy_dataframe(
    index_rows: list[dict[str, str]],
    parent_labels: np.ndarray,
    payload: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Assemble a per-candidate DataFrame for one strategy."""
    extra_keys = [key for key in payload.keys() if key != "score"]
    n = len(index_rows)
    data: dict[str, np.ndarray | list] = {
        "sample_idx": np.array(
            [int(row.get("sample_idx", i)) for i, row in enumerate(index_rows)],
            dtype=np.int64,
        ),
        "window_name": [row.get("window_name", "") for row in index_rows],
        "lat": np.array(
            [_coerce_float(row.get("lat", "")) for row in index_rows], dtype=np.float64
        ),
        "lon": np.array(
            [_coerce_float(row.get("lon", "")) for row in index_rows], dtype=np.float64
        ),
        "parent_label": parent_labels.astype(np.int32),
        "score": np.asarray(payload["score"], dtype=np.float32),
    }
    for key in extra_keys:
        arr = np.asarray(payload[key])
        if arr.ndim == 1 and arr.shape[0] == n:
            data[key] = arr
    return pd.DataFrame(data)


def write_strategy_parquet(
    path: str,
    index_rows: list[dict[str, str]],
    parent_labels: np.ndarray,
    payload: dict[str, np.ndarray],
) -> None:
    """Write aligned acquisition scores with strategy-specific diagnostics."""
    df = build_strategy_dataframe(index_rows, parent_labels, payload)
    df.to_parquet(path, index=False)
    print(f"[save] Strategy parquet -> {path}")


def save_strategy_outputs(
    strategy: str,
    input_dir: str,
    output_dir: str,
    reference_dir: str,
    index_rows: list[dict[str, str]],
    parent_labels: np.ndarray,
    payload: dict[str, np.ndarray],
    extra_summary: dict[str, object] | None = None,
) -> None:
    """Persist aligned scores, ranked sample IDs, and a summary JSON."""
    os.makedirs(output_dir, exist_ok=True)
    prefix = _strategy_output_prefix(strategy)
    parquet_path = os.path.join(output_dir, f"{prefix}_scores.parquet")
    ranked_path = os.path.join(output_dir, f"{prefix}_ranked_sample_idx.npy")
    summary_path = os.path.join(output_dir, f"{prefix}_summary.json")

    write_strategy_parquet(parquet_path, index_rows, parent_labels, payload)

    sample_indices = np.array(
        [int(row["sample_idx"]) for row in index_rows], dtype=np.int64
    )
    ranked_indices = np.argsort(-payload["score"], kind="stable")
    np.save(ranked_path, sample_indices[ranked_indices])
    print(f"[save] Ranked sample indices -> {ranked_path}")

    summary = {
        "strategy": strategy,
        "input_dir": input_dir,
        "reference_dir": reference_dir,
        "output_dir": output_dir,
        "candidate_size": int(len(index_rows)),
        "score_min": float(np.min(payload["score"])) if len(index_rows) else 0.0,
        "score_max": float(np.max(payload["score"])) if len(index_rows) else 0.0,
        "score_mean": float(np.mean(payload["score"])) if len(index_rows) else 0.0,
        "top5_sample_idx": sample_indices[ranked_indices[:5]].astype(int).tolist(),
    }
    if extra_summary:
        summary.update(extra_summary)
    save_summary_json(summary_path, summary)


def run_xglobal_bridge(args: argparse.Namespace) -> None:
    """Score candidates that bridge two frozen parent clusters."""
    state = load_candidate_state(args.input_dir, args.reference_dir, args.seed)
    model = state["model"]
    payload = strat.compute_xglobal_bridge_scores(
        state["pca_data"],
        model["parent_centroids"],
        state["residuals_by_parent"],
    )
    save_strategy_outputs(
        strategy="xglobal_bridge",
        input_dir=args.input_dir,
        output_dir=args.output_dir or os.path.join(args.input_dir, "_scores"),
        reference_dir=args.reference_dir,
        index_rows=state["index_rows"],
        parent_labels=state["parent_labels"],
        payload=payload,
        extra_summary={},
    )


def run_sparse_infill(args: argparse.Namespace) -> None:
    """Score candidates in sparse but still-supported residual regions."""
    state = load_candidate_state(args.input_dir, args.reference_dir, args.seed)
    model = state["model"]
    metric = str(model["distance_metric"].item())
    payload = strat.compute_sparse_infill_scores(
        state["residuals"],
        state["parent_labels"],
        state["residuals_by_parent"],
        k_sparse=args.k_sparse,
        k_support=args.k_support,
        sparse_percentile=args.sparse_percentile,
        support_percentile=args.support_percentile,
        metric=metric,
    )
    save_strategy_outputs(
        strategy="sparse-infill",
        input_dir=args.input_dir,
        output_dir=args.output_dir or os.path.join(args.input_dir, "_scores"),
        reference_dir=args.reference_dir,
        index_rows=state["index_rows"],
        parent_labels=state["parent_labels"],
        payload=payload,
        extra_summary={
            "k_sparse": args.k_sparse,
            "k_support": args.k_support,
            "sparse_percentile": args.sparse_percentile,
            "support_percentile": args.support_percentile,
            "distance_metric": metric,
        },
    )


def run_xlocal_bridge(args: argparse.Namespace) -> None:
    """Score candidates that connect two local residual-space modes."""
    state = load_candidate_state(args.input_dir, args.reference_dir, args.seed)
    payload = strat.compute_xlocal_bridge_scores(
        state["residuals"],
        state["parent_labels"],
        state["residuals_by_parent"],
        n_local_modes=args.local_modes,
        seed=args.seed,
    )
    save_strategy_outputs(
        strategy="xlocal_bridge",
        input_dir=args.input_dir,
        output_dir=args.output_dir or os.path.join(args.input_dir, "_scores"),
        reference_dir=args.reference_dir,
        index_rows=state["index_rows"],
        parent_labels=state["parent_labels"],
        payload=payload,
        extra_summary={"local_modes": args.local_modes},
    )


def run_prototypes(args: argparse.Namespace) -> None:
    """Score candidates by proximity to residual-space local prototypes."""
    state = load_candidate_state(args.input_dir, args.reference_dir, args.seed)
    payload = strat.compute_prototype_scores(
        state["residuals"],
        state["parent_labels"],
        state["residuals_by_parent"],
        n_local_prototypes=args.local_prototypes,
        radius_percentile=args.radius_percentile,
        coverage_k=args.coverage_k,
        seed=args.seed,
    )
    save_strategy_outputs(
        strategy="prototypes",
        input_dir=args.input_dir,
        output_dir=args.output_dir or os.path.join(args.input_dir, "_scores"),
        reference_dir=args.reference_dir,
        index_rows=state["index_rows"],
        parent_labels=state["parent_labels"],
        payload=payload,
        extra_summary={
            "local_prototypes": args.local_prototypes,
            "radius_percentile": args.radius_percentile,
            "coverage_k": args.coverage_k,
        },
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for acquisition scoring."""
    parser = argparse.ArgumentParser(
        description="Score candidate embeddings with multiple acquisition strategies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="strategy", required=True)

    def add_common_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--input-dir", required=True, help="Candidate directory with shard_*.npz"
        )
        p.add_argument(
            "--reference-dir",
            required=True,
            help="Directory containing reference artifacts from reference_model.py",
        )
        p.add_argument(
            "--output-dir",
            default=None,
            help="Where to write outputs. Defaults to {input_dir}/_scores",
        )
        p.add_argument("--seed", type=int, default=42, help="Random seed")

    xglobal_bridge = sub.add_parser(
        "xglobal_bridge",
        help="Score candidates that bridge two frozen parent clusters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(xglobal_bridge)
    xglobal_bridge.set_defaults(func=run_xglobal_bridge)

    sparse_infill = sub.add_parser(
        "sparse-infill",
        help="Score sparse-but-supported regions inside parent residual space.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(sparse_infill)
    sparse_infill.add_argument(
        "--k-sparse",
        type=int,
        default=256,
        help="k for the sparse-region statistic (mean kNN distance in residual space).",
    )
    sparse_infill.add_argument(
        "--k-support",
        type=int,
        default=32,
        help="k for the support statistic (mean kNN distance in residual space).",
    )
    sparse_infill.add_argument(
        "--sparse-percentile",
        type=float,
        default=90.0,
        help="Per-parent reference percentile where the sparsity soft gate turns on.",
    )
    sparse_infill.add_argument(
        "--support-percentile",
        type=float,
        default=60.0,
        help="Per-parent reference percentile where the support soft gate starts turning off.",
    )
    sparse_infill.set_defaults(func=run_sparse_infill)

    xlocal_bridge = sub.add_parser(
        "xlocal_bridge",
        help="Score points that bridge two local residual-space modes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(xlocal_bridge)
    xlocal_bridge.add_argument(
        "--local-modes",
        type=int,
        default=3,
        help="Number of local residual-space modes to fit per parent for xlocal bridge scoring.",
    )
    xlocal_bridge.set_defaults(func=run_xlocal_bridge)

    prototypes = sub.add_parser(
        "prototypes",
        help="Score embedding-only local prototypes / representative points.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(prototypes)
    prototypes.add_argument(
        "--local-prototypes",
        type=int,
        default=50,
        help="Number of residual-space KMeans prototype centroids to fit per parent cluster.",
    )
    prototypes.add_argument(
        "--radius-percentile",
        type=float,
        default=80.0,
        help="Percentile of reference distances to each local prototype used as its normalization radius.",
    )
    prototypes.add_argument(
        "--coverage-k",
        type=int,
        default=0,
        help=(
            "k for the sparsity-ratio coverage penalty. When > 0, each candidate's "
            "local sparsity in its matched prototype subcluster is compared to the "
            "reference self-sparsity. Candidates in already-dense regions are "
            "penalized. 0 disables the penalty."
        ),
    )
    prototypes.set_defaults(func=run_prototypes)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Dispatch the requested acquisition strategy subcommand."""
    args = parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])

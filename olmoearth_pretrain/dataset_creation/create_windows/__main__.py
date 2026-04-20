"""Unified CLI for creating rslearn windows.

Two input modes:

* ``--lonlats_json PATH`` — a JSON array of ``[lon, lat]`` pairs. Uses the legacy
  high-resolution NAIP lookup to pick a timestamp per coarse tile.
* ``--corpus_csv PATH`` — a studio corpus CSV with at minimum ``lon, lat, start_time``
  columns. Uses the corpus-provided timestamps directly; no NAIP gating.

Exactly one of the two must be supplied.
"""

import argparse
import csv
import json
from datetime import UTC, datetime

import pandas as pd
from upath import UPath

from .util import (
    STATUS_CREATED,
    create_windows_from_corpus,
    create_windows_with_highres_time,
)


def _load_corpus(
    csv_path: str,
) -> tuple[list[tuple[float, float, datetime]], pd.DataFrame]:
    """Read (lon, lat, center_time) tuples + the full dataframe from a corpus CSV."""
    df = pd.read_csv(csv_path)
    df["start_time"] = pd.to_datetime(df["start_time"], format="ISO8601", utc=True)
    entries: list[tuple[float, float, datetime]] = [
        (
            float(row["lon"]),
            float(row["lat"]),
            row["start_time"].to_pydatetime().astimezone(UTC),
        )
        for _, row in df.iterrows()
    ]
    return entries, df


def _write_mapping_csv(
    mapping_path: UPath,
    df: pd.DataFrame,
    results: list,
) -> None:
    """Persist the per-corpus-row tile + status mapping for traceability."""
    with mapping_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "corpus_idx",
            "lon",
            "lat",
            "corpus_time",
            "tile_crs",
            "tile_col",
            "tile_row",
            "selected_time",
            "status",
            "theme",
            "project_name",
        ])
        for i, ((tile, selected_time, status), (_, row)) in enumerate(
            zip(results, df.iterrows())
        ):
            w.writerow([
                i,
                row["lon"],
                row["lat"],
                row["start_time"].isoformat(),
                str(tile.crs),
                tile.col,
                tile.row,
                selected_time.isoformat(),
                status,
                row.get("theme", ""),
                row.get("project_name", ""),
            ])


def _run_lonlats(args: argparse.Namespace) -> None:
    with open(args.lonlats_json) as f:
        lonlats = [(lon, lat) for lon, lat in json.load(f)]
    print(f"Loaded {len(lonlats)} lon/lat pairs from {args.lonlats_json}")
    create_windows_with_highres_time(
        UPath(args.ds_path),
        lonlats,
        force_lowres_prob=args.force_lowres_prob,
        workers=args.workers,
    )


def _run_corpus(args: argparse.Namespace) -> None:
    entries, df = _load_corpus(args.corpus_csv)
    print(f"Loaded {len(entries)} corpus entries from {args.corpus_csv}")
    if "theme" in df.columns:
        print(f"  Themes: {df['theme'].nunique()} unique")
    print(f"  Time range: {df['start_time'].min()} to {df['start_time'].max()}")

    results = create_windows_from_corpus(
        ds_path=UPath(args.ds_path),
        corpus_entries=entries,
        verify_s2=args.verify_s2,
        min_s2_months=args.min_s2_months,
        workers=args.workers,
    )

    mapping_path = (
        UPath(args.mapping_csv)
        if args.mapping_csv
        else UPath(args.ds_path) / "corpus_mapping.csv"
    )
    _write_mapping_csv(mapping_path, df, results)

    n_created = sum(1 for _, _, s in results if s == STATUS_CREATED)
    n_skipped = len(results) - n_created
    print(f"\nSummary: {n_created} corpus entries placed, {n_skipped} skipped")
    print(f"Mapping saved to {mapping_path}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Create rslearn windows from lon/lat list OR studio corpus CSV.",
    )
    p.add_argument("--ds_path", required=True, help="rslearn dataset path")
    p.add_argument("--workers", type=int, default=32)

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--lonlats_json",
        help="JSON array of [lon, lat] pairs (legacy high-res NAIP path).",
    )
    src.add_argument(
        "--corpus_csv",
        help="Studio corpus CSV with lon, lat, start_time columns.",
    )

    p.add_argument(
        "--force_lowres_prob",
        type=float,
        default=0.25,
        help="[--lonlats_json only] probability to skip NAIP and use fallback 10m.",
    )

    p.add_argument(
        "--verify_s2",
        action="store_true",
        help="[--corpus_csv only] drop coarse tiles lacking enough S2 months.",
    )
    p.add_argument(
        "--min_s2_months",
        type=int,
        default=6,
        help="[--corpus_csv only] min distinct S2 months required when --verify_s2.",
    )
    p.add_argument(
        "--mapping_csv",
        default=None,
        help="[--corpus_csv only] output path for the corpus->tile mapping CSV.",
    )
    return p


def main() -> None:
    """Entrypoint."""
    args = _build_parser().parse_args()
    if args.lonlats_json:
        _run_lonlats(args)
    else:
        _run_corpus(args)


if __name__ == "__main__":
    main()

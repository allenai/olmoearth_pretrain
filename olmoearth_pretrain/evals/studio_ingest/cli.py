"""Command-line interface for Studio dataset ingestion.

This module provides CLI commands for:
- ingest: Full ingestion of a dataset
- validate: Validate a dataset without ingesting
- list: List all registered datasets
- info: Show details for a specific dataset
- compute-norm-stats: Compute normalization stats for existing dataset

Usage:
    uv run --group ingest python -m olmoearth_pretrain.evals.studio_ingest.cli <command> [options]

Examples:
    # Ingest a dataset
    python -m olmoearth_pretrain.evals.studio_ingest.cli ingest \\
        --name lfmc \\
        --display-name "Live Fuel Moisture Content" \\
        --source gs://bucket/lfmc \\
        --task-type regression \\
        --modalities sentinel2_l2a sentinel1 \\
        --property-name lfmc_value

    # Validate without ingesting
    python -m olmoearth_pretrain.evals.studio_ingest.cli validate \\
        --source gs://bucket/lfmc \\
        --modalities sentinel2_l2a

    # List all datasets
    python -m olmoearth_pretrain.evals.studio_ingest.cli list

    # Show dataset info
    python -m olmoearth_pretrain.evals.studio_ingest.cli info --name lfmc

Todo:
-----
- [ ] Add --dry-run flag to ingest command
- [ ] Add --output-format (json, table) for list/info commands
- [ ] Add remove command (with confirmation)
- [ ] Add update command for modifying existing entries
"""

from __future__ import annotations

import argparse
import logging
import sys

from olmoearth_pretrain.evals.studio_ingest.ingest import IngestConfig, ingest_dataset

# Band stats computation is done via band_stats.py CLI directly
# from olmoearth_pretrain.evals.studio_ingest.band_stats import compute_band_stats
from olmoearth_pretrain.evals.studio_ingest.registry import Registry
from olmoearth_pretrain.evals.studio_ingest.validate import validate_dataset

logger = logging.getLogger(__name__)


# =============================================================================
# Command: ingest
# =============================================================================


def cmd_ingest(args: argparse.Namespace) -> int:
    """Run the ingest command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    logger.info(f"Ingesting dataset: {args.name}")

    # Build config from args
    config = IngestConfig(
        name=args.name,
        source_path=args.source,
        olmoearth_run_config_path=args.olmoearth_run_config_path,
        max_samples=args.max_samples,
        sample_fraction=args.sample_fraction,
        groups=args.groups,
    )

    entry = ingest_dataset(config)
    print(f"\n✓ Successfully ingested dataset: {entry.name}")
    print(f"  Location: {entry.weka_path}")
    print(f"  Splits: {entry.splits}")
    # Turn into the eval dataset config
    eval_config = entry.to_eval_config()
    print(f"  Eval config: {eval_config}")
    return 0


# TODO: Use a better way of setting up the args that is easier to maintain
def add_ingest_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the ingest command."""
    # Required arguments
    parser.add_argument(
        "--name",
        required=True,
        help="Unique identifier for the dataset (e.g., 'lfmc')",
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Path to source rslearn dataset (e.g., 'gs://bucket/dataset')",
    )
    parser.add_argument(
        "--olmoearth-run-config-path",
        required=True,
        help="Path to olmoearth run config (e.g., 'path/to/olmoearth_run.yaml')",
    )

    # Optional sampling arguments
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process for stats computation",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=None,
        help="Fraction of samples to use (0.0-1.0) for stats computation",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=None,
        help="Dataset groups to filter by (e.g., 'train_group')",
    )



# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> int:
    """Main CLI entry point.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = argparse.ArgumentParser(
        prog="studio_ingest",
        description="Ingest Studio datasets into OlmoEarth eval system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest a classification dataset
  %(prog)s ingest --name lfmc --display-name "LFMC" --source gs://... \\
      --task-type classification --modalities sentinel2_l2a --property-name category

  # List all datasets
  %(prog)s list

  # Show dataset info
  %(prog)s info --name lfmc
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ingest command
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest a dataset from Studio/GCS",
    )
    add_ingest_args(ingest_parser)
    ingest_parser.set_defaults(func=cmd_ingest)

    # Parse args
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    # Run command
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

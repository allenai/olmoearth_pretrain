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
    )

    entry = ingest_dataset(config)
    print(f"\n✓ Successfully ingested dataset: {entry.name}")
    print(f"  Location: {entry.weka_path}")
    print(f"  Splits: {entry.splits}")
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


# =============================================================================
# Command: validate
# =============================================================================


def cmd_validate(args: argparse.Namespace) -> int:
    """Run the validate command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    logger.info(f"Validating dataset: {args.source}")

    result = validate_dataset(
        source_path=args.source,
        modalities=args.modalities,
        task_type=args.task_type,
        target_property=args.property_name,
    )

    print(result)

    return 0 if result.is_valid else 1


def add_validate_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the validate command."""
    parser.add_argument(
        "--source",
        required=True,
        help="Path to source rslearn dataset",
    )
    parser.add_argument(
        "--modalities",
        required=True,
        nargs="+",
        help="List of modality names to validate",
    )
    parser.add_argument(
        "--task-type",
        default="classification",
        choices=["classification", "regression", "segmentation"],
        help="Type of task (default: classification)",
    )
    parser.add_argument(
        "--property-name",
        default="category",
        help="Name of the property containing labels (default: category)",
    )


# =============================================================================
# Command: list
# =============================================================================


def cmd_list(args: argparse.Namespace) -> int:
    """Run the list command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        registry = Registry.load()
    except Exception as e:
        logger.error(f"Failed to load registry: {e}")
        return 1

    if len(registry) == 0:
        print("No datasets registered.")
        return 0

    print(f"\nRegistered datasets ({len(registry)}):\n")

    # Format as table
    print(f"{'Name':<20} {'Task Type':<15} {'Modalities':<30} {'Samples':<15}")
    print("-" * 80)

    for entry in registry:
        modalities = ", ".join(entry.modalities[:3])
        if len(entry.modalities) > 3:
            modalities += f" (+{len(entry.modalities) - 3})"

        total_samples = sum(entry.splits.values())
        print(
            f"{entry.name:<20} {entry.task_type:<15} {modalities:<30} {total_samples:<15}"
        )

    return 0


def add_list_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the list command."""
    # No additional arguments needed for now
    parser.add_argument(
        "--task-type",
        choices=["classification", "regression", "segmentation"],
        help="Filter by task type",
    )


# =============================================================================
# Command: info
# =============================================================================


def cmd_info(args: argparse.Namespace) -> int:
    """Run the info command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        registry = Registry.load()
        entry = registry.get(args.name)
    except KeyError as e:
        logger.error(str(e))
        return 1
    except Exception as e:
        logger.error(f"Failed to load registry: {e}")
        return 1

    # Print formatted info
    print(f"\n=== {entry.display_name} ({entry.name}) ===\n")
    print(f"Task Type:       {entry.task_type}")
    print(f"Target Property: {entry.target_property}")
    if entry.classes:
        print(f"Classes:         {', '.join(entry.classes)}")
    print(f"Modalities:      {', '.join(entry.modalities)}")
    print(f"Temporal Range:  {entry.temporal_range[0]} to {entry.temporal_range[1]}")
    print(f"Patch Size:      {entry.patch_size}px")
    print()
    print(f"Weka Path:       {entry.weka_path}")
    print(f"Source Path:     {entry.source_path}")
    print()
    print("Splits:")
    for split, count in entry.splits.items():
        print(f"  {split}: {count} samples")
    print()
    print(f"Created:         {entry.created_at}")
    print(f"Created By:      {entry.created_by}")
    if entry.notes:
        print(f"Notes:           {entry.notes}")

    return 0


def add_info_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the info command."""
    parser.add_argument(
        "--name",
        required=True,
        help="Name of the dataset to show info for",
    )


# =============================================================================
# Command: compute-norm-stats
# =============================================================================


def cmd_compute_norm_stats(args: argparse.Namespace) -> int:
    """Run the compute-norm-stats command.

    This prints instructions for using the band_stats CLI directly,
    which provides more control over the computation.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        registry = Registry.load()
        entry = registry.get(args.name)
    except KeyError as e:
        logger.error(str(e))
        return 1
    except Exception as e:
        logger.error(f"Failed to load registry: {e}")
        return 1

    # Print instructions for using band_stats CLI
    print(f"\nTo compute band stats for '{entry.name}', run:\n")
    print(
        f"uv run --group ingest python -m olmoearth_pretrain.evals.studio_ingest.band_stats \\\n"
        f"    --ds_path {entry.weka_path} \\\n"
        f"    --input_layers {' '.join(entry.modalities)} \\\n"
        f"    --output_json {entry.weka_path}/norm_stats.json"
    )
    print()
    return 0


def add_compute_norm_stats_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the compute-norm-stats command."""
    parser.add_argument(
        "--name",
        required=True,
        help="Name of the dataset to show band_stats command for",
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

    # validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a dataset without ingesting",
    )
    add_validate_args(validate_parser)
    validate_parser.set_defaults(func=cmd_validate)

    # list command
    list_parser = subparsers.add_parser(
        "list",
        help="List all registered datasets",
    )
    add_list_args(list_parser)
    list_parser.set_defaults(func=cmd_list)

    # info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show details for a specific dataset",
    )
    add_info_args(info_parser)
    info_parser.set_defaults(func=cmd_info)

    # compute-norm-stats command
    compute_norm_stats_parser = subparsers.add_parser(
        "compute-norm-stats",
        help="Compute normalization stats for existing dataset",
    )
    add_compute_norm_stats_args(compute_norm_stats_parser)
    compute_norm_stats_parser.set_defaults(func=cmd_compute_norm_stats)

    # Parse args
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    # Run command
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

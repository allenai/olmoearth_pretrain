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
import json
import logging
import sys

from olmoearth_pretrain.evals.studio_ingest.ingest import IngestConfig, ingest_dataset
from olmoearth_pretrain.evals.studio_ingest.registry import Registry

logger = logging.getLogger(__name__)


# =============================================================================
# Command: ingest
# =============================================================================


def parse_tags(tags_list: list[str] | None) -> dict[str, list[str]] | None:
    """Parse tags from CLI format to dict.

    Args:
        tags_list: List of strings like ["split=val,test", "quality=high"]

    Returns:
        Dict like {"split": ["val", "test"], "quality": ["high"]} or None
    """
    if not tags_list:
        return None
    tags_dict = {}
    for tag_str in tags_list:
        if "=" not in tag_str:
            raise ValueError(f"Invalid tag format '{tag_str}'. Expected 'key=val1,val2'")
        key, values = tag_str.split("=", 1)
        tags_dict[key] = [v.strip() for v in values.split(",")]
    return tags_dict


def cmd_ingest(args: argparse.Namespace) -> int:
    """Run the ingest command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    logger.info(f"Ingesting dataset: {args.name}")

    # Parse tags from CLI format
    tags = parse_tags(args.tags)

    # Build config from args
    config = IngestConfig(
        name=args.name,
        source_path=args.source,
        olmoearth_run_config_path=args.olmoearth_run_config_path,
        num_samples=args.num_samples,
        groups=args.groups,
        tags=tags,
    )

    entry = ingest_dataset(config)
    print(f"\n✓ Successfully ingested dataset: {entry.name}")

    if args.register:
        registry = Registry.load()
        registry.add(entry, overwrite=args.overwrite)
        registry.save()
        print(f"✓ Registered '{entry.name}' to registry")

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
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to process for stats computation (default: all)",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=None,
        help="Dataset groups to filter by (e.g., 'train_group')",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        default=None,
        help="Filter windows by tags. Format: key=val1,val2 (e.g., 'split=val quality=high,medium')",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Register the dataset to the registry after ingestion",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing registry entry if it exists",
    )


# =============================================================================
# Command: list
# =============================================================================


def cmd_list(args: argparse.Namespace) -> int:
    """List all registered datasets."""
    registry = Registry.load()

    if len(registry) == 0:
        print("No datasets registered.")
        return 0

    print(f"Registered datasets ({len(registry)}):\n")
    for entry in registry:
        print(f"  {entry.name}")
        print(f"    task: {entry.task_type}, classes: {entry.num_classes}")
        print(f"    modalities: {entry.modalities}")
        print(f"    path: {entry.weka_path or entry.source_path}")
        print()

    return 0


# =============================================================================
# Command: info
# =============================================================================


def cmd_info(args: argparse.Namespace) -> int:
    """Show detailed info for a dataset."""
    registry = Registry.load()

    try:
        entry = registry.get(args.name)
    except KeyError as e:
        print(f"Error: {e}")
        return 1

    print(json.dumps(entry.to_dict(), indent=2))
    return 0


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

    # list command
    list_parser = subparsers.add_parser(
        "list",
        help="List all registered datasets",
    )
    list_parser.set_defaults(func=cmd_list)

    # info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show detailed info for a dataset",
    )
    info_parser.add_argument(
        "--name",
        required=True,
        help="Name of the dataset",
    )
    info_parser.set_defaults(func=cmd_info)

    # Parse args
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    # Run command
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

"""Command-line interface for Studio dataset ingestion.

This module provides CLI commands for:
- ingest: Full ingestion of a dataset
- list: List all registered datasets
- info: Show details for a specific dataset

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

# =============================================================================
# Configure logging BEFORE importing other modules
# This ensures all loggers (including rslearn) have a handler
# =============================================================================
def _setup_logging() -> None:
    """Configure logging for CLI usage."""
    # Create handler that writes to stderr
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    ))

    # Configure root logger
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    # Explicitly enable our loggers and rslearn
    logging.getLogger("olmoearth_pretrain").setLevel(logging.INFO)
    logging.getLogger("rslearn").setLevel(logging.INFO)

_setup_logging()

# Now import the rest (their loggers will inherit from root)
from olmoearth_pretrain.evals.studio_ingest.ingest import IngestConfig, ingest_dataset
from olmoearth_pretrain.evals.studio_ingest.registry import Registry

logger = logging.getLogger(__name__)


# =============================================================================
# Command: ingest
# =============================================================================


def parse_tags(tags_list: list[str] | None) -> dict[str, str] | None:
    """Parse tags from CLI format to dict.

    Args:
        tags_list: List of strings like ["split=val", "quality=high"] or just ["oep_eval"]
            - "key=value" filters windows where tag key equals value
            - "key" (no value) filters windows where tag key exists (any value)

    Returns:
        Dict like {"split": "val", "quality": "high", "oep_eval": ""} or None
        Empty string value means "key exists" (rslearn skips value comparison for empty/falsy values)
    """
    if not tags_list:
        return None
    tags_dict = {}
    for tag_str in tags_list:
        if "=" in tag_str:
            key, value = tag_str.split("=", 1)
            tags_dict[key] = value.strip()
        else:
            # Key without value: filter by existence (empty string)
            tags_dict[tag_str] = ""
    return tags_dict


def cmd_ingest(args: argparse.Namespace) -> int:
    """Run the ingest command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    logger.info(f"Ingesting dataset: {args.name}")

    # Parse source tags from CLI format
    source_tags = parse_tags(args.source_tags)

    # Build config from args
    config = IngestConfig(
        name=args.name,
        source_path=args.source,
        olmoearth_run_config_path=args.olmoearth_run_config_path,
        source_groups=args.source_groups,
        source_tags=source_tags,
        val_test_split_ratio=args.val_test_split_ratio,
        train_val_split_ratio=args.train_val_split_ratio,
        split_seed=args.split_seed,
        num_samples=args.num_samples,
        untar_source=args.untar_source,
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

    # Source filtering arguments
    parser.add_argument(
        "--source-groups",
        nargs="+",
        default=None,
        help="Source dataset groups to pull from (e.g., 'train val')",
    )
    parser.add_argument(
        "--source-tags",
        nargs="+",
        default=None,
        help="Filter source windows by tags. Format: 'key=value' or 'key' (exists check)",
    )

    # Split configuration arguments
    parser.add_argument(
        "--val-test-split-ratio",
        type=float,
        default=0.5,
        help="Ratio of val to keep when splitting val into val+test (default: 0.5)",
    )
    parser.add_argument(
        "--train-val-split-ratio",
        type=float,
        default=0.8,
        help="Ratio of train to keep when splitting train into train+val (default: 0.8)",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed for split generation (default: 42)",
    )

    # Normalization arguments
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples for stats computation (default: all)",
    )

    # Archive handling
    parser.add_argument(
        "--untar-source",
        action="store_true",
        help="Source is a .tar.gz archive on GCS; stream and extract directly to Weka",
    )

    # Registry arguments
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

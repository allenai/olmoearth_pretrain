"""Shared command-line helpers for rslearn-to-OlmoEarth converters."""

import argparse


def add_common_arguments(
    parser: argparse.ArgumentParser, default_groups: list[str] | None
) -> None:
    """Add arguments shared by rslearn-to-OlmoEarth converter CLIs.

    Args:
        parser: Parser to add the argument to.
        default_groups: Groups used when ``--group`` is omitted. ``None`` scans all
            groups.
    """
    parser.add_argument(
        "--ds_path",
        type=str,
        help="Source rslearn dataset path",
        required=True,
    )
    parser.add_argument(
        "--olmoearth_path",
        type=str,
        help="Destination OlmoEarth Pretrain dataset path",
        required=True,
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of workers to use",
        default=32,
    )
    parser.add_argument(
        "--group",
        nargs="+",
        dest="groups",
        default=default_groups,
        help="rslearn window group(s) to convert",
    )

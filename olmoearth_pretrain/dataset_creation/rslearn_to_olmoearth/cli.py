"""Shared command-line helpers for rslearn-to-OlmoEarth converters."""

import argparse


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by rslearn-to-OlmoEarth converter CLIs.

    Args:
        parser: Parser to add the argument to.
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

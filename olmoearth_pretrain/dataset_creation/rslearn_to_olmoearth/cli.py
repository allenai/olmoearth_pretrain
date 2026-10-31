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


# Supported layouts for multitemporal raster layers (see multitemporal_raster.py).
MULTITEMPORAL_LAYOUTS = ("monthly", "period_mosaic")


def add_layout_argument(parser: argparse.ArgumentParser) -> None:
    """Add the --layout argument for multitemporal raster converters.

    Args:
        parser: Parser to add the argument to.
    """
    parser.add_argument(
        "--layout",
        type=str,
        choices=list(MULTITEMPORAL_LAYOUTS),
        help=(
            "Layout of the multitemporal rslearn layers. 'monthly' reads the twelve "
            "single-mosaic layers <prefix>_mo01..<prefix>_mo12. 'period_mosaic' reads "
            "one MOSAIC layer named <prefix> whose item groups are per-period mosaics "
            "(query_config.period_duration), with timestamps taken from the layer's "
            "group_time_ranges."
        ),
        default="monthly",
    )

"""Shared command-line helpers for rslearn-to-OlmoEarth converters."""

import argparse

from rslearn.dataset import Window


def filter_paired_secondary_windows(windows: list[Window]) -> list[Window]:
    """Drop the secondary member of each paired pre/post change window pair.

    Paired change samples in the open-set dataset consist of two windows at the same
    location that share one ``example_id`` (see ``create_windows.from_open_set``). The
    static (non-multitemporal) modality converters run per window and must emit each
    example exactly once, so they only process the pair's primary window (the one
    carrying the label layer; normally the post window). Windows without the
    ``paired_part`` option (regular samples, grid datasets) are kept unchanged.
    """
    return [
        w
        for w in windows
        if not w.options.get("paired_part") or w.options.get("paired_primary")
    ]


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

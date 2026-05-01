"""Create windows corresponding to a list of longitude/latitude."""

import argparse
import json
from datetime import UTC, datetime

from upath import UPath

from .util import create_windows_with_highres_time


def _parse_custom_time(s: str) -> datetime:
    """Parse ISO8601; treat naive values as UTC."""
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create windows based on specified locations",
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        help="Dataset path",
        required=True,
    )
    parser.add_argument(
        "--fname",
        type=str,
        help="JSON filename containing list of [lot, lat]",
        required=True,
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of worker processes",
        default=32,
    )
    parser.add_argument(
        "--custom_time",
        type=str,
        default=None,
        help=(
            "Optional ISO8601 timestamp (e.g. 2020-09-15 or 2020-09-15T12:00:00Z). "
            "When set, skips NAIP-based time selection and uses this as the window center time."
        ),
    )
    args = parser.parse_args()

    with open(args.fname) as f:
        lonlats = [(lon, lat) for lon, lat in json.load(f)]

    custom_time = (
        _parse_custom_time(args.custom_time) if args.custom_time is not None else None
    )

    create_windows_with_highres_time(
        UPath(args.ds_path),
        lonlats,
        force_lowres_prob=0.25,
        workers=args.workers,
        custom_time=custom_time,
    )

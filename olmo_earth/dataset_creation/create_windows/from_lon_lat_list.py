"""Create windows corresponding to a list of longitude/latitude."""

import argparse
import json

from upath import UPath

from .util import create_windows_with_highres_time

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
    args = parser.parse_args()

    with open(args.fname) as f:
        lonlats = [(lon, lat) for lon, lat in json.load(f)]

    create_windows_with_highres_time(
        UPath(args.ds_path), lonlats, force_lowres_prob=0.25, workers=args.workers
    )

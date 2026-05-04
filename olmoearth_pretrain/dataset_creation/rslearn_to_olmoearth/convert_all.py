"""Unified entry point to convert an rslearn dataset to OlmoEarth Pretrain format.

Usage:
    python -m olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth.convert_all \
        --ds_path /path/to/rslearn_dataset \
        --olmoearth_path /path/to/olmoearth_output \
        --workers 32

    # Convert only specific modalities:
    python -m olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth.convert_all \
        --ds_path ... --olmoearth_path ... \
        --modalities sentinel1,sentinel2_l2a,worldcover
"""

import argparse
import multiprocessing
from typing import Callable

import tqdm
from rslearn.dataset import Dataset, Window
from rslearn.utils.mp import star_imap_unordered
from upath import UPath

from .cdl import convert_cdl
from .openstreetmap import convert_openstreetmap
from .sentinel1 import convert_sentinel1
from .sentinel2_l2a import convert_sentinel2_l2a
from .srtm import convert_srtm
from .worldcereal import convert_worldcereal
from .worldcover import convert_worldcover
from .wri_canopy_height_map import convert_chm

# Per-window converter functions. Each takes (window, olmoearth_path, **kwargs).
CONVERTERS: dict[str, Callable] = {
    "cdl": convert_cdl,
    "openstreetmap": convert_openstreetmap,
    "sentinel1": convert_sentinel1,
    "sentinel2_l2a": convert_sentinel2_l2a,
    "srtm": convert_srtm,
    "worldcereal": convert_worldcereal,
    "worldcover": convert_worldcover,
    "wri_canopy_height_map": convert_chm,
}

# Modalities that accept use_temporal_stack kwarg.
TEMPORAL_MODALITIES = {"sentinel1", "sentinel2_l2a"}

ALL_MODALITIES = list(CONVERTERS.keys())


def _convert_window(
    window: Window,
    olmoearth_path: UPath,
    modalities: list[str],
    use_temporal_stack: bool,
) -> None:
    for mod in modalities:
        converter = CONVERTERS[mod]
        try:
            kwargs: dict = dict(window=window, olmoearth_path=olmoearth_path)
            if mod in TEMPORAL_MODALITIES:
                kwargs["use_temporal_stack"] = use_temporal_stack
            converter(**kwargs)
        except Exception as e:
            print(f"warning: {mod} failed for window {window.name}: {e}")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")

    parser = argparse.ArgumentParser(
        description="Convert rslearn dataset to OlmoEarth Pretrain format",
    )
    parser.add_argument(
        "--ds_path", type=str, required=True,
        help="Source rslearn dataset path",
    )
    parser.add_argument(
        "--olmoearth_path", type=str, required=True,
        help="Destination OlmoEarth Pretrain dataset path",
    )
    parser.add_argument(
        "--workers", type=int, default=32,
        help="Number of workers to use",
    )
    parser.add_argument(
        "--modalities", type=str, default=None,
        help=f"Comma-separated list of modalities to convert. Default: all. "
             f"Available: {','.join(ALL_MODALITIES)}",
    )
    parser.add_argument(
        "--legacy-monthly", action="store_true",
        help="Use legacy 12 separate _moNN layers for S1/S2 instead of temporal stack",
    )
    parser.add_argument(
        "--groups", type=str, default="res_10.0",
        help="Comma-separated window groups to load (default: res_10.0)",
    )
    args = parser.parse_args()

    modalities = ALL_MODALITIES
    if args.modalities:
        modalities = [m.strip() for m in args.modalities.split(",")]
        unknown = set(modalities) - set(ALL_MODALITIES)
        if unknown:
            parser.error(f"Unknown modalities: {unknown}. Available: {ALL_MODALITIES}")

    groups = [g.strip() for g in args.groups.split(",")]
    use_temporal_stack = not args.legacy_monthly

    print(f"Converting modalities: {modalities}")
    print(f"Temporal stack: {use_temporal_stack}")

    dataset = Dataset(UPath(args.ds_path))
    olmoearth_path = UPath(args.olmoearth_path)

    jobs = []
    for window in dataset.load_windows(
        workers=args.workers, show_progress=True, groups=groups,
    ):
        jobs.append(dict(
            window=window,
            olmoearth_path=olmoearth_path,
            modalities=modalities,
            use_temporal_stack=use_temporal_stack,
        ))

    print(f"Processing {len(jobs)} windows with {args.workers} workers...")

    p = multiprocessing.Pool(args.workers)
    outputs = star_imap_unordered(p, _convert_window, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()

    print("Done.")

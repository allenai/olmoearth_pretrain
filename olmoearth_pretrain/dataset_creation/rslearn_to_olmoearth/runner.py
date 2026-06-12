"""Shared CLI runner for rslearn-to-OlmoEarth window converters."""

import argparse
import multiprocessing
from collections.abc import Callable, Sequence
from typing import Any

import tqdm
from rslearn.dataset import Dataset, Window
from rslearn.utils.mp import star_imap_unordered
from upath import UPath

WindowConverter = Callable[[Window, UPath], None]


def make_window_converter_jobs(
    dataset: Dataset,
    olmoearth_path: UPath,
    *,
    workers: int,
    groups: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Build converter jobs for all matching rslearn windows."""
    load_kwargs: dict[str, Any] = {
        "workers": workers,
        "show_progress": True,
    }
    if groups is not None:
        load_kwargs["groups"] = list(groups)

    return [
        {
            "window": window,
            "olmoearth_path": olmoearth_path,
        }
        for window in dataset.load_windows(**load_kwargs)
    ]


def run_window_converter(
    converter: WindowConverter,
    *,
    groups: Sequence[str] | None = None,
    description: str = "Post-process OlmoEarth Pretrain data",
) -> None:
    """Run a window converter with the standard dataset conversion CLI."""
    multiprocessing.set_start_method("forkserver")

    parser = argparse.ArgumentParser(description=description)
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
    args = parser.parse_args()

    dataset = Dataset(UPath(args.ds_path))
    olmoearth_path = UPath(args.olmoearth_path)
    jobs = make_window_converter_jobs(
        dataset,
        olmoearth_path,
        workers=args.workers,
        groups=groups,
    )

    pool = multiprocessing.Pool(args.workers)
    outputs = star_imap_unordered(pool, converter, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    pool.close()

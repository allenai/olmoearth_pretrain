"""Create rslearn windows by copying (a subset of) windows from an existing dataset.

This is the entry point for building a dataset over the exact same locations and
timestamps as a previous corpus (e.g. the v1 `osm_sampling` dataset), so results are
directly comparable. Windows are copied verbatim: same name, group, projection,
bounds, time_range, and options. Only window metadata is copied -- no layer data.

A seeded random subset can be selected with `--num_windows`; the selected window
names (plus source path and seed) are persisted to `ds_path/selected_windows.json`
so the subset is reproducible and auditable.

If `--config-path` is provided, the command also writes `ds_path/config.json` via
`from_corpus.attach_dataset_config` (copy by default).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import random

import tqdm
from rslearn.dataset import Window
from rslearn.dataset.storage.file import FileWindowStorage, load_window
from rslearn.dataset.storage.storage import WindowStorage
from upath import UPath

from .from_corpus import DEFAULT_CONFIG_MODE, attach_dataset_config
from .util import star_imap


def list_window_names(src_ds_path: UPath, group: str) -> list[str]:
    """List window names in a group of the source dataset, sorted for determinism."""
    group_dir = src_ds_path / "windows" / group
    if not group_dir.exists():
        raise FileNotFoundError(f"window group directory {group_dir} does not exist")
    return sorted(p.name for p in group_dir.iterdir())


def select_window_names(
    names: list[str],
    num_windows: int | None,
    seed: int,
) -> list[str]:
    """Select a seeded random subset of window names (or all if num_windows is None)."""
    if num_windows is None or num_windows >= len(names):
        return names
    return sorted(random.Random(seed).sample(names, num_windows))


def _copy_one_window(
    src_storage: FileWindowStorage,
    dst_storage: WindowStorage,
    group: str,
    name: str,
) -> None:
    src_window = load_window(src_storage, src_storage.get_window_root(group, name))
    window = Window(
        storage=dst_storage,
        group=group,
        name=src_window.name,
        projection=src_window.projection,
        bounds=src_window.bounds,
        time_range=src_window.time_range,
        options=src_window.options,
    )
    window.save()


def copy_windows(
    src_ds_path: UPath,
    ds_path: UPath,
    group: str,
    names: list[str],
    workers: int = 32,
) -> None:
    """Copy the named windows from the source dataset into the target dataset.

    Args:
        src_ds_path: source rslearn dataset path (file window storage).
        ds_path: target rslearn dataset path.
        group: window group to copy from/to (same group name on both sides).
        names: window names to copy.
        workers: number of parallel worker processes.
    """
    src_storage = FileWindowStorage(src_ds_path)
    dst_storage = FileWindowStorage(ds_path)

    jobs = [
        dict(
            src_storage=src_storage,
            dst_storage=dst_storage,
            group=group,
            name=name,
        )
        for name in names
    ]

    if workers <= 1 or len(jobs) <= 1:
        for job in tqdm.tqdm(jobs, desc="Copy windows"):
            _copy_one_window(**job)
        return

    with multiprocessing.Pool(workers) as p:
        for _ in tqdm.tqdm(
            star_imap(p, _copy_one_window, jobs),
            desc="Copy windows",
            total=len(jobs),
        ):
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Create rslearn windows by copying a (seeded random subset of) windows "
            "from an existing rslearn dataset."
        )
    )
    parser.add_argument(
        "--src_ds_path",
        type=str,
        required=True,
        help="Source rslearn dataset path to copy windows from.",
    )
    parser.add_argument(
        "--ds_path", type=str, required=True, help="Target rslearn dataset path."
    )
    parser.add_argument(
        "--group",
        type=str,
        default="res_10",
        help="Window group to copy (default res_10).",
    )
    parser.add_argument(
        "--num_windows",
        type=int,
        default=None,
        help="Subset size; omit to copy every window in the group.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Subset sampling seed.")
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Optional config to expose as ds_path/config.json.",
    )
    parser.add_argument(
        "--config-mode",
        choices=["symlink", "copy"],
        default=DEFAULT_CONFIG_MODE,
        help="How to attach --config-path into the dataset root (default copy).",
    )
    parser.add_argument(
        "--force-config",
        action="store_true",
        help="Replace an existing ds_path/config.json if it differs.",
    )
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    src_ds_path = UPath(args.src_ds_path)
    ds_path = UPath(args.ds_path)

    if args.config_path:
        attach_dataset_config(
            ds_path=ds_path,
            config_path=UPath(args.config_path),
            mode=args.config_mode,
            force=args.force_config,
        )

    all_names = list_window_names(src_ds_path, args.group)
    selected = select_window_names(all_names, args.num_windows, args.seed)
    print(f"selected {len(selected)}/{len(all_names)} windows from {src_ds_path}")

    ds_path.mkdir(parents=True, exist_ok=True)
    manifest_path = ds_path / "selected_windows.json"
    with manifest_path.open("w") as f:
        json.dump(
            dict(
                src_ds_path=str(src_ds_path),
                group=args.group,
                seed=args.seed,
                num_windows=len(selected),
                total_source_windows=len(all_names),
                names=selected,
            ),
            f,
        )
    print(f"wrote manifest to {manifest_path}")

    copy_windows(src_ds_path, ds_path, args.group, selected, workers=args.workers)


__all__ = [
    "copy_windows",
    "list_window_names",
    "select_window_names",
]

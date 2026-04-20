"""Deterministic hash-based sharding of rslearn window names.

When a modality's materialize step is parallelized across multiple Beaker tasks,
we need to give each task a disjoint slice of the windows. We do this with a
stable hash (blake2b) of the window name so:

1. Every window lands in exactly one shard.
2. The partition is stable across runs -> resume relaunches the same shard
   contents, so `rslearn`'s `is_layer_completed` skips what the prior attempt
   finished.
3. The partition is uniform (blake2b is a good hash) so shards stay roughly
   balanced without us needing to inspect window sizes.

This is COMPLEMENTARY to `rslearn`'s own `random.shuffle(jobs)` inside
`apply_on_windows`: we distribute across tasks; rslearn load-balances within a
task's worker pool.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from upath import UPath


def hash_shard(name: str, num_shards: int) -> int:
    """Return the shard index (0..num_shards-1) for a given window name."""
    if num_shards <= 0:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if num_shards == 1:
        return 0
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % num_shards


def partition_windows(
    window_names: list[str], num_shards: int
) -> list[list[str]]:
    """Partition window names into `num_shards` disjoint lists by stable hash.

    Returns a list of length `num_shards`; each element is the (possibly empty)
    list of window names assigned to that shard. Order within a shard is the
    insertion order of the input, which keeps logs/shard files readable.
    """
    shards: list[list[str]] = [[] for _ in range(num_shards)]
    for name in window_names:
        shards[hash_shard(name, num_shards)].append(name)
    return shards


def enumerate_window_names(ds_path: UPath, group: str) -> list[str]:
    """List every window name present under ``{ds_path}/windows/{group}``.

    Matches rslearn's `FileWindowStorage.get_windows` layout: each subdir of
    the group directory is one window.
    """
    group_dir = ds_path / "windows" / group
    if not group_dir.exists():
        return []
    return sorted(p.name for p in group_dir.iterdir() if p.is_dir())


def shard_file_path(ds_path: UPath, modality: str, shard_idx: int) -> UPath:
    """Stable location for a shard's window-names file."""
    return ds_path / ".beaker_shards" / modality / f"shard_{shard_idx:04d}.txt"


def write_shard_files(
    ds_path: UPath,
    modality: str,
    shards: list[list[str]],
) -> list[UPath]:
    """Persist each shard's window names to ``.beaker_shards/{modality}/shard_NNNN.txt``.

    One window name per line. We overwrite existing shard files so a resume
    with the same window set produces identical shard files (and thus the
    exact same work per task).
    """
    paths: list[UPath] = []
    for i, names in enumerate(shards):
        path = shard_file_path(ds_path, modality, i)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(names) + ("\n" if names else ""))
        paths.append(path)
    return paths


def load_shard_file(path: Path | UPath) -> list[str]:
    """Read a shard file back into a list of window names."""
    text = path.read_text()
    return [line.strip() for line in text.splitlines() if line.strip()]

"""Tests for deterministic window sharding."""

from pathlib import Path

from upath import UPath

from olmoearth_pretrain.dataset_creation.shard_windows import (
    enumerate_window_names,
    hash_shard,
    load_shard_file,
    partition_windows,
    write_shard_files,
)


def test_hash_shard_is_deterministic_and_in_range() -> None:
    names = [f"window_{i}" for i in range(1000)]
    # Two independent passes produce identical shard assignments.
    a = [hash_shard(n, 16) for n in names]
    b = [hash_shard(n, 16) for n in names]
    assert a == b
    assert all(0 <= s < 16 for s in a)


def test_partition_covers_all_inputs_exactly_once() -> None:
    names = [f"window_{i}" for i in range(500)]
    shards = partition_windows(names, num_shards=7)
    flat = sorted(n for s in shards for n in s)
    assert flat == sorted(names)


def test_partition_is_reasonably_balanced() -> None:
    # 10k windows across 16 shards -> expect ~625 each; tolerate +/- 50%.
    names = [f"w_{i}" for i in range(10_000)]
    shards = partition_windows(names, 16)
    sizes = [len(s) for s in shards]
    assert min(sizes) >= 400
    assert max(sizes) <= 850


def test_num_shards_one_returns_everything_in_shard_zero() -> None:
    names = [f"w_{i}" for i in range(50)]
    shards = partition_windows(names, 1)
    assert len(shards) == 1
    assert shards[0] == names


def test_write_and_load_shard_files(tmp_path: Path) -> None:
    ds_path = UPath(tmp_path / "ds")
    ds_path.mkdir()
    shards = [
        ["a", "b", "c"],
        [],
        ["d"],
    ]
    paths = write_shard_files(ds_path, "sentinel2_l2a", shards)
    assert len(paths) == 3
    assert paths[0].name == "shard_0000.txt"
    assert load_shard_file(paths[0]) == ["a", "b", "c"]
    assert load_shard_file(paths[1]) == []
    assert load_shard_file(paths[2]) == ["d"]


def test_enumerate_window_names(tmp_path: Path) -> None:
    ds_path = UPath(tmp_path / "ds")
    group_dir = ds_path / "windows" / "res_10"
    group_dir.mkdir(parents=True)
    for name in ["w_1", "w_2", "w_3"]:
        (group_dir / name).mkdir()
    # Non-dir entries (stray files) are ignored.
    (group_dir / "README.txt").write_text("hi")
    assert enumerate_window_names(ds_path, "res_10") == ["w_1", "w_2", "w_3"]
    assert enumerate_window_names(ds_path, "missing") == []

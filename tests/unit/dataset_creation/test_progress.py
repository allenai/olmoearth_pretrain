"""Unit tests for the progress / heartbeat module."""

from pathlib import Path

from upath import UPath

from olmoearth_pretrain.dataset_creation.progress import (
    HeartbeatWriter,
    format_description_line,
    format_rollup,
    read_all_heartbeats,
    read_heartbeat,
)


def _make_window_dirs(
    ds_path: Path, group: str, names: list[str], completed_layer: str
) -> None:
    """Create the on-disk shape rslearn's FileWindowStorage uses.

    For each completed window we `touch` the `completed` marker; the rest stay
    "in progress". This mirrors `FileWindowStorage.mark_layer_completed`.
    """
    for name in names:
        layer_dir = ds_path / "windows" / group / name / "layers" / completed_layer
        layer_dir.mkdir(parents=True, exist_ok=True)


def _mark_completed(
    ds_path: Path, group: str, names: list[str], layer_name: str
) -> None:
    for name in names:
        (
            ds_path / "windows" / group / name / "layers" / layer_name / "completed"
        ).touch()


def test_heartbeat_counts_completed_markers(tmp_path: Path) -> None:
    ds_path = tmp_path / "ds"
    hb_dir = tmp_path / "heartbeats"
    windows = [f"w_{i}" for i in range(5)]
    _make_window_dirs(ds_path, "res_10", windows, "worldcover")

    writer = HeartbeatWriter(
        heartbeat_dir=UPath(hb_dir),
        shard_id="shard-0",
        modality="worldcover",
        layer_names=["worldcover"],
        ds_path=UPath(ds_path),
        group="res_10",
        window_names=windows,
        interval_s=60.0,  # long: we'll exercise start/stop manually
    )
    writer.start()

    first = read_heartbeat(UPath(hb_dir), "shard-0")
    assert first is not None
    assert first.total == 5
    assert first.done == 0
    assert first.finished is False

    _mark_completed(ds_path, "res_10", windows[:3], "worldcover")

    writer.stop()
    final = read_heartbeat(UPath(hb_dir), "shard-0")
    assert final is not None
    assert final.total == 5
    assert final.done == 3
    assert final.finished is True


def test_context_manager_captures_exceptions(tmp_path: Path) -> None:
    ds_path = tmp_path / "ds"
    hb_dir = tmp_path / "heartbeats"
    (ds_path / "windows" / "res_10").mkdir(parents=True)
    try:
        with HeartbeatWriter(
            heartbeat_dir=UPath(hb_dir),
            shard_id="shard-0",
            modality="landsat",
            layer_names=["landsat"],
            ds_path=UPath(ds_path),
            group="res_10",
            window_names=[],
            interval_s=60.0,
        ):
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    hb = read_heartbeat(UPath(hb_dir), "shard-0")
    assert hb is not None
    assert hb.finished is True
    assert hb.error is not None
    assert "RuntimeError" in hb.error
    assert "boom" in hb.error


def test_multi_layer_all_must_complete(tmp_path: Path) -> None:
    """A window is only 'done' when every named layer is completed."""
    ds_path = tmp_path / "ds"
    hb_dir = tmp_path / "heartbeats"
    windows = ["w_a", "w_b", "w_c"]
    layers = ["sentinel2_mo01", "sentinel2_mo02", "sentinel2_mo03"]
    for w in windows:
        for layer in layers:
            (ds_path / "windows" / "res_10" / w / "layers" / layer).mkdir(parents=True)

    # w_a: all layers done; w_b: only 2/3; w_c: none.
    for layer in layers:
        (ds_path / "windows" / "res_10" / "w_a" / "layers" / layer / "completed").touch()
    for layer in layers[:2]:
        (ds_path / "windows" / "res_10" / "w_b" / "layers" / layer / "completed").touch()

    with HeartbeatWriter(
        heartbeat_dir=UPath(hb_dir),
        shard_id="shard-0",
        modality="sentinel2_l2a",
        layer_names=layers,
        ds_path=UPath(ds_path),
        group="res_10",
        window_names=windows,
        interval_s=60.0,
    ):
        pass

    hb = read_heartbeat(UPath(hb_dir), "shard-0")
    assert hb is not None
    assert hb.total == 3
    assert hb.done == 1  # only w_a has all 3 layers done


def test_read_all_and_format(tmp_path: Path) -> None:
    ds_path = tmp_path / "ds"
    hb_dir = tmp_path / "heartbeats"
    windows_a = [f"a_{i}" for i in range(2)]
    windows_b = [f"b_{i}" for i in range(4)]
    _make_window_dirs(ds_path, "res_10", windows_a, "sentinel2")
    _make_window_dirs(ds_path, "res_10", windows_b, "sentinel1")
    _mark_completed(ds_path, "res_10", windows_a, "sentinel2")
    _mark_completed(ds_path, "res_10", windows_b[:1], "sentinel1")

    for shard_id, modality, layer, names in [
        ("s2-shard-0", "sentinel2_l2a", "sentinel2", windows_a),
        ("s1-shard-0", "sentinel1", "sentinel1", windows_b),
    ]:
        with HeartbeatWriter(
            heartbeat_dir=UPath(hb_dir),
            shard_id=shard_id,
            modality=modality,
            layer_names=[layer],
            ds_path=UPath(ds_path),
            group="res_10",
            window_names=names,
            interval_s=60.0,
        ):
            pass

    hbs = read_all_heartbeats(UPath(hb_dir))
    assert len(hbs) == 2
    assert {h.shard_id for h in hbs} == {"s2-shard-0", "s1-shard-0"}

    lines = [format_description_line(h) for h in hbs]
    assert any("sentinel2_l2a/s2-shard-0: 2/2 (100.0%) DONE" in ln for ln in lines)
    assert any("sentinel1/s1-shard-0: 1/4 (25.0%) DONE" in ln for ln in lines)

    rollup = format_rollup(hbs)
    assert "Overall: 3/6 (50.0%) across 2 shards" in rollup

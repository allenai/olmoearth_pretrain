"""Tests for shared rslearn-to-OlmoEarth converter runner helpers."""

import sys
from collections.abc import Iterator
from typing import Any

from pytest import MonkeyPatch
from upath import UPath

from olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth import runner


class _FakeDataset:
    instances: list["_FakeDataset"] = []

    def __init__(self, path: UPath) -> None:
        self.path = path
        self.load_windows_kwargs: dict[str, Any] | None = None
        self.instances.append(self)

    def load_windows(self, **kwargs: Any) -> list[str]:
        self.load_windows_kwargs = kwargs
        return ["window-a", "window-b"]


def test_make_window_converter_jobs_omits_groups_when_unset() -> None:
    """No-group converters should keep the original load_windows call shape."""
    dataset = _FakeDataset(UPath("/tmp/ds"))

    runner.make_window_converter_jobs(dataset, UPath("/tmp/out"), workers=2)

    assert dataset.load_windows_kwargs == {
        "workers": 2,
        "show_progress": True,
    }


def test_run_window_converter_uses_standard_cli_and_pool(
    monkeypatch: MonkeyPatch,
) -> None:
    """The CLI runner should build jobs and dispatch them through star_imap."""
    calls: dict[str, Any] = {}

    class FakePool:
        def __init__(self, workers: int) -> None:
            calls["pool_workers"] = workers

        def __enter__(self) -> "FakePool":
            calls["pool_entered"] = True
            return self

        def __exit__(self, *exc_info: object) -> None:
            calls["pool_exited"] = True

    def fake_star_imap_unordered(
        pool: FakePool,
        converter: runner.WindowConverter,
        jobs: list[dict[str, Any]],
    ) -> list[str]:
        calls["star_pool"] = pool
        calls["converter"] = converter
        calls["jobs"] = jobs
        return ["ok" for _ in jobs]

    def fake_tqdm(outputs: list[str], *, total: int) -> Iterator[str]:
        calls["tqdm_total"] = total
        return iter(outputs)

    def converter(window: object, olmoearth_path: UPath) -> None:
        pass

    _FakeDataset.instances.clear()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "converter",
            "--ds_path",
            "/tmp/ds",
            "--olmoearth_path",
            "/tmp/out",
            "--workers",
            "3",
        ],
    )
    monkeypatch.setattr(runner, "Dataset", _FakeDataset)
    monkeypatch.setattr(
        runner.multiprocessing,
        "set_start_method",
        lambda method: calls.setdefault("start_method", method),
    )
    monkeypatch.setattr(runner.multiprocessing, "Pool", FakePool)
    monkeypatch.setattr(runner, "star_imap_unordered", fake_star_imap_unordered)
    monkeypatch.setattr(runner.tqdm, "tqdm", fake_tqdm)

    runner.run_window_converter(converter, groups=["res_10"])

    assert calls["start_method"] == "forkserver"
    assert calls["pool_workers"] == 3
    assert calls["pool_entered"] is True
    assert calls["pool_exited"] is True
    assert calls["converter"] is converter
    assert calls["tqdm_total"] == 2
    assert calls["jobs"] == [
        {"window": "window-a", "olmoearth_path": UPath("/tmp/out")},
        {"window": "window-b", "olmoearth_path": UPath("/tmp/out")},
    ]
    assert _FakeDataset.instances[0].path == UPath("/tmp/ds")
    assert _FakeDataset.instances[0].load_windows_kwargs == {
        "workers": 3,
        "show_progress": True,
        "groups": ["res_10"],
    }

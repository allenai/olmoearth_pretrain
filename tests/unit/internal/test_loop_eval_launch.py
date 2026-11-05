"""Tests for the in-loop Beaker eval launcher."""

from pathlib import Path

import pytest

from olmoearth_pretrain.internal import loop_eval_launch


def test_eval_job_targets_every_cluster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The launcher command lists all clusters, not just the first."""
    (tmp_path / "step80000").mkdir()
    launched: list[list[str]] = []
    monkeypatch.setattr(
        loop_eval_launch.subprocess,
        "Popen",
        lambda cmd, **kwargs: launched.append(cmd),
    )
    assert loop_eval_launch.launch_checkpoint_eval_job(
        module_path="scripts/x.py",
        checkpoint_dir=str(tmp_path),
        step=80000,
        clusters=["ai2/jupiter", "ai2/ceres"],
        run_name="run_eval_step80000",
        priority="urgent",
    )
    (cmd,) = launched
    assert "ai2/jupiter" in cmd
    assert "--launch.clusters=[ai2/jupiter,ai2/ceres]" in cmd

"""Live progress tracking for distributed dataset-creation Beaker tasks.

Design
------
Each shard task writes a tiny JSON `heartbeat.json` to weka every
`interval_s` seconds. The orchestrator polls these files and folds them into
Beaker experiment descriptions so the state of a multi-shard job is visible
from the Beaker UI in near-real-time. No Beaker creds are needed in the worker.

How completion is counted
-------------------------
rslearn marks a window/layer materialized by `touch`-ing a file named
``completed`` under::

    {ds_path}/windows/{group}/{window_name}/layers/{layer_name}[.group_idx]/completed

(see rslearn/dataset/storage/file.py -> mark_layer_completed).

We count those markers for the windows in the shard to report "X / N done".
Progress counting runs on a fixed cadence in a background thread so the main
`rslearn` subprocess is unaffected.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from upath import UPath

logger = logging.getLogger(__name__)


HEARTBEAT_FILENAME = "heartbeat.json"


@dataclass
class Heartbeat:
    """The JSON payload each worker writes to its heartbeat file.

    ``layer_names`` may list one or many layers; a window counts as `done` only
    if every named layer has its completion marker present. Stored as a list so
    multi-layer modalities (Sentinel-2 has 13 monthly layers) get accurate %
    reporting instead of overcounting.
    """

    shard_id: str
    modality: str
    layer_names: list[str]
    total: int
    done: int
    updated_at: str  # ISO-8601 UTC, seconds precision
    started_at: str
    finished: bool = False
    error: str | None = None

    @property
    def pct(self) -> float:
        return 0.0 if self.total == 0 else 100.0 * self.done / self.total


def heartbeat_path(heartbeat_dir: UPath, shard_id: str) -> UPath:
    """Stable path for a shard's heartbeat file."""
    return heartbeat_dir / f"{shard_id}.{HEARTBEAT_FILENAME}"


def _count_completed(
    ds_path: UPath,
    group: str,
    window_names: list[str],
    layer_names: list[str],
) -> int:
    """Count windows with the ``completed`` marker for EVERY layer in ``layer_names``.

    The marker path is what rslearn writes in
    ``FileWindowStorage.mark_layer_completed`` (``windows/{group}/{name}/layers/{layer}/completed``).
    A window counts as done only when all supplied layers are completed, so
    multi-layer modalities report accurate overall progress.
    """
    if not layer_names:
        return 0
    windows_root = ds_path / "windows" / group
    done = 0
    for name in window_names:
        window_dir = windows_root / name / "layers"
        if all((window_dir / layer / "completed").exists() for layer in layer_names):
            done += 1
    return done


class HeartbeatWriter:
    """Background thread that periodically emits a shard's progress to weka.

    Usage::

        with HeartbeatWriter(...) as hb:
            subprocess.run(["rslearn", "dataset", "materialize", ...])
            # writer keeps pulsing until the `with` block exits.

    A final heartbeat is written on exit (including on exception), marking
    `finished=True` and optionally capturing an error message.
    """

    def __init__(
        self,
        *,
        heartbeat_dir: UPath,
        shard_id: str,
        modality: str,
        layer_names: list[str],
        ds_path: UPath,
        group: str,
        window_names: list[str],
        interval_s: float = 30.0,
    ) -> None:
        self.shard_id = shard_id
        self.modality = modality
        self.layer_names = layer_names
        self.ds_path = ds_path
        self.group = group
        self.window_names = window_names
        self.interval_s = interval_s
        self._path = heartbeat_path(heartbeat_dir, shard_id)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._started_at = datetime.now(UTC).isoformat(timespec="seconds")
        self._error: str | None = None

    def start(self) -> None:
        """Begin the heartbeat loop and emit an initial heartbeat immediately."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._write(finished=False)
        self._thread.start()

    def stop(self, error: str | None = None) -> None:
        """Stop the loop and write one final heartbeat with `finished=True`."""
        self._error = error
        self._stop.set()
        self._thread.join(timeout=self.interval_s + 5)
        # Final heartbeat reflects terminal state.
        self._write(finished=True)

    def __enter__(self) -> "HeartbeatWriter":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        err = None if exc is None else f"{exc_type.__name__}: {exc}"
        self.stop(error=err)

    def _loop(self) -> None:
        while not self._stop.wait(timeout=self.interval_s):
            try:
                self._write(finished=False)
            except Exception as e:  # pragma: no cover - transient fs issues
                logger.warning("heartbeat write failed: %s", e)

    def _write(self, *, finished: bool) -> None:
        done = _count_completed(
            self.ds_path, self.group, self.window_names, self.layer_names
        )
        hb = Heartbeat(
            shard_id=self.shard_id,
            modality=self.modality,
            layer_names=self.layer_names,
            total=len(self.window_names),
            done=done,
            updated_at=datetime.now(UTC).isoformat(timespec="seconds"),
            started_at=self._started_at,
            finished=finished,
            error=self._error,
        )
        # Best-effort atomic write: write to tmp then rename.
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(hb)))
        tmp.rename(self._path)


def read_heartbeat(heartbeat_dir: UPath, shard_id: str) -> Heartbeat | None:
    """Load a shard's latest heartbeat, or None if it hasn't been written yet."""
    p = heartbeat_path(heartbeat_dir, shard_id)
    if not p.exists():
        return None
    try:
        return Heartbeat(**json.loads(p.read_text()))
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        # Likely a concurrent write mid-rename — treat as "no heartbeat yet".
        logger.debug("failed to parse heartbeat at %s: %s", p, e)
        return None


def read_all_heartbeats(heartbeat_dir: UPath) -> list[Heartbeat]:
    """Load every shard's latest heartbeat from `heartbeat_dir`."""
    if not heartbeat_dir.exists():
        return []
    out: list[Heartbeat] = []
    for p in heartbeat_dir.iterdir():
        if not p.name.endswith(f".{HEARTBEAT_FILENAME}"):
            continue
        shard_id = p.name[: -len(f".{HEARTBEAT_FILENAME}")]
        hb = read_heartbeat(heartbeat_dir, shard_id)
        if hb is not None:
            out.append(hb)
    return out


def format_description_line(hb: Heartbeat, *, stall_threshold_s: float = 300.0) -> str:
    """One-line human-readable status for a single shard.

    Example::

        sentinel2_l2a/shard-0: 1234/10000 (12.3%) running
        sentinel2_l2a/shard-1: 10000/10000 (100.0%) DONE
        landsat/shard-2: 450/10000 (4.5%) STALLED for 6m
    """
    tag = f"{hb.modality}/{hb.shard_id}"
    pct = f"{hb.pct:.1f}%"
    suffix = _status_suffix(hb, stall_threshold_s=stall_threshold_s)
    return f"{tag}: {hb.done}/{hb.total} ({pct}) {suffix}"


def _status_suffix(hb: Heartbeat, *, stall_threshold_s: float) -> str:
    if hb.finished:
        return "FAILED: " + hb.error if hb.error else "DONE"
    age = _age_seconds(hb.updated_at)
    if age is not None and age > stall_threshold_s:
        return f"STALLED for {_fmt_duration(age)}"
    return "running"


def _age_seconds(iso_ts: str) -> float | None:
    try:
        then = datetime.fromisoformat(iso_ts)
    except ValueError:
        return None
    now = datetime.now(UTC)
    return (now - then).total_seconds()


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h{(seconds % 3600) // 60}m"


def format_rollup(heartbeats: list[Heartbeat]) -> str:
    """Multi-line summary of all shards, suitable for the parent experiment's description."""
    if not heartbeats:
        return "(no shards reporting yet)"
    # Sort for stable ordering in the UI.
    heartbeats = sorted(heartbeats, key=lambda h: (h.modality, h.shard_id))
    total = sum(h.total for h in heartbeats)
    done = sum(h.done for h in heartbeats)
    pct = 0.0 if total == 0 else 100.0 * done / total
    header = f"Overall: {done}/{total} ({pct:.1f}%) across {len(heartbeats)} shards"
    lines = [header] + [format_description_line(h) for h in heartbeats]
    return "\n".join(lines)


def load_window_names(shard_file: Path | UPath) -> list[str]:
    """Read window names from a shard file (one per line, blank/# lines skipped).

    Companion to the shard writer implemented in `shard_windows.py`.
    """
    text = shard_file.read_text() if isinstance(shard_file, Path) else shard_file.read_text()
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.startswith("#")
    ]

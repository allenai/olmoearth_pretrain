"""Diagnostics for eval embedding loops that stop making progress.

A daemon thread watches the time since the last ``beat()``. When it exceeds
``timeout_s`` it logs the Python stacks of the main process, sends SIGUSR1 to
the DataLoader worker processes (which dump their own stacks to stderr once
``register_worker_stack_dump`` ran in them), and logs each worker's kernel
state and wait channel, so an IO hang (state D) shows up in the job log.
"""

import faulthandler
import logging
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def register_worker_stack_dump() -> None:
    """Dump this process's Python stacks to stderr on SIGUSR1."""
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)


def _read(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except OSError as e:
        return f"<{e.__class__.__name__}>"


def _proc_state(pid: int) -> str:
    status = _read(f"/proc/{pid}/status")
    state = next(
        (line for line in status.splitlines() if line.startswith("State:")), "?"
    )
    rss = next((line for line in status.splitlines() if line.startswith("VmRSS:")), "?")
    return f"{state} {rss} wchan={_read(f'/proc/{pid}/wchan')}"


def mem_summary() -> str:
    """Return this process's RSS and the host's MemAvailable."""
    rss = next(
        (
            line.split(":", 1)[1].strip()
            for line in _read("/proc/self/status").splitlines()
            if line.startswith("VmRSS:")
        ),
        "?",
    )
    avail = next(
        (
            line.split(":", 1)[1].strip()
            for line in _read("/proc/meminfo").splitlines()
            if line.startswith("MemAvailable:")
        ),
        "?",
    )
    return f"rss={rss} mem_available={avail}"


class HangWatchdog:
    """Logs diagnostics when ``beat()`` has not been called for ``timeout_s``."""

    def __init__(self, name: str, timeout_s: float = 600.0) -> None:
        """Create a watchdog; call ``start()`` to arm it."""
        self.name = name
        self.timeout_s = timeout_s
        self._last = time.monotonic()
        self._last_label = "start"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "HangWatchdog":
        """Start the watcher thread."""
        self._last = time.monotonic()
        self._thread.start()
        return self

    def beat(self, label: str) -> None:
        """Record progress."""
        self._last = time.monotonic()
        self._last_label = label

    def stop(self) -> None:
        """Stop the watcher thread."""
        self._stop.set()

    def _run(self) -> None:
        fired_at = None
        while not self._stop.wait(30.0):
            idle = time.monotonic() - self._last
            if idle < self.timeout_s:
                fired_at = None
                continue
            # Re-dump every timeout_s while stalled.
            if fired_at is not None and time.monotonic() - fired_at < self.timeout_s:
                continue
            fired_at = time.monotonic()
            self._dump(idle)

    def _dump(self, idle: float) -> None:
        logger.warning(
            "[hang_watchdog %s] no progress for %.0fs after %s; %s",
            self.name,
            idle,
            self._last_label,
            mem_summary(),
        )
        logger.warning(
            "[hang_watchdog %s] main pid %d: %s",
            self.name,
            os.getpid(),
            _proc_state(os.getpid()),
        )
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        for child in mp.active_children():
            if child.pid is None:
                continue
            logger.warning(
                "[hang_watchdog %s] child %s pid %d: %s stack=%s",
                self.name,
                child.name,
                child.pid,
                _proc_state(child.pid),
                _read(f"/proc/{child.pid}/stack").replace("\n", " | "),
            )
            try:
                os.kill(child.pid, signal.SIGUSR1)
            except OSError as e:
                logger.warning("SIGUSR1 to %d failed: %s", child.pid, e)
            # Give each worker time to finish its dump so they don't interleave.
            time.sleep(2.0)
        sys.stderr.flush()

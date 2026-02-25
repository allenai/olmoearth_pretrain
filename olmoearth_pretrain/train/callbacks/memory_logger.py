"""Callback that logs host memory stats to a CSV file."""

import csv
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional, TextIO

import psutil

from olmo_core.train.callbacks.callback import Callback

log = logging.getLogger(__name__)


@dataclass
class MemoryLoggerCallback(Callback):
    """Logs system and process memory to a CSV file every ``interval`` steps.

    Columns written:
        step, wall_time, sys_mem_used_pct, sys_mem_avail_gb,
        proc_rss_gb, proc_vms_gb, num_child_procs
    """

    interval: int = 10
    log_path: str = "memory_log.csv"

    _file: Optional[TextIO] = field(default=None, init=False, repr=False)
    _writer: Optional[csv.writer] = field(default=None, init=False, repr=False)

    def pre_train(self):
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        if rank != 0:
            return
        self._file = open(self.log_path, "w", newline="", buffering=1)
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            "step", "wall_time",
            "sys_mem_used_pct", "sys_mem_avail_gb",
            "proc_rss_gb", "proc_vms_gb", "num_child_procs",
        ])
        log.info("MemoryLoggerCallback writing to %s", self.log_path)

    def post_step(self):
        if self._writer is None:
            return
        if self.step % self.interval != 0:
            return

        vm = psutil.virtual_memory()
        proc = psutil.Process(os.getpid())
        mem = proc.memory_info()

        self._writer.writerow([
            self.step,
            f"{time.time():.2f}",
            f"{vm.percent:.1f}",
            f"{vm.available / (1024**3):.2f}",
            f"{mem.rss / (1024**3):.2f}",
            f"{mem.vms / (1024**3):.2f}",
            len(proc.children(recursive=True)),
        ])

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None

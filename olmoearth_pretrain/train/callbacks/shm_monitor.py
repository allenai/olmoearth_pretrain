"""Callback that monitors /dev/shm and process memory usage."""

import logging
import os
import shutil
from dataclasses import dataclass

import psutil

from olmo_core.train.callbacks.callback import Callback

log = logging.getLogger(__name__)


@dataclass
class ShmMonitorCallback(Callback):
    """Logs /dev/shm and per-process memory stats every ``interval`` steps."""

    interval: int = 10
    enabled: bool = True

    def post_step(self):
        if not self.enabled:
            return
        if self.step % self.interval != 0:
            return
        if int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) != 0:
            return

        try:
            usage = shutil.disk_usage("/dev/shm")
            used_gb = usage.used / (1024**3)
            total_gb = usage.total / (1024**3)
            pct = (usage.used / usage.total) * 100

            self.trainer.record_metric("shm/used_gb", used_gb)
            self.trainer.record_metric("shm/total_gb", total_gb)
            self.trainer.record_metric("shm/used_pct", pct)
        except OSError:
            pass

        try:
            proc = psutil.Process(os.getpid())
            mem = proc.memory_info()
            self.trainer.record_metric("mem/main_rss_gb", mem.rss / (1024**3))
            self.trainer.record_metric("mem/main_vms_gb", mem.vms / (1024**3))

            children = proc.children(recursive=True)
            child_rss = sum(c.memory_info().rss for c in children) / (1024**3)
            self.trainer.record_metric("mem/children_rss_gb", child_rss)
            self.trainer.record_metric("mem/total_rss_gb", mem.rss / (1024**3) + child_rss)
            self.trainer.record_metric("mem/num_children", len(children))

            log.info(
                f"mem: main_rss={mem.rss / (1024**3):.2f}GB, "
                f"children_rss={child_rss:.2f}GB ({len(children)} procs), "
                f"shm={used_gb:.2f}/{total_gb:.2f}GB"
            )
        except (psutil.NoSuchProcess, OSError):
            pass

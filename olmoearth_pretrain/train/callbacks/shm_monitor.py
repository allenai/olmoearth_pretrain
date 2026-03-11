"""Callback that monitors /dev/shm, process memory, and CUDA host memory."""

import logging
import os
import shutil
from dataclasses import dataclass

import psutil
import torch

from olmo_core.train.callbacks.callback import Callback

log = logging.getLogger(__name__)


@dataclass
class ShmMonitorCallback(Callback):
    """Logs /dev/shm, per-process memory, and CUDA host memory stats every ``interval`` steps."""

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
            used_gb = 0
            total_gb = 0

        try:
            proc = psutil.Process(os.getpid())
            mem = proc.memory_info()
            main_rss = mem.rss / (1024**3)
            self.trainer.record_metric("mem/main_rss_gb", main_rss)
            self.trainer.record_metric("mem/main_vms_gb", mem.vms / (1024**3))

            children = proc.children(recursive=True)
            child_rss = sum(c.memory_info().rss for c in children) / (1024**3)
            self.trainer.record_metric("mem/children_rss_gb", child_rss)
            self.trainer.record_metric("mem/total_rss_gb", main_rss + child_rss)
            self.trainer.record_metric("mem/num_children", len(children))
        except (psutil.NoSuchProcess, OSError):
            main_rss = 0
            child_rss = 0

        # /proc/self/status breaks RSS into anonymous, file-backed, and shared
        try:
            to_gb = 1 / (1024 * 1024)  # /proc/self/status reports in kB
            status = {}
            with open("/proc/self/status") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        status[parts[0].rstrip(":")] = parts[1]

            rss_anon = int(status.get("RssAnon", 0)) * to_gb
            rss_file = int(status.get("RssFile", 0)) * to_gb
            rss_shmem = int(status.get("RssShmem", 0)) * to_gb
            vm_swap = int(status.get("VmSwap", 0)) * to_gb

            self.trainer.record_metric("mem/rss_anon_gb", rss_anon)
            self.trainer.record_metric("mem/rss_file_gb", rss_file)
            self.trainer.record_metric("mem/rss_shmem_gb", rss_shmem)
            self.trainer.record_metric("mem/vm_swap_gb", vm_swap)

            log.info(
                f"mem: rss={main_rss:.2f}GB (anon={rss_anon:.2f}, file={rss_file:.2f}, "
                f"shmem={rss_shmem:.2f}), children={child_rss:.2f}GB"
            )
        except Exception as e:
            log.warning(f"Failed to read /proc/self/status: {e}")

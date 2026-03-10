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

        if torch.cuda.is_available():
            try:
                stats = torch.cuda.memory_stats()
                to_gb = 1 / (1024**3)

                # GPU device memory
                self.trainer.record_metric(
                    "cuda/allocated_gb", stats.get("allocated_bytes.all.current", 0) * to_gb
                )
                self.trainer.record_metric(
                    "cuda/reserved_gb", stats.get("reserved_bytes.all.current", 0) * to_gb
                )
                self.trainer.record_metric(
                    "cuda/active_gb", stats.get("active_bytes.all.current", 0) * to_gb
                )

                # Pinned (host) memory managed by PyTorch's caching allocator
                pinned_current = stats.get("pinned_bytes.all.current", 0) * to_gb
                pinned_peak = stats.get("pinned_bytes.all.peak", 0) * to_gb
                self.trainer.record_metric("cuda/pinned_current_gb", pinned_current)
                self.trainer.record_metric("cuda/pinned_peak_gb", pinned_peak)

                # Number of pinned memory allocations
                pinned_allocs = stats.get("pinned_bytes.all.allocated", 0)
                pinned_frees = stats.get("pinned_bytes.all.freed", 0)
                self.trainer.record_metric("cuda/pinned_alloc_count", pinned_allocs)
                self.trainer.record_metric("cuda/pinned_free_count", pinned_frees)
                self.trainer.record_metric("cuda/pinned_live_count", pinned_allocs - pinned_frees)

                log.info(
                    f"mem: rss={main_rss:.2f}GB, children={child_rss:.2f}GB, "
                    f"cuda_alloc={stats.get('allocated_bytes.all.current', 0) * to_gb:.2f}GB, "
                    f"pinned={pinned_current:.4f}GB (peak={pinned_peak:.4f}GB, "
                    f"live_allocs={pinned_allocs - pinned_frees})"
                )
            except Exception:
                pass

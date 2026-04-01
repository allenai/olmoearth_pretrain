"""Callback that monitors /dev/shm, process memory, and CUDA host memory."""

import gc
import logging
import os
import shutil
import tracemalloc
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
    # Set to True to enable tracemalloc-based Python heap profiling (has overhead)
    tracemalloc_enabled: bool = False
    tracemalloc_interval: int = 50
    tracemalloc_top_n: int = 20
    # Set to True to log counts of Python objects by type
    objcount_enabled: bool = False
    objcount_interval: int = 50
    objcount_top_n: int = 20

    def pre_train(self):
        # Internal profiling state (not part of config, set lazily)
        self._tracemalloc_snapshot = None
        self._objcounts_prev: dict = {}
        if self.tracemalloc_enabled and int(os.environ.get("RANK", "0")) == 0:
            tracemalloc.start(25)
            log.info("tracemalloc started")

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

        # tracemalloc: log top Python allocation sites and deltas
        if not hasattr(self, "_tracemalloc_snapshot"):
            self._tracemalloc_snapshot = None
            self._objcounts_prev = {}
        if self.tracemalloc_enabled and tracemalloc.is_tracing():
            if self.step % self.tracemalloc_interval == 0:
                try:
                    snapshot = tracemalloc.take_snapshot()
                    if self._tracemalloc_snapshot is not None:
                        stats = snapshot.compare_to(
                            self._tracemalloc_snapshot, "lineno"
                        )
                        top = sorted(stats, key=lambda s: s.size_diff, reverse=True)[
                            : self.tracemalloc_top_n
                        ]
                        lines = [
                            f"  +{s.size_diff/1024:.1f}KB ({s.count_diff:+d} objs) {s.traceback}"
                            for s in top
                            if s.size_diff > 0
                        ]
                        if lines:
                            log.info(
                                f"[tracemalloc step={self.step}] top growing allocations:\n"
                                + "\n".join(lines)
                            )
                    else:
                        top = snapshot.statistics("lineno")[: self.tracemalloc_top_n]
                        lines = [
                            f"  {s.size/1024:.1f}KB ({s.count} objs) {s.traceback}"
                            for s in top
                        ]
                        log.info(
                            f"[tracemalloc step={self.step}] top allocations (baseline):\n"
                            + "\n".join(lines)
                        )
                    self._tracemalloc_snapshot = snapshot
                except Exception as e:
                    log.warning(f"tracemalloc failed: {e}")

        # objcount: log counts of Python objects by type and detect growing types
        if self.objcount_enabled and self.step % self.objcount_interval == 0:
            try:
                counts: dict[str, int] = {}
                for obj in gc.get_objects():
                    t = type(obj).__name__
                    counts[t] = counts.get(t, 0) + 1
                # Find types whose count grew since last check
                if self._objcounts_prev:
                    deltas = {
                        t: counts.get(t, 0) - self._objcounts_prev.get(t, 0)
                        for t in set(counts) | set(self._objcounts_prev)
                    }
                    growing = sorted(
                        ((t, d) for t, d in deltas.items() if d > 0),
                        key=lambda x: -x[1],
                    )[: self.objcount_top_n]
                    if growing:
                        log.info(
                            f"[objcount step={self.step}] growing object types:\n"
                            + "\n".join(f"  {t}: +{d}" for t, d in growing)
                        )
                else:
                    top = sorted(counts.items(), key=lambda x: -x[1])[
                        : self.objcount_top_n
                    ]
                    log.info(
                        f"[objcount step={self.step}] baseline object counts:\n"
                        + "\n".join(f"  {t}: {n}" for t, n in top)
                    )
                self._objcounts_prev = counts
            except Exception as e:
                log.warning(f"objcount failed: {e}")

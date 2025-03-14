"""Data worker memory monitor callback."""

import json
import logging
import os
import pickle
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import psutil
import torch
from olmo_core.train.callbacks.callback import Callback
from tabulate import tabulate

# Memory Monitor adapted from https://github.com/ppwwyyxx/RAM-multiprocess-dataloader/blob/main/common.py

logger = logging.getLogger(__name__)


def get_mem_info(pid: int) -> dict[str, int]:
    res = defaultdict(int)
    for mmap in psutil.Process(pid).memory_maps():
        res["rss"] += mmap.rss
        res["pss"] += mmap.pss
        res["uss"] += mmap.private_clean + mmap.private_dirty
        res["shared"] += mmap.shared_clean + mmap.shared_dirty
        if mmap.path.startswith("/"):
            res["shared_file"] += mmap.shared_clean + mmap.shared_dirty
    return res


class MemoryMonitor:
    def __init__(self, pids: list[int] | None = None):
        if pids is None:
            pids = [os.getpid()]
        self.pids = pids

    def add_pid(self, pid: int):
        assert pid not in self.pids
        self.pids.append(pid)

    def _refresh(self):
        self.data = {pid: get_mem_info(pid) for pid in self.pids}
        return self.data

    def table(self) -> str:
        self._refresh()
        table = []
        keys = list(list(self.data.values())[0].keys())
        now = str(int(time.perf_counter() % 1e5))
        for pid, data in self.data.items():
            table.append((now, str(pid)) + tuple(self.format(data[k]) for k in keys))
        return tabulate(table, headers=["time", "PID"] + keys)

    def str(self):
        self._refresh()
        keys = list(list(self.data.values())[0].keys())
        res = []
        for pid in self.pids:
            s = f"PID={pid}"
            for k in keys:
                v = self.format(self.data[pid][k])
                s += f", {k}={v}"
            res.append(s)
        return "\n".join(res)

    @staticmethod
    def format(size: int) -> str:
        for unit in ("", "K", "M", "G"):
            if size < 1024:
                break
            size /= 1024.0
        return "%.1f%s" % (size, unit)


@dataclass
class DataWorkerMemoryMonitor(Callback):
    """Data worker memory monitor callback."""

    set_pids_for_epoch: bool = False

    def pre_epoch(self):
        self.set_pids_for_epoch = False
        self.memory_monitor = MemoryMonitor()
        # I want to grab the dataloader Pids from the trainer and do memroy monitor
        # Likely should see if this is the issue before going to solve it

    def pre_load_batch(self):
        ## Only works for num workers > 1
        iterator = self.trainer.data_loader._iterator

        if self.set_pids_for_epoch:
            [self.memory_monitor.add_pid(w.pid) for w in iterator._workers]
            self.set_pids_for_epoch = True

    def pre_step(self, batch):
        logger.info(self.memory_monitor.str())

"""Callback that monitors /dev/shm usage and logs it as a training metric."""

import logging
import os
import shutil
from dataclasses import dataclass

from olmo_core.train.callbacks.callback import Callback

log = logging.getLogger(__name__)


@dataclass
class ShmMonitorCallback(Callback):
    """Logs /dev/shm usage every ``interval`` steps as a training metric."""

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

            log.info(f"/dev/shm: {used_gb:.2f}/{total_gb:.2f} GB ({pct:.1f}%)")
        except OSError:
            pass

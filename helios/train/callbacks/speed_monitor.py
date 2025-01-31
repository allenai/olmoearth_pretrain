"""Speed monitor callback for the trainer for Helios."""

import time
from typing import Any

from olmo_core.train.callbacks.speed_monitor import SpeedMonitorCallback


# FOR NOW FORGET ABOUT TOKENS AND STUFF
class HeliosSpeedMonitorCallback(SpeedMonitorCallback):
    """Speed monitor callback for the trainer for Helios."""

    def pre_step(self, batch: Any) -> None:
        """Pre-step callback for the speed monitor."""
        self._batch_load_time = time.perf_counter() - self._batch_load_start

    def post_step(self) -> None:
        """Post-step callback for the speed monitor."""
        counter = time.perf_counter()
        self.trainer.record_metric(
            "throughput/device/data loading (s)", self._batch_load_time
        )
        self._first_step: bool
        if self._first_step:
            # Now we can start recording.
            self._total_steps = 0
            self._total_tokens = 0
            self._start_time = counter
            self._step_last_logged = counter
            self._first_step = False
            return

        step_time = counter - self._step_last_logged
        total_time = counter - self._start_time
        self._step_last_logged = counter

        bps = 1 / step_time
        bps_avg = self._total_steps / total_time
        data_pct = 100 * self._batch_load_time / step_time

        self.trainer.record_metric(
            "throughput/total tokens", self.trainer.global_train_tokens_seen
        )
        self.trainer.record_metric("throughput/device/data loading (%)", data_pct)
        self.trainer.record_metric("throughput/device/BPS", bps)
        self.trainer.record_metric("throughput/device/BPS (actual avg)", bps_avg)

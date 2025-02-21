"""Speed monitor callback for the trainer for Helios."""

import time
from typing import Any

from olmo_core.train.callbacks.speed_monitor import SpeedMonitorCallback


# FOR NOW FORGET ABOUT TOKENS AND STUFF
# TODO: update this for V2
class HeliosSpeedMonitorCallback(SpeedMonitorCallback):
    """Speed monitor callback for the trainer for Helios."""

    def pre_step(self, batch: Any) -> None:
        """Pre-step callback for the speed monitor."""
        self._batch_load_time = time.perf_counter() - self._batch_load_start
        # We right now don't know how many masked and unmasked tokens are avalaible at this point
        # May want to save the values directly to the trainer to update global train tokens seen
        # also may want to do more speciifc clacls

        ## These tokens might be needed to go in pre_optim_step
        self._step_tokens = batch["input_ids"].numel() // self._parallel_degree
        self._total_steps += 1
        self._total_tokens += self._step_tokens

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
        tps = self._step_tokens / step_time
        tps_avg = self._total_tokens / total_time

        self.trainer.record_metric(
            "throughput/total tokens", self.trainer.global_train_tokens_seen
        )
        self.trainer.record_metric("throughput/device/TPS", tps)
        self.trainer.record_metric("throughput/device/TPS (actual avg)", tps_avg)
        self.trainer.record_metric("throughput/device/data loading (%)", data_pct)
        self.trainer.record_metric("throughput/device/BPS", bps)
        self.trainer.record_metric("throughput/device/BPS (actual avg)", bps_avg)

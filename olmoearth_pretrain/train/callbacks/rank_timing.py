"""Per-rank step timing, reduced across ranks (max / mean / min).

The speed monitor reports rank 0 only, where waiting for a slower rank shows up as
model or callback time at the next collective. This logs every rank's own data wait
(previous step's end to this step's start) and step time (start to end), so one
straggling rank (a high max with a low mean) can be told from all ranks being slow
together (max close to mean).
"""

import time
from dataclasses import dataclass
from typing import Any

from olmo_core.train.callbacks import Callback
from olmo_core.train.common import ReduceType


@dataclass
class RankTimingCallback(Callback):
    """Log ``rank_timing/{data_wait,step}_s_{max,mean,min}`` every step."""

    def pre_train(self) -> None:  # noqa: D102
        self._last_end: float | None = None
        self._start = 0.0

    def pre_step(self, batch: Any) -> None:  # noqa: D102
        self._start = time.perf_counter()
        if self._last_end is not None:
            self._record("data_wait", self._start - self._last_end)

    def post_step(self) -> None:  # noqa: D102
        self._last_end = time.perf_counter()
        self._record("step", self._last_end - self._start)

    def _record(self, name: str, seconds: float) -> None:
        record = self.trainer.record_metric
        record(f"rank_timing/{name}_s_max", seconds, ReduceType.max)
        record(f"rank_timing/{name}_s_mean", seconds, ReduceType.mean)
        # min across ranks = -max(-x)
        record(f"rank_timing/{name}_s_negmin", -seconds, ReduceType.max)

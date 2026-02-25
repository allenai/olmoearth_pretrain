"""Garbage collector callback with full gen2 collection support."""

import gc
import logging
from dataclasses import dataclass
from typing import Optional

from olmo_core.train.callbacks.garbage_collector import GarbageCollectorCallback

log = logging.getLogger(__name__)


@dataclass
class FullGCCallback(GarbageCollectorCallback):
    """Extends GarbageCollectorCallback to periodically run full gen2 collections.

    The upstream callback only runs gc.collect(1), which never frees gen2 objects.
    This subclass adds a full gc.collect() every ``full_gc_interval`` steps.
    """

    full_gc_interval: int = 50

    def post_step(self):
        if not self.enabled:
            return
        if self.step % self.gc_interval == 0:
            gc.collect(1)
        if self.step % self.full_gc_interval == 0:
            gc.collect()

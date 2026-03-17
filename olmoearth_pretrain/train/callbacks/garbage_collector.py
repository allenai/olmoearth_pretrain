"""Garbage collector callback with malloc_trim support."""

import ctypes
import gc
import logging
import platform
from dataclasses import dataclass

from olmo_core.train.callbacks.garbage_collector import GarbageCollectorCallback

log = logging.getLogger(__name__)

_libc = None
if platform.system() == "Linux":
    try:
        _libc = ctypes.CDLL("libc.so.6")
    except OSError:
        pass


@dataclass
class FullGCCallback(GarbageCollectorCallback):
    """Extends GarbageCollectorCallback with full gen2 collection and malloc_trim.

    glibc's malloc keeps freed memory in internal free lists instead of returning
    it to the OS, causing RSS to grow over time. Calling malloc_trim(0) forces
    glibc to release freed pages back to the OS.
    """

    full_gc_interval: int = 50
    malloc_trim_interval: int = 50

    def post_step(self):
        if not self.enabled:
            return
        if self.step % self.gc_interval == 0:
            gc.collect(1)
        if self.step % self.full_gc_interval == 0:
            gc.collect()
        if self.step % self.malloc_trim_interval == 0 and _libc is not None:
            _libc.malloc_trim(0)

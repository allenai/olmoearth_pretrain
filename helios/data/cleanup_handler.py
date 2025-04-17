"""Process that detaches and handles SIGTERM to cleanup a directory.

It is intended to work with Beaker experiments to cleanup the directory in case the
experiment fails or gets pre-empted.
"""

import os
import shutil
import signal
import sys
import time
from collections.abc import Callable
from typing import Any


def cleanup(tmp_dir: str) -> None:
    """Cleanup the specified temporary directory."""
    shutil.rmtree(tmp_dir, ignore_errors=True)


def get_cleanup_signal_handler(tmp_dir: str) -> Callable[[int, Any], None]:
    """Make a signal handler that cleans up the specified directory before exiting.

    This should be passed as the handler to signal.signal.

    Args:
        tmp_dir: the directory to delete when the signal is received.
    """

    def cleanup_signal_handler(signo: int, stack_frame: Any) -> None:
        print(f"cleanup_signal_handler: caught signal {signo}, cleaning up {tmp_dir}")
        cleanup(tmp_dir)
        sys.exit(1)

    return cleanup_signal_handler


if __name__ == "__main__":
    tmp_dir = sys.argv[1]
    print(f"setting up cleanup for {tmp_dir}")
    # Detach from main process.
    os.setsid()
    # Setup signal handler.
    signal.signal(signal.SIGTERM, get_cleanup_signal_handler(tmp_dir))
    # Poll parent process.
    while True:
        time.sleep(1)
        # If parent is gone, the pid should be 1 (init).
        if os.getppid() != 1:
            continue
        break
    print(f"cleaning up {tmp_dir} since parent process exited")
    cleanup(tmp_dir)

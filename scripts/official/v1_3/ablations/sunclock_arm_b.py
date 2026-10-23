"""Sun-clock test arm B: time_code=doy_sun. See ``sunclock_common.py``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sunclock_common import run  # noqa: E402

if __name__ == "__main__":
    run("doy_sun")

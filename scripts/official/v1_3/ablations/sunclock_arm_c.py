"""Sun-clock test arm C: time_code=doy. See ``sunclock_common.py``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sunclock_common import run  # noqa: E402

if __name__ == "__main__":
    run("doy")

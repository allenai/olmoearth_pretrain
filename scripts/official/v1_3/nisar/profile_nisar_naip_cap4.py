"""Profile nisar/nisar_naip_cap4.py (see profile_arm.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from profile_arm import run_profiled  # noqa: E402

if __name__ == "__main__":
    run_profiled("nisar/nisar_naip_cap4")
